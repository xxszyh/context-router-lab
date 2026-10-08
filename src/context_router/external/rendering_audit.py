"""Audit actual rendered evidence after selection, without a generator or label leakage."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from context_router.external.longmemeval import iter_instances, validate_instance
from context_router.external.longmemeval_answers import (
    INSTRUCTIONS,
    MemoryBudget,
    _context,
    dated_sessions,
    file_hash,
    generation_prompt,
    load_plan,
    object_hash,
    text_hash,
)
from context_router.external.longmemeval_rendering import FULL_HISTORY_RENDERING, index_session
from context_router.external.scale_qa import OrderingHits, paired_ordering_comparison


def _annotation(instance: dict[str, Any], gold: set[str]) -> tuple[str, dict[str, list[set[int]]]]:
    copies: dict[str, list[tuple[Any, ...]]] = {}
    sources = {}
    for sid, turns in zip(
        instance["haystack_session_ids"], instance["haystack_sessions"], strict=True
    ):
        if sid not in gold:
            continue
        labels = tuple(turn.get("has_answer") for turn in turns)
        if any(label is not None and type(label) is not bool for label in labels):
            raise ValueError("has_answer annotations must be booleans or absent")
        copies.setdefault(sid, []).append(labels)
        sources[sid] = (turns, labels)
    if any(any(copy != versions[0] for copy in versions) for versions in copies.values()):
        return "conflicting_duplicate_annotations", {}
    expected = {}
    for sid, (turns, labels) in sources.items():
        source = index_session(tuple((str(t.get("role", "unknown")), t["content"]) for t in turns))
        expected[sid] = [
            {
                i
                for i, word in enumerate(source.words)
                if start <= word.start() and word.end() <= end
            }
            for label, (start, end) in zip(labels, source.content_ranges, strict=True)
            if label is True
        ]
    if any(not turns or any(not words for words in turns) for turns in expected.values()):
        return "missing_gold_turn_annotations", expected
    return "complete", expected


def _summary(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    result = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        answerable = [row for row in selected if not row["expects_refusal"]]
        eligible = [row for row in answerable if row["annotation_status"] == "complete"]
        retrieved = [row for row in eligible if row["all_gold_sessions_retrieved"]]

        def mean(subset: list[dict[str, Any]], field: str) -> float | None:
            return statistics.mean(row[field] for row in subset) if subset else None

        result[arm] = {
            "questions": len(selected),
            "answerable_questions": len(answerable),
            "abstention_questions": len(selected) - len(answerable),
            "annotation_eligible_questions": len(eligible),
            "annotation_unknown_questions": len(answerable) - len(eligible),
            "all_gold_session_recall": mean(answerable, "all_gold_sessions_retrieved"),
            "all_annotated_turns_fully_retained": mean(
                eligible, "all_annotated_turns_fully_retained"
            ),
            "all_annotated_turns_touched": mean(eligible, "all_annotated_turns_touched"),
            "mean_annotated_word_coverage": mean(eligible, "annotated_word_coverage"),
            "retrieved_and_annotation_eligible_questions": len(retrieved),
            "full_retention_given_retrieved": mean(retrieved, "all_annotated_turns_fully_retained"),
            "retrieval_misses": sum(not row["all_gold_sessions_retrieved"] for row in eligible),
            "rendering_losses": sum(
                not row["all_annotated_turns_fully_retained"] for row in retrieved
            ),
            "mean_memory_units": mean(selected, "memory_units"),
            "duplicated_word_occurrences": sum(
                row["duplicated_word_occurrences"] for row in selected
            ),
            "retained_word_occurrences": sum(row["retained_word_occurrences"] for row in selected),
        }
    return result


def audit_answer_plans(
    dataset: Path,
    plan_files: list[Path],
    *,
    tokenizer_file: Path | None = None,
    seed: int = 20261007,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if not 1 <= len(plan_files) <= 2:
        raise ValueError("audit requires one plan or a paired baseline/candidate")
    plans = [load_plan(path) for path in plan_files]
    first = plans[0]
    digest = file_hash(dataset)
    budget_metadata = first["budget"]
    if budget_metadata["unit"] == "characters":
        if tokenizer_file is not None:
            raise ValueError("character plans must not use a tokenizer during audit")
        budget = MemoryBudget(budget_metadata["maximum"])
    elif budget_metadata["unit"] == "tokenizer_tokens" and tokenizer_file is not None:
        budget = MemoryBudget(
            budget_metadata["maximum"],
            tokenizer_file=tokenizer_file,
            model=budget_metadata["generator_model"],
        )
    else:
        raise ValueError("token-budget audit requires the declared local tokenizer file")
    if budget.metadata != budget_metadata:
        raise ValueError("audit tokenizer/budget does not match the plan")
    arms = first["arms"]
    wanted = set(first["question_ids"])
    if not wanted or len(wanted) != len(first["question_ids"]):
        raise ValueError("plan has empty or duplicate question ids")
    by_plan = []
    for plan in plans:
        if plan["dataset_sha256"] != digest:
            raise ValueError("plan and audit dataset do not match")
        if any(plan[field] != first[field] for field in ("budget", "arms", "retrieval_sha256")):
            raise ValueError("paired plans require the same budget, arms and retrieval report")
        if set(plan["question_ids"]) != wanted:
            raise ValueError("paired plans require the same questions")
        indexed = {(row["question_id"], row["arm"]): row for row in plan["rows"]}
        expected = {(question, arm) for question in wanted for arm in arms}
        if set(indexed) != expected or len(indexed) != len(plan["rows"]):
            raise ValueError("plan must have exactly one row for every question and arm")
        by_plan.append(indexed)
    if len(plans) == 2 and plans[0]["rendering"] == plans[1]["rendering"]:
        raise ValueError("paired audit needs distinct rendering policies")
    records: list[list[dict[str, Any]]] = [[] for _ in plans]
    seen: set[str] = set()
    for instance in iter_instances(dataset):
        key = instance.get("question_id")
        if not isinstance(key, str):
            raise ValueError("dataset question_id must be a string")
        if key not in wanted:
            continue
        if key in seen:
            raise ValueError("duplicate dataset question_id")
        seen.add(key)
        validate_instance(instance)
        gold = set(instance["answer_session_ids"])
        refusal = not gold or key.endswith("_abs")
        status, annotations = ("abstention", {}) if refusal else _annotation(instance, gold)
        sessions, _ = dated_sessions(instance)
        for arm in arms:
            pair = (key, arm)
            original_ids = by_plan[0][pair]["selected_session_ids"]
            for i, (plan, indexed) in enumerate(zip(plans, by_plan, strict=True)):
                row = indexed[pair]
                ids = row["selected_session_ids"]
                if ids != original_ids:
                    raise ValueError("paired rendering audit must not change selected sessions")
                if plan["rendering"] == FULL_HISTORY_RENDERING and (
                    arm != "full_history"
                    or ids != list(sessions)
                    or row.get("source_session_count") != len(sessions)
                ):
                    raise ValueError(
                        "complete-history control must preserve every unique source session"
                    )
                if (
                    row["question"] != instance["question"]
                    or row["question_type"] != instance["question_type"]
                ):
                    raise ValueError("plan question differs from source")
                memory, trace = _context(
                    instance["question"],
                    [sessions[sid] for sid in ids],
                    budget,
                    rendering=plan["rendering"],
                )
                prompt = generation_prompt(instance["question"], instance["question_date"], memory)
                if (
                    row["prompt"] != prompt
                    or row["instructions"] != INSTRUCTIONS
                    or row["prompt_sha256"]
                    != object_hash({"instructions": INSTRUCTIONS, "prompt": prompt})
                    or row["memory_units"] != budget.count(memory)
                ):
                    raise ValueError("rendered prompt does not reproduce the archived plan")
                retained = {
                    sid: {word for lo, hi in item["retained_word_ranges"] for word in range(lo, hi)}
                    for sid, item in zip(ids, trace, strict=True)
                }
                total = sum(len(words) for turns in annotations.values() for words in turns)
                hits = sum(
                    len(words & retained.get(sid, set()))
                    for sid, turns in annotations.items()
                    for words in turns
                )
                complete = status == "complete"
                records[i].append(
                    {
                        "question_id": key,
                        "question_type": instance["question_type"],
                        "arm": arm,
                        "expects_refusal": refusal,
                        "annotation_status": status,
                        "all_gold_sessions_retrieved": None if refusal else gold <= set(ids),
                        "annotated_words": total if complete else None,
                        "annotated_word_coverage": hits / total if complete else None,
                        "all_annotated_turns_fully_retained": all(
                            words <= retained.get(sid, set())
                            for sid, turns in annotations.items()
                            for words in turns
                        )
                        if complete
                        else None,
                        "all_annotated_turns_touched": all(
                            bool(words & retained.get(sid, set()))
                            for sid, turns in annotations.items()
                            for words in turns
                        )
                        if complete
                        else None,
                        "memory_units": budget.count(memory),
                        "memory_sha256": text_hash(memory),
                        "retained_word_occurrences": sum(
                            item["retained_word_occurrences"] for item in trace
                        ),
                        "duplicated_word_occurrences": sum(
                            item["duplicated_word_occurrences"] for item in trace
                        ),
                    }
                )
        if progress is not None:
            progress(len(seen))
    if seen != wanted:
        raise ValueError("plan questions are missing from the dataset")
    results = [
        {
            "plan_sha256": plan["plan_sha256"],
            "rendering": plan["rendering"],
            "summary": _summary(rows, arms),
            "by_type": {
                kind: _summary([r for r in rows if r["question_type"] == kind], arms)
                for kind in sorted({r["question_type"] for r in rows})
            },
            "records": rows,
        }
        for plan, rows in zip(plans, records, strict=True)
    ]
    comparisons = []
    if len(plans) == 2:
        for arm in arms:
            fields = {
                plan["rendering"]: [
                    int(r["all_annotated_turns_fully_retained"])
                    for r in rows
                    if r["arm"] == arm and r["annotation_status"] == "complete"
                ]
                for plan, rows in zip(plans, records, strict=True)
            }
            count = len(next(iter(fields.values())))
            if count:
                comparison = paired_ordering_comparison(
                    OrderingHits(count, fields, [], None),
                    [(plans[1]["rendering"], plans[0]["rendering"])],
                    seed=seed,
                )[0]
                comparisons.append({"arm": arm, **asdict(comparison)})
    return {
        "schema_version": "1.0",
        "kind": "longmemeval-rendering-audit",
        "dataset_sha256": digest,
        "budget": budget.metadata,
        "seed": seed,
        "answer_quality": "not_measured",
        "network_calls": 0,
        "sample_is_held_out": False,
        "comparisons_are_exploratory": True,
        "annotation_metric": "complete_whitespace_word_retention_of_all_annotated_gold_turns",
        "metric_limit": (
            "Strict span retention is neither sufficient nor necessary for answer correctness."
        ),
        "plans": results,
        "paired_full_retention": comparisons,
    }
