"""Frozen, blinded rendering and complete-history controls, without model calls.

The inspected sample stays exploratory. Content hashes bind artifacts to a design;
they do not prove that the design was registered before somebody saw answers.
"""

from __future__ import annotations

import json
import math
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from context_router.external.longmemeval import write_report
from context_router.external.longmemeval_answers import (
    BOUNDED_MEMORY,
    COMPLETE_HISTORY,
    DEFAULT_ARMS,
    file_hash,
    judge_requests_for_rows,
    load_exchange_rows,
    load_plan,
    object_hash,
    score_answer_rows,
    validate_answer_rows,
    write_exchange_rows,
)
from context_router.external.longmemeval_rendering import FULL_HISTORY_RENDERING, RENDERINGS
from context_router.external.scale_qa import OrderingHits, paired_ordering_comparison

SIDES = ("baseline", "candidate")
PUBLIC_FIELDS = ("request_id", "prompt_sha256", "instructions", "prompt")


@dataclass(frozen=True)
class QualityCriteria:
    """Freeze the decision rules before exporting a comparison's generation inputs."""

    minimum_accuracy_gain: float = 0.0
    maximum_failed_completion_increase: float = 0.0
    maximum_mean_memory_ratio: float = 1.0
    maximum_exact_mcnemar_p: float = 0.05

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise ValueError("quality criteria must be finite numbers")
        if not (
            0 <= self.minimum_accuracy_gain <= 1
            and 0 <= self.maximum_failed_completion_increase <= 1
            and self.maximum_mean_memory_ratio > 0
            and 0 <= self.maximum_exact_mcnemar_p <= 1
        ):
            raise ValueError("invalid quality criteria bounds")


DEFAULT_CRITERIA = QualityCriteria()


def _matched_rows(
    plans: dict[str, dict[str, Any]], arm: str, comparison_kind: str = "rendering"
) -> dict[str, dict[str, dict[str, Any]]]:
    if comparison_kind not in ("rendering", "full-history"):
        raise ValueError("comparison kind must be rendering or full-history")
    if arm not in DEFAULT_ARMS or arm == "query_only":
        raise ValueError("choose a memory arm: joint, hybrid or router")
    first = plans["baseline"]
    common = ("dataset_sha256", "retrieval_sha256", "budget", "duplicate_date_policy", "scope")
    indexed = {}
    for side, plan in plans.items():
        selected_arm = (
            "full_history" if comparison_kind == "full-history" and side == "baseline" else arm
        )
        if any(plan[field] != first[field] for field in common):
            raise ValueError("rendering comparison requires matched dataset, retrieval and budget")
        if plan["rendering"] not in RENDERINGS:
            raise ValueError("unknown rendering policy")
        complete = selected_arm == "full_history"
        expected_cap = COMPLETE_HISTORY if complete else BOUNDED_MEMORY
        if (
            plan.get("memory_cap_policy", BOUNDED_MEMORY) != expected_cap
            or (plan["rendering"] == FULL_HISTORY_RENDERING) != complete
        ):
            raise ValueError(
                "comparison requires an explicit complete-history baseline and bounded candidate"
            )
        questions = plan["question_ids"]
        if (
            not questions
            or any(not isinstance(q, str) or not q for q in questions)
            or len(set(questions)) != len(questions)
            or set(questions) != set(first["question_ids"])
            or selected_arm not in plan["arms"]
            or (complete and plan["arms"] != ["full_history"])
        ):
            raise ValueError("comparison requires the same complete question set and selected arm")
        rows = {(row["question_id"], row["arm"]): row for row in plan["rows"]}
        expected = {(q, a) for q in questions for a in plan["arms"]}
        if set(rows) != expected or len(rows) != len(plan["rows"]):
            raise ValueError("plan must contain exactly one row per question and arm")
        if len({row["request_id"] for row in plan["rows"]}) != len(plan["rows"]):
            raise ValueError("plan request ids must be unique")
        indexed[side] = {q: rows[q, selected_arm] for q in questions}
        for row in indexed[side].values():
            if type(row.get("expects_refusal", False)) is not bool:
                raise ValueError("refusal flags must be JSON booleans")
            if (
                type(row["memory_units"]) is not int
                or row["memory_units"] < 0
                or (not complete and row["memory_units"] > plan["budget"]["maximum"])
            ):
                raise ValueError("memory units must be a nonnegative integer within the budget")
            if complete and (
                type(row.get("source_session_count")) is not int
                or row["source_session_count"] != len(row["selected_session_ids"])
                or len(set(row["selected_session_ids"])) != row["source_session_count"]
            ):
                raise ValueError(
                    "complete-history baseline must declare all unique source sessions"
                )
            if row["prompt_sha256"] != object_hash(
                {"instructions": row["instructions"], "prompt": row["prompt"]}
            ):
                raise ValueError("plan prompt failed its integrity check")
    if plans["baseline"]["rendering"] == plans["candidate"]["rendering"]:
        raise ValueError("comparison needs distinct rendering policies")
    fields: tuple[str, ...] = (
        "question",
        "question_type",
        "reference",
        "instructions",
        "ambiguous_date_session_ids",
    )
    if comparison_kind == "rendering":
        fields = (*fields, "selected_session_ids")
    for question, before in indexed["baseline"].items():
        after = indexed["candidate"][question]
        if any(before[field] != after[field] for field in fields) or before.get(
            "expects_refusal", False
        ) != after.get("expects_refusal", False):
            raise ValueError("only rendering may change; questions, labels and sessions must match")
    return indexed


def _request_id(source_hash: str, side: str, original: str) -> str:
    return object_hash(["longmemeval-quality-comparison-v1", source_hash, side, original])[:32]


def prepare_quality_comparison(
    baseline_file: Path,
    candidate_file: Path,
    output: Path,
    *,
    arm: str = "joint",
    comparison_kind: str = "rendering",
    seed: int = 20261007,
    criteria: QualityCriteria = DEFAULT_CRITERIA,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("comparison output already exists; choose a new directory")
    sources = dict(zip(SIDES, (baseline_file, candidate_file), strict=True))
    originals = {side: load_plan(path) for side, path in sources.items()}
    indexed = _matched_rows(originals, arm, comparison_kind)
    questions = sorted(originals["baseline"]["question_ids"])
    plans: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    for side, source in originals.items():
        rows = [
            {
                **indexed[side][q],
                "source_request_id": indexed[side][q]["request_id"],
                "request_id": _request_id(
                    source["plan_sha256"], side, indexed[side][q]["request_id"]
                ),
            }
            for q in questions
        ]
        plan = {
            **source,
            "arms": [
                "full_history" if comparison_kind == "full-history" and side == "baseline" else arm
            ],
            "question_ids": questions,
            "rows": rows,
            "seed": seed,
            "source_plan_sha256": source["plan_sha256"],
            "sample_is_held_out": False,
        }
        plan.pop("plan_sha256")
        plan["plan_sha256"] = object_hash(plan)
        plans[side] = plan
        requests.extend({field: row[field] for field in PUBLIC_FIELDS} for row in rows)
    random.Random(seed).shuffle(requests)
    first = plans["baseline"]
    manifest = {
        "schema_version": "1.0",
        "kind": "longmemeval-quality-comparison",
        "primary_arm": arm,
        "seed": seed,
        "question_ids": questions,
        "dataset_sha256": first["dataset_sha256"],
        "retrieval_sha256": first["retrieval_sha256"],
        "budget": first["budget"],
        "criteria": asdict(criteria),
        "plans": {
            side: {
                "file": f"{side}.plan.private.json",
                "plan_sha256": plan["plan_sha256"],
                "source_plan_sha256": originals[side]["plan_sha256"],
                "source_file_sha256": file_hash(sources[side]),
                "rendering": plan["rendering"],
            }
            for side, plan in plans.items()
        },
        "sample_is_held_out": False,
        "comparisons_are_exploratory": True,
        "generation_configuration_frozen": False,
        "registration_timing_verified": False,
    }
    status = {
        "status": "pending_generation",
        "answer_quality": "not_measured",
        "questions": len(questions),
        "requests": len(requests),
        "primary_arm": arm,
        "answerable_questions": sum(not r.get("expects_refusal", False) for r in first["rows"]),
        "abstention_questions": sum(r.get("expects_refusal", False) for r in first["rows"]),
        "network_calls": 0,
        "sample_is_held_out": False,
        "generation_configuration_frozen": False,
    }
    if comparison_kind == "full-history":
        for metadata in (manifest, status):
            metadata["comparison_kind"] = comparison_kind
            metadata["baseline_arm"] = "full_history"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        directory = Path(temporary) / "comparison"
        directory.mkdir()
        for side, plan in plans.items():
            write_report(directory / f"{side}.plan.private.json", plan)
        write_exchange_rows(directory / "generation.requests.jsonl", requests)
        manifest["generation_requests_sha256"] = file_hash(directory / "generation.requests.jsonl")
        manifest["comparison_sha256"] = object_hash(manifest)
        write_report(directory / "comparison.private.json", manifest)
        write_report(directory / "status.json", status)
        directory.replace(output)
    return {**status, "comparison_sha256": manifest["comparison_sha256"]}


def load_quality_comparison(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = manifest.pop("comparison_sha256", None)
    if manifest.get("kind") != "longmemeval-quality-comparison" or object_hash(manifest) != digest:
        raise ValueError("quality comparison failed its integrity check")
    manifest["comparison_sha256"] = digest
    if set(manifest["criteria"]) != set(asdict(DEFAULT_CRITERIA)):
        raise ValueError("comparison must explicitly freeze every quality criterion")
    QualityCriteria(**manifest["criteria"])
    comparison_kind = manifest.get("comparison_kind", "rendering")
    baseline_arm = "full_history" if comparison_kind == "full-history" else manifest["primary_arm"]
    if manifest.get("baseline_arm", baseline_arm) != baseline_arm or (
        comparison_kind == "full-history" and manifest.get("baseline_arm") != baseline_arm
    ):
        raise ValueError("comparison baseline arm does not match its kind")
    plans = {}
    for side in SIDES:
        entry = manifest["plans"][side]
        if entry["file"] != f"{side}.plan.private.json":
            raise ValueError("comparison must use its local frozen plan files")
        plan = load_plan(path.parent / entry["file"])
        if (
            plan["plan_sha256"] != entry["plan_sha256"]
            or plan["source_plan_sha256"] != entry["source_plan_sha256"]
            or plan["rendering"] != entry["rendering"]
            or plan["arms"] != [baseline_arm if side == "baseline" else manifest["primary_arm"]]
            or plan["question_ids"] != manifest["question_ids"]
            or plan["seed"] != manifest["seed"]
            or any(plan[f] != manifest[f] for f in ("dataset_sha256", "retrieval_sha256", "budget"))
            or any(
                r["request_id"]
                != _request_id(entry["source_plan_sha256"], side, r["source_request_id"])
                for r in plan["rows"]
            )
        ):
            raise ValueError("comparison's frozen plan changed")
        plans[side] = plan
    _matched_rows(plans, manifest["primary_arm"], comparison_kind)
    if (
        file_hash(path.parent / "generation.requests.jsonl")
        != manifest["generation_requests_sha256"]
    ):
        raise ValueError("comparison's generation requests changed")
    combined = _combined_plan(manifest, plans)
    public = load_exchange_rows(
        path.parent / "generation.requests.jsonl", {r["request_id"] for r in combined["rows"]}
    )
    if any(public[r["request_id"]] != {f: r[f] for f in PUBLIC_FIELDS} for r in combined["rows"]):
        raise ValueError("comparison's generation requests do not reproduce the frozen plans")
    return manifest, plans


def _combined_plan(manifest: dict[str, Any], plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": [row for side in SIDES for row in plans[side]["rows"]],
        "budget": manifest["budget"],
        "seed": manifest["seed"],
    }


def prepare_quality_judgments(comparison: Path, answers_file: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("judge output already exists; choose a new file")
    manifest, plans = load_quality_comparison(comparison)
    combined = _combined_plan(manifest, plans)
    answers = load_exchange_rows(answers_file, {r["request_id"] for r in combined["rows"]})
    rows = judge_requests_for_rows(combined, answers)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_exchange_rows(output, rows)
    return {
        "status": "pending_judgments",
        "requests": len(rows),
        "answer_quality": "not_measured",
        "network_calls": 0,
    }


def _paired(
    records: dict[str, list[dict[str, Any]]], *, refusal: bool, seed: int, kind: str | None = None
) -> dict[str, Any] | None:
    indexed = {
        side: {
            r["question_id"]: r
            for r in rows
            if r["expects_refusal"] == refusal and (kind is None or r["question_type"] == kind)
        }
        for side, rows in records.items()
    }
    questions = sorted(indexed["baseline"])
    if set(questions) != set(indexed["candidate"]):
        raise ValueError("unbalanced cross-rendering paired stratum")
    if not questions:
        return None
    fields = {side: [int(indexed[side][q]["correct"]) for q in questions] for side in SIDES}
    return asdict(
        paired_ordering_comparison(
            OrderingHits(len(questions), fields, [], None), [("candidate", "baseline")], seed=seed
        )[0]
    )


def _sample_gate(
    manifest: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    pair: dict[str, Any] | None,
    refusal_pair: dict[str, Any] | None,
) -> dict[str, Any]:
    if pair is None:
        return {"status": "not_measured_no_answerable_questions", "conditions": {}}
    arm = manifest["primary_arm"]
    before = reports["baseline"]["summary"][manifest.get("baseline_arm", arm)]
    after = reports["candidate"]["summary"][arm]
    baseline_memory, candidate_memory = before["mean_memory_units"], after["mean_memory_units"]
    memory_ratio = candidate_memory / baseline_memory if baseline_memory else None
    failed_delta = after["failed_completion_rate"] - before["failed_completion_rate"]
    criteria = manifest["criteria"]
    conditions = {
        "exact_mcnemar_within_limit": pair["exact_mcnemar_p"]
        <= criteria["maximum_exact_mcnemar_p"],
        "accuracy_interval_above_minimum_gain": pair["bootstrap_low"]
        > criteria["minimum_accuracy_gain"],
        "failed_completion_increase_within_limit": failed_delta
        <= criteria["maximum_failed_completion_increase"],
        "mean_memory_ratio_within_limit": memory_ratio is not None
        and memory_ratio <= criteria["maximum_mean_memory_ratio"],
    }
    if refusal_pair is not None:
        conditions["no_observed_abstention_accuracy_regression"] = refusal_pair["difference"] >= 0
    if manifest.get("comparison_kind") == "full-history":
        conditions["complete_history_baseline_completed"] = all(
            row["completed"] for row in reports["baseline"]["records"]
        )
    return {
        "status": "sample_criteria_passed"
        if all(conditions.values())
        else "sample_criteria_not_passed",
        "conditions": conditions,
        "mean_memory_ratio": memory_ratio,
        "failed_completion_increase": failed_delta,
        "abstention_measured": refusal_pair is not None,
    }


def score_quality_comparison(
    comparison: Path, answers_file: Path, judgments_file: Path
) -> dict[str, Any]:
    manifest, plans = load_quality_comparison(comparison)
    combined = _combined_plan(manifest, plans)
    answers = load_exchange_rows(answers_file, {r["request_id"] for r in combined["rows"]})
    validate_answer_rows(combined, answers)
    judgments = load_exchange_rows(judgments_file, set(answers))
    if any(
        not isinstance(j.get("judgment_config"), dict) or not j["judgment_config"]
        for j in judgments.values()
    ):
        raise ValueError("comparison requires explicit judgment_config, including a human protocol")
    reports = {}
    for side, plan in plans.items():
        keys = {r["request_id"] for r in plan["rows"]}
        reports[side] = score_answer_rows(
            plan,
            {k: answers[k] for k in keys},
            {k: judgments[k] for k in keys},
            seed=manifest["seed"],
        )
    first = reports["baseline"]
    if any(reports["candidate"][f] != first[f] for f in ("judge_model", "judgment_config")):
        raise ValueError("both renderings must use the same judge and judging configuration")
    if first["generator_model"] == first["judge_model"]:
        raise ValueError("comparison requires a judge distinct from the generator")
    records = {side: report["records"] for side, report in reports.items()}
    pair = _paired(records, refusal=False, seed=manifest["seed"])
    refusal_pair = _paired(records, refusal=True, seed=manifest["seed"])
    return {
        "schema_version": "1.0",
        "kind": "longmemeval-quality-comparison-report",
        "status": "complete",
        "answer_quality": "judged_reference_accuracy",
        "comparison_sha256": manifest["comparison_sha256"],
        "answers_sha256": file_hash(answers_file),
        "judgments_sha256": file_hash(judgments_file),
        "dataset_sha256": manifest["dataset_sha256"],
        "retrieval_sha256": manifest["retrieval_sha256"],
        "primary_arm": manifest["primary_arm"],
        "baseline_arm": manifest.get("baseline_arm", manifest["primary_arm"]),
        "comparison_kind": manifest.get("comparison_kind", "rendering"),
        "seed": manifest["seed"],
        "criteria": manifest["criteria"],
        "plans": reports,
        "primary_answerable_comparison": pair,
        "abstention_comparison": refusal_pair,
        "by_type": {
            kind: _paired(records, refusal=False, seed=manifest["seed"], kind=kind)
            for kind in sorted(
                {r["question_type"] for r in records["baseline"] if not r["expects_refusal"]}
            )
        },
        "sample_gate": _sample_gate(manifest, reports, pair, refusal_pair),
        "sample_is_held_out": False,
        "comparisons_are_exploratory": True,
        "secondary_comparisons_are_exploratory": True,
        "default_promotion_eligible": False,
        "registration_timing_verified": False,
        "generator_token_budget_verified": False,
        "judge_identity_differs_from_generator": True,
        "network_calls": 0,
    }
