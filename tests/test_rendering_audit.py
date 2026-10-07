from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_longmemeval_abstention import refusal_bundle
from test_longmemeval_answers import bundle as bundle
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.longmemeval import evaluate_longmemeval, write_report
from context_router.external.longmemeval_answers import (
    MemoryBudget,
    _context,
    load_plan,
    object_hash,
    prepare_answer_plan,
)
from context_router.external.longmemeval_rendering import (
    TURN_RENDERING,
    UNION_RENDERING,
    VisibleSession,
    render_memory,
)
from context_router.external.rendering_audit import audit_answer_plans
from context_router.providers.embedding import HashEmbeddingProvider


def test_union_removes_overlap_and_keeps_original_order_and_whitespace() -> None:
    content = "\n".join(f"word{i:03d}" for i in range(400))
    sessions = [VisibleSession("2026/01/01", (("user", content),))]
    budget = MemoryBudget(5000)
    old, legacy_trace = render_memory("word300", sessions, budget)
    new, union_trace = render_memory("word300", sessions, budget, policy=UNION_RENDERING)
    assert legacy_trace[0]["duplicated_word_occurrences"] == 80
    assert union_trace[0]["duplicated_word_occurrences"] == 0
    assert union_trace[0]["retained_unique_words"] == 401
    assert new.endswith("user: " + content)
    assert old.index("word300") < old.index("word000")
    assert new.index("word000") < new.index("word300")
    assert len(new) <= budget.maximum


@pytest.mark.parametrize(
    "policy", ["equal-share-query-passages-v1", UNION_RENDERING, TURN_RENDERING]
)
def test_partial_final_word_is_never_credited_as_complete_evidence(policy: str) -> None:
    content = "one two " + "longword" * 100
    sessions = [VisibleSession("date", (("user", content),))]
    text, trace = render_memory("absent", sessions, MemoryBudget(128), policy=policy)
    assert "one two" in text
    assert trace[0]["retained_word_ranges"] == [[0, 3]]
    assert trace[0]["source_words"] == 4


def test_renderer_never_reads_answer_labels() -> None:
    first = [{"role": "user", "content": "Original Case 中文 😀", "has_answer": True}]
    other = [{**first[0], "has_answer": False, "answer": "LABEL-SENTINEL"}]
    for policy in ("equal-share-query-passages-v1", UNION_RENDERING, TURN_RENDERING):
        before = _context("Case", [("date", first)], MemoryBudget(128), rendering=policy)
        after = _context("Case", [("date", other)], MemoryBudget(128), rendering=policy)
        assert before == after
        assert "LABEL-SENTINEL" not in before[0]


def test_whole_turn_selection_skips_oversized_turns_and_preserves_speaker_order() -> None:
    turns = (
        ("user", "archiver zip"),
        ("assistant", "archiver " * 200),
        ("user", "archiver changed to tar"),
    )
    memory, trace = render_memory(
        "archiver", [VisibleSession("date", turns)], MemoryBudget(128), policy=TURN_RENDERING
    )
    assert "user: archiver zip" in memory
    assert "user: archiver changed to tar" in memory
    assert "assistant: " not in memory
    assert memory.index("zip") < memory.index("tar")
    assert trace[0]["selection_unit"] == "turn"
    assert trace[0]["duplicated_word_occurrences"] == 0


@pytest.mark.parametrize("rendering", [UNION_RENDERING, TURN_RENDERING])
def test_archived_plan_is_reproduced_and_new_plan_has_matched_paired_audit(
    bundle: tuple[Path, Path, Path],
    rendering: str,
) -> None:
    dataset, retrieval, original = bundle
    candidate = original.with_name("union-plan")
    prepare_answer_plan(
        dataset, retrieval, candidate, budget=MemoryBudget(256), rendering=rendering
    )
    # Simulate a saved plan from before provenance tracing was implemented.
    legacy_file = original / "plan.private.json"
    legacy = load_plan(legacy_file)
    for row in legacy["rows"]:
        row["passage_audit"] = [
            {key: item[key] for key in ("slot", "passages", "source_passages")}
            for item in row["passage_audit"]
        ]
    legacy.pop("plan_sha256")
    legacy["plan_sha256"] = object_hash(legacy)
    write_report(legacy_file, legacy)
    report = audit_answer_plans(dataset, [legacy_file, candidate / "plan.private.json"])
    assert report["answer_quality"] == "not_measured"
    assert report["network_calls"] == 0
    assert report["sample_is_held_out"] is False
    assert len(report["paired_full_retention"]) == 4
    for result in report["plans"]:
        assert result["summary"]["joint"]["annotation_eligible_questions"] == 2
        assert result["summary"]["query_only"]["all_annotated_turns_fully_retained"] == 0
        assert all(row["memory_units"] <= 256 for row in result["records"])


def test_missing_annotations_are_unknown_and_not_silently_credited(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, retrieval, original = bundle
    samples = json.loads(dataset.read_text())
    for sample in samples:
        for session in sample["haystack_sessions"]:
            for turn in session:
                turn.pop("has_answer", None)
    dataset.write_text(json.dumps(samples))
    write_report(retrieval, evaluate_longmemeval(dataset, HashEmbeddingProvider()))
    output = original.with_name("unlabelled")
    prepare_answer_plan(
        dataset, retrieval, output, budget=MemoryBudget(256), arms=("hybrid", "query_only")
    )
    report = audit_answer_plans(dataset, [output / "plan.private.json"])
    summary = report["plans"][0]["summary"]["hybrid"]
    assert summary["annotation_unknown_questions"] == 2
    assert summary["annotation_eligible_questions"] == 0
    assert summary["all_annotated_turns_fully_retained"] is None


def test_conflicting_copy_annotations_are_unknown_even_when_visible_text_matches(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, retrieval, original = bundle
    samples = json.loads(dataset.read_text())
    for sample in samples:
        sample["haystack_session_ids"].append(sample["haystack_session_ids"][0])
        sample["haystack_dates"].append(sample["haystack_dates"][0])
        copied = [dict(turn, has_answer=False) for turn in sample["haystack_sessions"][0]]
        sample["haystack_sessions"].append(copied)
    dataset.write_text(json.dumps(samples))
    write_report(retrieval, evaluate_longmemeval(dataset, HashEmbeddingProvider()))
    directory = original.with_name("annotation-conflict")
    prepare_answer_plan(dataset, retrieval, directory, budget=MemoryBudget(256), arms=("hybrid",))
    report = audit_answer_plans(dataset, [directory / "plan.private.json"])
    assert all(
        row["annotation_status"] == "conflicting_duplicate_annotations"
        and row["all_annotated_turns_fully_retained"] is None
        for row in report["plans"][0]["records"]
    )


def test_prompt_changes_with_a_recomputed_plan_hash_still_fail_source_reproduction(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, _, directory = bundle
    path = directory / "plan.private.json"
    plan = load_plan(path)
    plan["rows"][0]["prompt"] += "Unrecorded text"
    plan.pop("plan_sha256")
    plan["plan_sha256"] = object_hash(plan)
    write_report(path, plan)
    with pytest.raises(ValueError, match="does not reproduce"):
        audit_answer_plans(dataset, [path])


def test_identical_words_in_another_turn_do_not_replace_source_positions(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, retrieval, original = bundle
    samples = json.loads(dataset.read_text())[:1]
    sample = samples[0]
    sample["question"] = "anchor"
    sample["haystack_session_ids"] = ["one"]
    sample["answer_session_ids"] = ["one"]
    sample["haystack_dates"] = ["2026/01/01"]
    sample["haystack_sessions"] = [
        [
            {"role": "user", "content": "anchor " + "repeated " * 200},
            {"role": "user", "content": "repeated " * 200, "has_answer": True},
        ]
    ]
    dataset.write_text(json.dumps(samples))
    write_report(retrieval, evaluate_longmemeval(dataset, HashEmbeddingProvider()))
    output = original.with_name("repeated")
    prepare_answer_plan(dataset, retrieval, output, budget=MemoryBudget(128), arms=("hybrid",))
    report = audit_answer_plans(dataset, [output / "plan.private.json"])
    row = report["plans"][0]["records"][0]
    assert row["all_gold_sessions_retrieved"] is True
    assert row["all_annotated_turns_touched"] is False
    assert row["annotated_word_coverage"] == 0
    assert report["plans"][0]["summary"]["hybrid"]["rendering_losses"] == 1


def test_abstention_does_not_become_perfect_evidence_retention(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, _, directory = refusal_bundle(bundle, only=True)
    report = audit_answer_plans(dataset, [directory / "plan.private.json"])
    assert all(row["annotated_word_coverage"] is None for row in report["plans"][0]["records"])
    assert report["plans"][0]["summary"]["hybrid"]["all_gold_session_recall"] is None
    assert report["plans"][0]["summary"]["hybrid"]["annotation_eligible_questions"] == 0


def test_audit_rejects_changed_inputs_and_matched_budget_violations(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, retrieval, original = bundle
    candidate = original.with_name("other-budget")
    prepare_answer_plan(
        dataset, retrieval, candidate, budget=MemoryBudget(512), rendering=UNION_RENDERING
    )
    with pytest.raises(ValueError, match="same budget"):
        audit_answer_plans(
            dataset, [original / "plan.private.json", candidate / "plan.private.json"]
        )
    dataset.write_text(dataset.read_text() + " ")
    with pytest.raises(ValueError, match="dataset do not match"):
        audit_answer_plans(dataset, [original / "plan.private.json"])


def test_cli_exposes_opt_in_rendering_and_offline_audit(bundle: tuple[Path, Path, Path]) -> None:
    dataset, retrieval, original = bundle
    runner = CliRunner()
    candidate = original.with_name("cli-union")
    result = runner.invoke(
        app,
        [
            "longmemeval-answer-plan",
            str(dataset),
            str(retrieval),
            str(candidate),
            "--rendering",
            UNION_RENDERING,
            "--memory-budget",
            "256",
        ],
    )
    assert result.exit_code == 0, result.output
    report_file = original / "audit.json"
    result = runner.invoke(
        app,
        [
            "longmemeval-render-audit",
            str(dataset),
            str(original / "plan.private.json"),
            str(report_file),
            "--compare-plan",
            str(candidate / "plan.private.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not correctness" in result.output
    assert json.loads(report_file.read_text())["network_calls"] == 0


def test_local_tokenizer_budget_is_verified_for_audit(bundle: tuple[Path, Path, Path]) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    dataset, retrieval, original = bundle
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_file = original / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    budget = MemoryBudget(128, tokenizer_file=tokenizer_file, model="fixture-generator")
    output = original.with_name("token-plan")
    prepare_answer_plan(dataset, retrieval, output, budget=budget, rendering=UNION_RENDERING)
    with pytest.raises(ValueError, match="local tokenizer"):
        audit_answer_plans(dataset, [output / "plan.private.json"])
    report = audit_answer_plans(
        dataset, [output / "plan.private.json"], tokenizer_file=tokenizer_file
    )
    assert report["budget"] == budget.metadata
    assert all(row["memory_units"] <= 128 for row in report["plans"][0]["records"])
