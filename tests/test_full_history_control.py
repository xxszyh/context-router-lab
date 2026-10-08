from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_longmemeval import RecordingScorer
from test_longmemeval_abstention import refusal_bundle
from test_longmemeval_answers import bundle as bundle
from test_quality_comparison import exchange, reseal, score_rows
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.longmemeval import evaluate_longmemeval, write_report
from context_router.external.longmemeval_answers import (
    COMPLETE_HISTORY,
    DEFAULT_ARMS,
    MemoryBudget,
    load_plan,
    prepare_answer_plan,
    text_hash,
)
from context_router.external.longmemeval_rendering import (
    FULL_HISTORY_RENDERING,
    TURN_RENDERING,
    VisibleSession,
    render_memory,
)
from context_router.external.quality_comparison import (
    QualityCriteria,
    load_quality_comparison,
    prepare_quality_comparison,
)
from context_router.external.rendering_audit import audit_answer_plans
from context_router.providers.embedding import HashEmbeddingProvider


def controls(
    source: tuple[Path, Path, Path], *, include_abstention: bool = False
) -> tuple[Path, Path]:
    dataset, retrieval, original = source
    samples = json.loads(dataset.read_text(encoding="utf-8"))
    for sample in samples:
        sample["haystack_sessions"][-1].append(
            {"role": "assistant", "content": "padding " * 300 + "\n中文 tail-sentinel"}
        )
    dataset.write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    report = evaluate_longmemeval(
        dataset,
        HashEmbeddingProvider(),
        scorer=RecordingScorer(),
        include_abstention=include_abstention,
    )
    for row in [*report["records"], *report.get("abstention_records", [])]:
        row["arms"]["joint"]["retrieved_ids"] = row["arms"]["joint"]["retrieved_ids"][:2]
    write_report(retrieval, report)
    baseline, candidate = original.with_name("full"), original.with_name("routed")
    prepare_answer_plan(
        dataset,
        retrieval,
        baseline,
        budget=MemoryBudget(256),
        arms=("full_history",),
        include_abstention=include_abstention,
    )
    prepare_answer_plan(
        dataset,
        retrieval,
        candidate,
        budget=MemoryBudget(256),
        arms=("joint",),
        rendering=TURN_RENDERING,
        include_abstention=include_abstention,
    )
    return baseline / "plan.private.json", candidate / "plan.private.json"


def comparison_for(source: tuple[Path, Path, Path]) -> Path:
    before, after = controls(source)
    output = source[2].with_name("full-comparison")
    prepare_quality_comparison(
        before,
        after,
        output,
        comparison_kind="full-history",
        # A two-question fixture exercises the gate; production keeps alpha .05.
        criteria=QualityCriteria(maximum_exact_mcnemar_p=1),
    )
    return output / "comparison.private.json"


def test_complete_renderer_preserves_whole_source_independent_of_query_and_cap() -> None:
    content = "padding " * 500 + "\n中文  🚀\n tail-sentinel\t"
    sessions = [
        VisibleSession("later", (("user", content), ("assistant", "second turn"))),
        VisibleSession("earlier", (("user", "last session"),)),
    ]
    memory, trace = render_memory(
        "padding", sessions, MemoryBudget(128), policy=FULL_HISTORY_RENDERING
    )
    other, _ = render_memory(
        "unrelated", sessions, MemoryBudget(2048), policy=FULL_HISTORY_RENDERING
    )
    assert memory == other
    assert len(memory) > 2048
    assert "user: " + content + "\nassistant: second turn" in memory
    assert memory.index("date later") < memory.index("date earlier")
    assert memory.endswith("user: last session")
    assert all(t["selection_unit"] == "complete_session" for t in trace)
    assert all(t["source_words"] == t["retained_unique_words"] for t in trace)
    assert all(t["duplicated_word_occurrences"] == 0 for t in trace)


def test_full_plan_is_explicit_uncapped_label_free_and_source_auditable(
    bundle: tuple[Path, Path, Path],
) -> None:
    assert load_plan(bundle[2] / "plan.private.json")["arms"] == list(DEFAULT_ARMS)
    baseline, candidate = controls(bundle)
    full = load_plan(baseline)
    assert full["arms"] == ["full_history"]
    assert full["rendering"] == FULL_HISTORY_RENDERING
    assert full["memory_cap_policy"] == COMPLETE_HISTORY
    assert all(r["memory_units"] > full["budget"]["maximum"] for r in full["rows"])
    assert all(
        r["source_session_count"] == len(r["selected_session_ids"]) == 3 for r in full["rows"]
    )
    assert all("tail-sentinel" in r["prompt"] for r in full["rows"])
    public = (baseline.parent / "generation.requests.jsonl").read_text(encoding="utf-8")
    for forbidden in ("REFERENCE-ONLY-SENTINEL", "answer_gold_hint", "has_answer", "full_history"):
        assert forbidden not in public
    audit = audit_answer_plans(bundle[0], [baseline])
    assert audit["plans"][0]["summary"]["full_history"]["all_annotated_turns_fully_retained"] == 1
    assert audit["plans"][0]["summary"]["full_history"]["duplicated_word_occurrences"] == 0
    with pytest.raises(ValueError, match="same budget, arms"):
        audit_answer_plans(bundle[0], [baseline, candidate])


@pytest.mark.parametrize(
    "arms,rendering",
    [
        (("full_history", "joint"), FULL_HISTORY_RENDERING),
        (("joint",), FULL_HISTORY_RENDERING),
        (("full_history",), TURN_RENDERING),
    ],
)
def test_uncapped_rendering_cannot_silently_replace_routed_prompts(
    bundle: tuple[Path, Path, Path], arms: tuple[str, ...], rendering: str
) -> None:
    output = bundle[2].with_name("invalid")
    with pytest.raises(ValueError, match="full_history"):
        prepare_answer_plan(
            bundle[0], bundle[1], output, budget=MemoryBudget(256), arms=arms, rendering=rendering
        )
    assert not output.exists()


def test_full_control_requires_explicit_comparison_kind(
    bundle: tuple[Path, Path, Path],
) -> None:
    before, after = controls(bundle)
    with pytest.raises(ValueError, match="explicit complete-history"):
        prepare_quality_comparison(before, after, bundle[2].with_name("accidental"))
    output = bundle[2].with_name("comparison")
    status = prepare_quality_comparison(before, after, output, comparison_kind="full-history")
    manifest, plans = load_quality_comparison(output / "comparison.private.json")
    assert status["network_calls"] == 0 and status["requests"] == 4
    assert manifest["comparison_kind"] == "full-history"
    assert manifest["baseline_arm"] == "full_history"
    assert manifest["criteria"]["maximum_exact_mcnemar_p"] == 0.05
    assert plans["baseline"]["arms"] == ["full_history"]
    assert plans["candidate"]["arms"] == ["joint"]
    assert all(len(r["selected_session_ids"]) == 2 for r in plans["candidate"]["rows"])


@pytest.mark.parametrize("failure", [None, "length", "context_limit", "error", "empty"])
def test_failed_full_history_baseline_never_qualifies_as_quality_improvement(
    bundle: tuple[Path, Path, Path], failure: str | None
) -> None:
    comparison = comparison_for(bundle)
    _, plans = load_quality_comparison(comparison)
    baseline_ids = {r["request_id"] for r in plans["baseline"]["rows"]}
    answers, judgments = exchange(comparison)
    for judgment in judgments:
        judgment["correct"] = judgment["request_id"] not in baseline_ids
    for answer in answers:
        if answer["request_id"] in baseline_ids and failure is not None:
            answer["stop_reason"] = "stop" if failure == "empty" else failure
            if failure == "empty":
                answer["hypothesis"] = ""
                for judgment in judgments:
                    if judgment["request_id"] == answer["request_id"]:
                        judgment["answer_sha256"] = text_hash("")
    report = score_rows(comparison, list(reversed(answers)), list(reversed(judgments)))
    assert report["primary_answerable_comparison"]["questions"] == 2
    assert report["primary_answerable_comparison"]["difference"] == 1
    gate = report["sample_gate"]
    assert gate["conditions"]["complete_history_baseline_completed"] is (failure is None)
    assert gate["status"] == (
        "sample_criteria_passed" if failure is None else "sample_criteria_not_passed"
    )
    assert report["comparison_kind"] == "full-history"
    assert report["plans"]["baseline"]["summary"]["full_history"]["count"] == 2
    assert report["default_promotion_eligible"] is False


def test_source_audit_rejects_a_partial_control_resealed_as_full_history(
    bundle: tuple[Path, Path, Path],
) -> None:
    before, _ = controls(bundle)
    full = load_plan(before)
    full["rows"][0]["selected_session_ids"].pop()
    full["rows"][0]["source_session_count"] -= 1
    reseal(before, full)
    with pytest.raises(ValueError, match="every unique source session"):
        audit_answer_plans(bundle[0], [before])


@pytest.mark.parametrize("only", [False, True])
def test_full_history_refusal_keeps_its_denominator_and_baseline_failure_guard(
    bundle: tuple[Path, Path, Path], only: bool
) -> None:
    source = refusal_bundle(bundle, only=only)
    before, after = controls(source, include_abstention=True)
    output = source[2].with_name("refusal-full-comparison")
    prepare_quality_comparison(
        before,
        after,
        output,
        comparison_kind="full-history",
        criteria=QualityCriteria(maximum_exact_mcnemar_p=1),
    )
    comparison = output / "comparison.private.json"
    _, plans = load_quality_comparison(comparison)
    baseline_ids = {r["request_id"] for r in plans["baseline"]["rows"]}
    refusal_ids = {
        r["request_id"] for p in plans.values() for r in p["rows"] if r["expects_refusal"]
    }
    answers, judgments = exchange(comparison)
    for judgment in judgments:
        judgment["correct"] = (
            judgment["request_id"] in refusal_ids or judgment["request_id"] not in baseline_ids
        )
    for answer in answers:
        if answer["request_id"] in baseline_ids & refusal_ids:
            answer["stop_reason"] = "context_limit"
    report = score_rows(comparison, answers, judgments)
    assert report["abstention_comparison"]["questions"] == 1
    assert (
        report["plans"]["baseline"]["abstention_summary"]["full_history"]["failed_completion_rate"]
        == 1
    )
    if only:
        assert report["primary_answerable_comparison"] is None
        assert report["sample_gate"]["status"] == "not_measured_no_answerable_questions"
    else:
        assert report["primary_answerable_comparison"]["questions"] == 2
        assert report["sample_gate"]["conditions"]["complete_history_baseline_completed"] is False


@pytest.mark.parametrize("problem", ["cap", "count", "reference", "budget", "kind"])
def test_full_comparison_rejects_mismatched_controls(
    bundle: tuple[Path, Path, Path], problem: str
) -> None:
    before, after = controls(bundle)
    full = load_plan(before)
    kind = "full-history"
    if problem == "cap":
        full.pop("memory_cap_policy")
    elif problem == "count":
        full["rows"][0]["source_session_count"] += 1
    elif problem == "reference":
        full["rows"][0]["reference"] = "changed"
    elif problem == "budget":
        full["budget"]["maximum"] += 1
    else:
        kind = "typo"
    reseal(before, full)
    with pytest.raises(ValueError):
        prepare_quality_comparison(
            before, after, bundle[2].with_name("invalid"), comparison_kind=kind
        )


def test_cli_exports_and_loads_complete_history_comparison(
    bundle: tuple[Path, Path, Path],
) -> None:
    before, after = controls(bundle)
    output = bundle[2].with_name("cli-comparison")
    result = CliRunner().invoke(
        app,
        [
            "longmemeval-quality-plan",
            str(before),
            str(after),
            str(output),
            "--comparison-kind",
            "full-history",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not_measured" in result.output
    manifest, _ = load_quality_comparison(output / "comparison.private.json")
    assert manifest["comparison_kind"] == "full-history"


def test_full_manifest_cannot_omit_its_baseline_identity(
    bundle: tuple[Path, Path, Path],
) -> None:
    comparison = comparison_for(bundle)
    manifest = json.loads(comparison.read_text(encoding="utf-8"))
    manifest.pop("baseline_arm")
    reseal(comparison, manifest, field="comparison_sha256")
    with pytest.raises(ValueError, match="baseline arm"):
        load_quality_comparison(comparison)
