from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_longmemeval_abstention import refusal_bundle
from test_longmemeval_answers import bundle as bundle
from test_longmemeval_answers import write_lines
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.answer_runner import run_requests
from context_router.external.longmemeval import write_report
from context_router.external.longmemeval_answers import (
    MemoryBudget,
    load_plan,
    object_hash,
    prepare_answer_plan,
    text_hash,
)
from context_router.external.longmemeval_rendering import TURN_RENDERING
from context_router.external.quality_comparison import (
    QualityCriteria,
    load_quality_comparison,
    prepare_quality_comparison,
    prepare_quality_judgments,
    score_quality_comparison,
)


def candidate_plan(source: tuple[Path, Path, Path], *, refusal: bool = False) -> Path:
    dataset, retrieval, original = source
    output = original.with_name(original.name + "-turn")
    prepare_answer_plan(
        dataset,
        retrieval,
        output,
        budget=MemoryBudget(256),
        arms=tuple(load_plan(original / "plan.private.json")["arms"]),
        include_abstention=refusal,
        rendering=TURN_RENDERING,
    )
    return output / "plan.private.json"


@pytest.fixture
def comparison(bundle: tuple[Path, Path, Path]) -> Path:
    original = bundle[2] / "plan.private.json"
    output = bundle[2].with_name("comparison")
    prepare_quality_comparison(original, candidate_plan(bundle), output)
    return output / "comparison.private.json"


def exchange(comparison: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, plans = load_quality_comparison(comparison)
    answers, judgments = [], []
    for side, plan in plans.items():
        for request in plan["rows"]:
            answers.append(
                {
                    "request_id": request["request_id"],
                    "prompt_sha256": request["prompt_sha256"],
                    "hypothesis": "fixture response",
                    "model": "fixture-generator-2026-01-01",
                    "generation_config": {"temperature": 0, "max_output_tokens": 128},
                    "stop_reason": "stop",
                    "input_tokens": None,
                    "output_tokens": None,
                }
            )
            judgments.append(
                {
                    "request_id": request["request_id"],
                    "answer_sha256": text_hash("fixture response"),
                    "correct": side == "candidate" or request["question_id"] == "q0",
                    "judge_model": "fixture-judge-2026-01-01",
                    "judgment_config": {"protocol": "fixture-rubric-v1", "temperature": 0},
                }
            )
    return answers, judgments


def score_rows(
    comparison: Path, answers: list[dict[str, Any]], judgments: list[dict[str, Any]]
) -> dict[str, Any]:
    return score_quality_comparison(
        comparison,
        write_lines(comparison.parent / "answers.jsonl", answers),
        write_lines(comparison.parent / "judgments.jsonl", judgments),
    )


def reseal(path: Path, content: dict[str, Any], field: str = "plan_sha256") -> None:
    content.pop(field)
    content[field] = object_hash(content)
    write_report(path, content)


def test_public_comparison_is_blinded_deterministic_and_offline(
    bundle: tuple[Path, Path, Path], comparison: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, plans = load_quality_comparison(comparison)
    requests = [
        json.loads(line)
        for line in (comparison.parent / "generation.requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 4
    assert len({r["request_id"] for r in requests}) == 4
    assert all(
        set(r) == {"request_id", "prompt_sha256", "instructions", "prompt"} for r in requests
    )
    exported = json.dumps(requests)
    assert "REFERENCE-ONLY-SENTINEL" not in exported
    assert "answer_gold_hint" not in exported
    assert TURN_RENDERING not in exported
    assert manifest["criteria"]["maximum_mean_memory_ratio"] == 1
    assert manifest["sample_is_held_out"] is False
    assert manifest["generation_configuration_frozen"] is False
    assert all(plan["arms"] == ["joint"] for plan in plans.values())
    assert all(r["request_id"] != r["source_request_id"] for p in plans.values() for r in p["rows"])
    second = comparison.parent.with_name("comparison-copy")
    status = prepare_quality_comparison(
        bundle[2] / "plan.private.json",
        bundle[2].with_name("plan-turn") / "plan.private.json",
        second,
    )
    assert status["requests"] == 4
    assert status["answer_quality"] == "not_measured"
    assert status["network_calls"] == 0
    assert manifest["comparison_sha256"] == status["comparison_sha256"]
    assert (second / "generation.requests.jsonl").read_bytes() == (
        comparison.parent / "generation.requests.jsonl"
    ).read_bytes()

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry run must not construct a client")

    monkeypatch.setattr("httpx.Client", forbidden_client)
    output = comparison.parent / "preview.answers.jsonl"
    preview = run_requests(
        comparison.parent / "generation.requests.jsonl",
        output,
        stage="generation",
        model="fixture-generator-2026-01-01",
    )
    assert preview["status"] == "dry_run" and preview["requests"] == 4
    assert not output.exists()
    with pytest.raises(ValueError, match="already exists"):
        prepare_quality_comparison(
            bundle[2] / "plan.private.json", bundle[2] / "plan.private.json", second
        )


def test_identical_prompts_get_distinct_opaque_ids_and_are_not_deduplicated(
    bundle: tuple[Path, Path, Path],
) -> None:
    original = bundle[2] / "plan.private.json"
    candidate = candidate_plan(bundle)
    before, after = load_plan(original), load_plan(candidate)
    by_question = {r["question_id"]: r for r in before["rows"] if r["arm"] == "joint"}
    for row in after["rows"]:
        if row["arm"] == "joint":
            row["prompt"] = by_question[row["question_id"]]["prompt"]
            row["prompt_sha256"] = by_question[row["question_id"]]["prompt_sha256"]
    reseal(candidate, after)
    output = original.parent.with_name("identical-inputs")
    prepare_quality_comparison(original, candidate, output)
    _, plans = load_quality_comparison(output / "comparison.private.json")
    for left, right in zip(plans["baseline"]["rows"], plans["candidate"]["rows"], strict=True):
        assert left["prompt_sha256"] == right["prompt_sha256"]
        assert left["request_id"] != right["request_id"]


def test_paired_scoring_aligns_question_ids_not_source_or_import_order(
    bundle: tuple[Path, Path, Path],
) -> None:
    candidate = candidate_plan(bundle)
    changed = load_plan(candidate)
    changed["question_ids"].reverse()
    changed["rows"].reverse()
    reseal(candidate, changed)
    output = bundle[2].with_name("reversed-comparison")
    prepare_quality_comparison(bundle[2] / "plan.private.json", candidate, output)
    comparison = output / "comparison.private.json"
    answers, judgments = exchange(comparison)
    report = score_rows(comparison, list(reversed(answers)), list(reversed(judgments)))
    pair = report["primary_answerable_comparison"]
    assert pair["questions"] == 2
    assert pair["difference"] == 0.5
    assert pair["left_wins"] == 1 and pair["right_wins"] == 0
    assert pair["exact_mcnemar_p"] == 1
    assert pair["bootstrap_low"] == 0
    assert report["sample_gate"]["status"] == "sample_criteria_not_passed"
    assert report["sample_is_held_out"] is False
    assert report["default_promotion_eligible"] is False
    assert report["plans"]["baseline"]["summary"]["joint"]["accuracy"] == 0.5
    assert report["plans"]["candidate"]["summary"]["joint"]["accuracy"] == 1
    assert report["plans"]["candidate"]["summary"]["joint"]["total_reported_tokens"] is None


def test_judging_batch_hides_side_and_model_and_requires_all_answers(comparison: Path) -> None:
    answers, _ = exchange(comparison)
    answers_file = write_lines(comparison.parent / "answers.jsonl", answers)
    output = comparison.parent / "judge.requests.jsonl"
    status = prepare_quality_judgments(comparison, answers_file, output)
    assert status["requests"] == 4 and status["network_calls"] == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {r["request_id"] for r in rows} == {a["request_id"] for a in answers}
    assert all(
        set(r) == {"request_id", "answer_sha256", "prompt", "answer_completed", "answer_empty"}
        for r in rows
    )
    assert "REFERENCE-ONLY-SENTINEL" in output.read_text()
    assert "fixture-generator" not in output.read_text()
    assert all('"arm"' not in r["prompt"] and '"rendering"' not in r["prompt"] for r in rows)
    write_lines(answers_file, answers[:-1])
    partial_output = output.with_name("partial-judge.jsonl")
    with pytest.raises(ValueError, match="missing"):
        prepare_quality_judgments(comparison, answers_file, partial_output)
    assert not partial_output.exists()


@pytest.mark.parametrize(
    "problem",
    [
        "model",
        "generation_config",
        "judge",
        "judgment_config",
        "self_judge",
        "missing_judge_config",
        "hash",
        "duplicate",
        "missing",
    ],
)
def test_mismatched_or_incomplete_exchanges_cannot_pass_gate(
    comparison: Path, problem: str
) -> None:
    answers, judgments = exchange(comparison)
    if problem == "model":
        answers[-1]["model"] = "other-generator-version"
    elif problem == "generation_config":
        answers[-1]["generation_config"] = {"temperature": 1}
    elif problem == "judge":
        for row in judgments[2:]:
            row["judge_model"] = "other-judge-version"
    elif problem == "judgment_config":
        for row in judgments[2:]:
            row["judgment_config"] = {"temperature": 1}
    elif problem == "self_judge":
        for row in judgments:
            row["judge_model"] = answers[0]["model"]
    elif problem == "missing_judge_config":
        for row in judgments:
            row.pop("judgment_config")
    elif problem == "hash":
        judgments[-1]["answer_sha256"] = "stale"
    elif problem == "duplicate":
        answers.append(answers[0])
    else:
        judgments.pop()
    with pytest.raises(ValueError):
        score_rows(comparison, answers, judgments)


def test_empty_and_truncated_answers_remain_wrong_in_the_paired_denominator(
    comparison: Path,
) -> None:
    answers, judgments = exchange(comparison)
    answers[2]["hypothesis"] = ""
    judgments[2]["answer_sha256"] = text_hash("")
    answers[3]["stop_reason"] = "length"
    report = score_rows(comparison, answers, judgments)
    candidate = report["plans"]["candidate"]["summary"]["joint"]
    assert candidate["count"] == 2
    assert candidate["accuracy"] == 0
    assert candidate["truncated_rate"] == 0.5
    assert candidate["failed_completion_rate"] == 1
    assert report["primary_answerable_comparison"]["questions"] == 2
    assert report["sample_gate"]["status"] == "sample_criteria_not_passed"


@pytest.mark.parametrize(
    "problem",
    [
        "budget",
        "retrieval",
        "question",
        "reference",
        "sessions",
        "refusal",
        "missing_row",
        "same_rendering",
        "prompt",
    ],
)
def test_only_matched_rendering_inputs_are_accepted(
    bundle: tuple[Path, Path, Path], problem: str
) -> None:
    original = bundle[2] / "plan.private.json"
    candidate = candidate_plan(bundle)
    plan = load_plan(candidate)
    row = next(r for r in plan["rows"] if r["arm"] == "joint")
    if problem == "budget":
        plan["budget"]["maximum"] += 1
    elif problem == "retrieval":
        plan["retrieval_sha256"] = "different"
    elif problem == "question":
        row["question"] = "another question"
    elif problem == "reference":
        row["reference"] = "another reference"
    elif problem == "sessions":
        row["selected_session_ids"] = []
    elif problem == "refusal":
        row["expects_refusal"] = True
    elif problem == "missing_row":
        plan["rows"].remove(row)
    elif problem == "same_rendering":
        plan["rendering"] = load_plan(original)["rendering"]
    else:
        row["prompt"] = "changed without updating the prompt hash"
    reseal(candidate, plan)
    output = bundle[2].with_name("invalid-comparison")
    with pytest.raises(ValueError):
        prepare_quality_comparison(original, candidate, output)
    assert not output.exists()


@pytest.mark.parametrize("artifact", ["manifest", "frozen_plan", "requests"])
def test_frozen_artifacts_cannot_be_silently_changed(comparison: Path, artifact: str) -> None:
    if artifact == "manifest":
        content = json.loads(comparison.read_text())
        content["criteria"]["minimum_accuracy_gain"] = 0.5
        write_report(comparison, content)
    elif artifact == "frozen_plan":
        path = comparison.parent / "candidate.plan.private.json"
        content = load_plan(path)
        content["rows"][0]["memory_units"] += 1
        reseal(path, content)
    else:
        path = comparison.parent / "generation.requests.jsonl"
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed|integrity"):
        load_quality_comparison(comparison)


@pytest.mark.parametrize("only", [False, True])
def test_refusal_keeps_its_own_denominator_and_never_creates_an_answerable_gain(
    bundle: tuple[Path, Path, Path], only: bool
) -> None:
    source = refusal_bundle(bundle, only=only)
    output = source[2].with_name("refusal-comparison")
    prepare_quality_comparison(
        source[2] / "plan.private.json",
        candidate_plan(source, refusal=True),
        output,
        arm="hybrid",
    )
    comparison = output / "comparison.private.json"
    answers, judgments = exchange(comparison)
    report = score_rows(comparison, answers, judgments)
    assert report["abstention_comparison"]["questions"] == 1
    assert report["plans"]["candidate"]["abstention_summary"]["hybrid"]["count"] == 1
    if only:
        assert report["primary_answerable_comparison"] is None
        assert report["sample_gate"]["status"] == "not_measured_no_answerable_questions"
        assert report["plans"]["candidate"]["summary"]["hybrid"]["accuracy"] is None
    else:
        assert report["primary_answerable_comparison"]["questions"] == 2
    assert report["default_promotion_eligible"] is False


def test_frozen_criteria_and_independent_validation_are_separate(
    bundle: tuple[Path, Path, Path],
) -> None:
    output = bundle[2].with_name("strict-criteria")
    prepare_quality_comparison(
        bundle[2] / "plan.private.json",
        candidate_plan(bundle),
        output,
        criteria=QualityCriteria(minimum_accuracy_gain=1, maximum_mean_memory_ratio=10),
    )
    comparison = output / "comparison.private.json"
    answers, judgments = exchange(comparison)
    for judgment in judgments[:2]:
        judgment["correct"] = False
    report = score_rows(comparison, answers, judgments)
    assert report["primary_answerable_comparison"]["difference"] == 1
    assert report["sample_gate"]["conditions"]["accuracy_interval_above_minimum_gain"] is False
    assert report["default_promotion_eligible"] is False


@pytest.mark.parametrize("condition", ["pass", "memory", "empty"])
def test_sample_gate_checks_quality_cost_and_failures_without_promoting(
    bundle: tuple[Path, Path, Path],
    condition: str,
) -> None:
    output = bundle[2].with_name("criteria-comparison")
    prepare_quality_comparison(
        bundle[2] / "plan.private.json",
        candidate_plan(bundle),
        output,
        criteria=QualityCriteria(
            maximum_mean_memory_ratio=0.01 if condition == "memory" else 10,
            maximum_exact_mcnemar_p=1,
        ),
    )
    comparison = output / "comparison.private.json"
    answers, judgments = exchange(comparison)
    for judgment in judgments[:2]:
        judgment["correct"] = False
    if condition == "empty":
        answers[-1]["hypothesis"] = ""
        judgments[-1]["answer_sha256"] = text_hash("")
    report = score_rows(comparison, answers, judgments)
    gate = report["sample_gate"]
    if condition == "pass":
        assert gate["status"] == "sample_criteria_passed"
        assert all(gate["conditions"].values())
    elif condition == "memory":
        assert gate["conditions"]["accuracy_interval_above_minimum_gain"] is True
        assert gate["conditions"]["mean_memory_ratio_within_limit"] is False
    else:
        assert gate["conditions"]["failed_completion_increase_within_limit"] is False
    assert report["default_promotion_eligible"] is False


def test_refusal_regression_blocks_a_positive_answerable_sample_gate(
    bundle: tuple[Path, Path, Path],
) -> None:
    source = refusal_bundle(bundle)
    output = source[2].with_name("refusal-regression")
    prepare_quality_comparison(
        source[2] / "plan.private.json",
        candidate_plan(source, refusal=True),
        output,
        arm="hybrid",
        criteria=QualityCriteria(maximum_mean_memory_ratio=10, maximum_exact_mcnemar_p=1),
    )
    comparison = output / "comparison.private.json"
    _, plans = load_quality_comparison(comparison)
    answers, judgments = exchange(comparison)
    by_id = {j["request_id"]: j for j in judgments}
    for side, plan in plans.items():
        for request in plan["rows"]:
            by_id[request["request_id"]]["correct"] = (
                side == "baseline" if request["expects_refusal"] else side == "candidate"
            )
    report = score_rows(comparison, answers, judgments)
    assert report["primary_answerable_comparison"]["difference"] == 1
    assert report["abstention_comparison"]["difference"] == -1
    assert report["sample_gate"]["status"] == "sample_criteria_not_passed"
    assert (
        report["sample_gate"]["conditions"]["no_observed_abstention_accuracy_regression"] is False
    )


def test_mcnemar_is_required_even_when_tiny_sample_bootstrap_is_positive(
    bundle: tuple[Path, Path, Path],
) -> None:
    output = bundle[2].with_name("tiny-sample")
    prepare_quality_comparison(
        bundle[2] / "plan.private.json",
        candidate_plan(bundle),
        output,
        criteria=QualityCriteria(maximum_mean_memory_ratio=10),
    )
    comparison = output / "comparison.private.json"
    answers, judgments = exchange(comparison)
    for judgment in judgments[:2]:
        judgment["correct"] = False
    report = score_rows(comparison, answers, judgments)
    assert report["primary_answerable_comparison"]["bootstrap_low"] == 1
    assert report["primary_answerable_comparison"]["exact_mcnemar_p"] == 0.5
    assert report["sample_gate"]["conditions"]["exact_mcnemar_within_limit"] is False
    assert report["sample_gate"]["status"] == "sample_criteria_not_passed"


def test_changed_public_content_must_match_frozen_prompts_even_if_file_hash_is_updated(
    comparison: Path,
) -> None:
    from context_router.external.longmemeval_answers import file_hash

    requests_file = comparison.parent / "generation.requests.jsonl"
    rows = [json.loads(line) for line in requests_file.read_text().splitlines()]
    rows[0]["prompt"] = "changed prompt"
    rows[0]["prompt_sha256"] = object_hash(
        {"instructions": rows[0]["instructions"], "prompt": rows[0]["prompt"]}
    )
    write_lines(requests_file, rows)
    manifest = json.loads(comparison.read_text())
    manifest["generation_requests_sha256"] = file_hash(requests_file)
    reseal(comparison, manifest, "comparison_sha256")
    with pytest.raises(ValueError, match="reproduce"):
        load_quality_comparison(comparison)


def test_cli_exports_a_blinded_comparison_and_judging_batch(
    bundle: tuple[Path, Path, Path],
    comparison: Path,
) -> None:
    runner = CliRunner()
    output = comparison.parent.with_name("cli-comparison")
    result = runner.invoke(
        app,
        [
            "longmemeval-quality-plan",
            str(bundle[2] / "plan.private.json"),
            str(bundle[2].with_name("plan-turn") / "plan.private.json"),
            str(output),
            "--minimum-accuracy-gain",
            "0.05",
        ],
    )
    assert result.exit_code == 0, result.output
    exported = output / "comparison.private.json"
    manifest, _ = load_quality_comparison(exported)
    assert manifest["criteria"]["minimum_accuracy_gain"] == 0.05
    answers, _ = exchange(exported)
    answers_file = write_lines(output / "answers.jsonl", answers)
    judge_file = output / "judge.requests.jsonl"
    result = runner.invoke(
        app, ["longmemeval-quality-judge-plan", str(exported), str(answers_file), str(judge_file)]
    )
    assert result.exit_code == 0, result.output
    assert len(judge_file.read_text().splitlines()) == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_gate_criteria_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        QualityCriteria(minimum_accuracy_gain=value)


def test_cli_runs_the_complete_fixture_exchange_and_protects_report_inputs(
    comparison: Path,
) -> None:
    answers, judgments = exchange(comparison)
    answers_file = write_lines(comparison.parent / "answers.jsonl", answers)
    judgments_file = write_lines(comparison.parent / "judgments.jsonl", judgments)
    runner = CliRunner()
    output = comparison.parent / "quality.json"
    arguments = [
        "longmemeval-quality-score",
        str(comparison),
        str(answers_file),
        str(judgments_file),
    ]
    result = runner.invoke(app, [*arguments, str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["answer_quality"] == "judged_reference_accuracy"
    assert "exploratory" in result.output
    assert runner.invoke(app, [*arguments, str(output)]).exit_code != 0
    protected = comparison.parent / "answers.jsonl.tmp"
    write_lines(protected, answers)
    report_path = comparison.parent / "answers.jsonl"
    report_path.unlink()
    before = protected.read_bytes()
    assert (
        runner.invoke(
            app, [*arguments[:2], str(protected), str(judgments_file), str(report_path)]
        ).exit_code
        != 0
    )
    assert protected.read_bytes() == before
