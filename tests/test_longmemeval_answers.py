from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_longmemeval import RecordingScorer, instance
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.longmemeval import evaluate_longmemeval, write_report
from context_router.external.longmemeval_answers import (
    MemoryBudget,
    load_plan,
    prepare_answer_plan,
    prepare_judge_requests,
    score_answer_plan,
    text_hash,
)
from context_router.providers.embedding import HashEmbeddingProvider


def write_lines(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    samples = []
    for i in range(2):
        sample = instance()
        sample.update(
            question_id=f"q{i}",
            question_date="2026/01/04 12:00",
            haystack_dates=["2026/01/01", "2026/01/02", "2026/01/03"],
            answer="REFERENCE-ONLY-SENTINEL",
        )
        # Real benchmark IDs sometimes encode answer status. They must stay private.
        sample["haystack_session_ids"][0] = "answer_gold_hint"
        sample["answer_session_ids"][0] = "answer_gold_hint"
        if i == 1:
            sample["question_type"] = "temporal-reasoning"
        samples.append(sample)
    dataset = tmp_path / "source.json"
    dataset.write_text(json.dumps(samples), encoding="utf-8")
    retrieval = tmp_path / "retrieval.json"
    write_report(
        retrieval, evaluate_longmemeval(dataset, HashEmbeddingProvider(), scorer=RecordingScorer())
    )
    output = tmp_path / "plan"
    prepare_answer_plan(dataset, retrieval, output, budget=MemoryBudget(256))
    return dataset, retrieval, output


def exchanges(
    bundle: tuple[Path, Path, Path],
) -> tuple[Path, Path, Path, list[dict[str, Any]], list[dict[str, Any]]]:
    output = bundle[2]
    plan_path = output / "plan.private.json"
    plan = load_plan(plan_path)
    answers, judgments = [], []
    for row in plan["rows"]:
        answer = {
            "request_id": row["request_id"],
            "prompt_sha256": row["prompt_sha256"],
            "hypothesis": "fixture response",
            "model": "fixture-generator-2026-01-01",
            "generation_config": {"temperature": 0, "max_output_tokens": 128},
            "stop_reason": "stop",
            "input_tokens": None,
            "output_tokens": None,
        }
        answers.append(answer)
        judgments.append(
            {
                "request_id": row["request_id"],
                "answer_sha256": text_hash(answer["hypothesis"]),
                "correct": row["arm"] == "joint",
                "judge_model": "fixture-judge-2026-01-01",
            }
        )
    return (
        plan_path,
        write_lines(output / "answers.jsonl", answers),
        write_lines(output / "judgments.jsonl", judgments),
        answers,
        judgments,
    )


def test_generation_export_has_dates_but_no_labels_or_reference(
    bundle: tuple[Path, Path, Path],
) -> None:
    _, _, directory = bundle
    exported = (directory / "generation.requests.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        "REFERENCE-ONLY-SENTINEL",
        "answer_gold_hint",
        "has_answer",
        "reference",
        '"arm"',
    ):
        assert forbidden not in exported
    assert "2026/01/04 12:00" in exported
    assert "2026/01/01" in exported
    plan = load_plan(directory / "plan.private.json")
    assert all(row["memory_units"] <= 256 for row in plan["rows"])
    assert len(plan["rows"]) == 8
    assert plan["abstention_evaluated"] is False
    status = json.loads((directory / "status.json").read_text())
    assert status["answer_quality"] == "not_measured"
    assert status["network_calls"] == 0


def test_plan_is_deterministic_and_does_not_overwrite(bundle: tuple[Path, Path, Path]) -> None:
    dataset, retrieval, output = bundle
    other = output.with_name("other")
    prepare_answer_plan(dataset, retrieval, other, budget=MemoryBudget(256))
    assert (output / "plan.private.json").read_bytes() == (other / "plan.private.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        prepare_answer_plan(dataset, retrieval, output, budget=MemoryBudget(256))


def test_integer_reference_in_official_temporal_questions_is_supported(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, retrieval, output = bundle
    samples = json.loads(dataset.read_text())
    for sample in samples:
        sample["answer"] = 19
    dataset.write_text(json.dumps(samples))
    write_report(
        retrieval,
        evaluate_longmemeval(dataset, HashEmbeddingProvider(), scorer=RecordingScorer()),
    )
    new_output = output.with_name("integer-reference")
    prepare_answer_plan(dataset, retrieval, new_output, budget=MemoryBudget(256))
    plan = load_plan(new_output / "plan.private.json")
    assert all(row["reference"] == "19" for row in plan["rows"])
    assert all(
        "reference" not in json.loads(line)
        for line in (new_output / "generation.requests.jsonl").read_text().splitlines()
    )


def test_duplicate_dates_are_preserved_without_dropping_questions(
    bundle: tuple[Path, Path, Path],
) -> None:
    dataset, retrieval, output = bundle
    samples = json.loads(dataset.read_text())
    sample = samples[0]
    sample["haystack_session_ids"].append(sample["haystack_session_ids"][0])
    sample["haystack_sessions"].append(copy.deepcopy(sample["haystack_sessions"][0]))
    sample["haystack_dates"].append("2026/01/09")
    dataset.write_text(json.dumps(samples))
    write_report(
        retrieval,
        evaluate_longmemeval(dataset, HashEmbeddingProvider(), scorer=RecordingScorer()),
    )
    new_output = output.with_name("ambiguous-dates")
    status = prepare_answer_plan(dataset, retrieval, new_output, budget=MemoryBudget(12000))
    assert status["questions"] == 2
    assert status["questions_with_ambiguous_session_dates"] == 1
    exported = (new_output / "generation.requests.jsonl").read_text()
    assert "ambiguous; multiple recorded dates: 2026/01/01 | 2026/01/09" in exported
    assert "answer_gold_hint" not in exported


def test_changed_source_and_plan_are_rejected(bundle: tuple[Path, Path, Path]) -> None:
    dataset, retrieval, output = bundle
    dataset.write_text(dataset.read_text() + " ")
    with pytest.raises(ValueError, match="do not match"):
        prepare_answer_plan(dataset, retrieval, output.with_name("new"), budget=MemoryBudget(256))
    path = output / "plan.private.json"
    raw = json.loads(path.read_text())
    raw["rows"][0]["prompt"] += "changed"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="integrity"):
        load_plan(path)


def test_judge_is_blinded_and_reports_only_after_complete_import(
    bundle: tuple[Path, Path, Path],
) -> None:
    plan, answers_file, judgments_file, _, _ = exchanges(bundle)
    output = bundle[2] / "judge.requests.jsonl"
    status = prepare_judge_requests(plan, answers_file, output)
    assert status["answer_quality"] == "not_measured"
    judge_text = output.read_text()
    assert "REFERENCE-ONLY-SENTINEL" in judge_text
    assert '"arm"' not in judge_text
    assert "fixture-generator" not in judge_text
    report = score_answer_plan(plan, answers_file, judgments_file)
    assert report["summary"]["joint"]["accuracy"] == 1
    assert report["summary"]["hybrid"]["accuracy"] == 0
    assert report["paired"][0]["left_wins"] == 2
    assert report["paired"][0]["right_wins"] == 0
    assert report["by_type"]["temporal-reasoning"]["joint"]["count"] == 1
    assert report["sample_is_held_out"] is False
    assert report["generator_token_budget_verified"] is False
    assert report["summary"]["joint"]["reported_usage_rows"] == 0
    assert report["summary"]["joint"]["total_reported_tokens"] is None
    assert report["questions_with_ambiguous_session_dates"] == 0


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "unknown", "prompt", "model", "settings", "usage"]
)
def test_bad_answer_import_cannot_improve_a_score(
    bundle: tuple[Path, Path, Path], problem: str
) -> None:
    plan, answer_file, judgment_file, answers, _ = exchanges(bundle)
    bad = copy.deepcopy(answers)
    if problem == "missing":
        bad.pop()
    elif problem == "duplicate":
        bad.append(bad[0])
    elif problem == "unknown":
        bad[0]["request_id"] = "unknown"
    elif problem == "prompt":
        bad[0]["prompt_sha256"] = "wrong"
    elif problem == "model":
        bad[0]["model"] = "different-model"
    elif problem == "settings":
        bad[0]["generation_config"] = {"temperature": 1}
    else:
        bad[0]["input_tokens"] = -1
    write_lines(answer_file, bad)
    with pytest.raises(ValueError):
        score_answer_plan(plan, answer_file, judgment_file)


@pytest.mark.parametrize("problem", ["hash", "string_boolean", "judge_model", "missing"])
def test_bad_judgments_are_not_silently_accepted(
    bundle: tuple[Path, Path, Path], problem: str
) -> None:
    plan, answer_file, judgment_file, _, judgments = exchanges(bundle)
    if problem == "hash":
        judgments[0]["answer_sha256"] = "stale"
    elif problem == "string_boolean":
        judgments[0]["correct"] = "yes"
    elif problem == "judge_model":
        judgments[0]["judge_model"] = "different-judge"
    else:
        judgments.pop()
    write_lines(judgment_file, judgments)
    with pytest.raises(ValueError):
        score_answer_plan(plan, answer_file, judgment_file)


@pytest.mark.parametrize("stop", ["length", "max_tokens", "incomplete", "error", "content_filter"])
def test_partial_or_failed_response_is_not_counted_correct(
    bundle: tuple[Path, Path, Path], stop: str
) -> None:
    plan, answer_file, judgment_file, answers, judgments = exchanges(bundle)
    answers[0]["stop_reason"] = stop
    judgments[0]["correct"] = True
    write_lines(answer_file, answers)
    write_lines(judgment_file, judgments)
    report = score_answer_plan(plan, answer_file, judgment_file)
    assert report["records"][0]["correct"] is False
    assert report["records"][0]["completed"] is False


def test_exact_token_budget_preserves_original_text_and_ignores_saved_truncation(
    tmp_path: Path,
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0, "Case": 1}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=2)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    budget = MemoryBudget(128, tokenizer_file=path, model="fixture-generator")
    assert budget.count("Case " * 200) == 200
    assert budget.clip("Case 中文 😀", 1) == "Case"
    assert budget.clip("Case 中文 😀", 10) == "Case 中文 😀"


def test_cli_exports_without_models_and_protects_input(bundle: tuple[Path, Path, Path]) -> None:
    source, retrieval, output = bundle
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "longmemeval-answer-plan",
            str(source),
            str(retrieval),
            str(output.with_name("cli")),
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "not measured" in result.output
    plan, answers, judgments, _, _ = exchanges(bundle)
    result = runner.invoke(
        app, ["longmemeval-answer-score", str(plan), str(answers), str(judgments), str(answers)]
    )
    assert result.exit_code != 0
    assert "inputs must remain intact" in result.output
