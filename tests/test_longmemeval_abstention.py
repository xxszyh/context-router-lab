from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_longmemeval import instance
from test_longmemeval_answers import bundle as bundle
from test_longmemeval_answers import exchanges, write_lines
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


@pytest.mark.parametrize("empty_gold", [False, True])
def test_abstention_never_inflates_recall_and_scope_can_extend_checkpoint(
    tmp_path: Path, empty_gold: bool
) -> None:
    refusal = instance()
    refusal["question_id"] = "refusal_abs"
    if empty_gold:
        refusal["answer_session_ids"] = []
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps([refusal, instance()]))
    checkpoint = tmp_path / "retrieval.checkpoint.jsonl"
    baseline = evaluate_longmemeval(source, HashEmbeddingProvider(), checkpoint=checkpoint)
    assert baseline["summary"]["questions"] == 1

    class CountingEmbedder(HashEmbeddingProvider):
        calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return super().embed(texts)

    embedder = CountingEmbedder()
    combined = evaluate_longmemeval(
        source, embedder, checkpoint=checkpoint, resume=True, include_abstention=True
    )
    assert combined["summary"] == baseline["summary"]
    assert combined["records"] == baseline["records"]
    assert combined["abstention_questions"] == 1
    assert embedder.calls > 0
    refusal_record = combined["abstention_records"][0]
    assert refusal_record["candidate_all_gold"] is None
    assert refusal_record["oracle_all_at_depth"] is None
    assert all(arm["metrics"] is None for arm in refusal_record["arms"].values())
    calls = embedder.calls
    only = evaluate_longmemeval(
        source, embedder, checkpoint=checkpoint, resume=True, abstention_only=True
    )
    assert embedder.calls == calls
    assert only["summary"] == {"questions": 0, "arms": {}}
    assert only["records"] == []
    assert only["abstention_records"] == combined["abstention_records"]


def refusal_bundle(
    bundle: tuple[Path, Path, Path], *, only: bool = False
) -> tuple[Path, Path, Path]:
    source, retrieval, output = bundle
    samples = json.loads(source.read_text())
    refusal = copy.deepcopy(samples[0])
    refusal["question_id"] = "private-refusal_abs"
    refusal["answer"] = "REFUSAL-REFERENCE-SENTINEL"
    source.write_text(json.dumps([refusal] if only else [*samples, refusal]))
    write_report(
        retrieval,
        evaluate_longmemeval(source, HashEmbeddingProvider(), include_abstention=True),
    )
    refusal_output = output.with_name("refusal-plan")
    status = prepare_answer_plan(
        source,
        retrieval,
        refusal_output,
        budget=MemoryBudget(256),
        arms=("hybrid", "router", "query_only"),
        include_abstention=True,
    )
    assert status["abstention_questions"] == 1
    assert status["answer_quality"] == "not_measured"
    return source, retrieval, refusal_output


@pytest.mark.parametrize("only", [False, True])
def test_refusal_is_private_until_judging_and_has_a_separate_denominator(
    bundle: tuple[Path, Path, Path], only: bool
) -> None:
    prepared = refusal_bundle(bundle, only=only)
    plan_file, answers_file, judgments_file, answers, judgments = exchanges(prepared)
    plan = load_plan(plan_file)
    exported = (prepared[2] / "generation.requests.jsonl").read_text()
    for forbidden in ("private-refusal_abs", "REFUSAL-REFERENCE-SENTINEL", "expects_refusal"):
        assert forbidden not in exported
    by_id = {row["request_id"]: row for row in plan["rows"]}
    for answer, judgment in zip(answers, judgments, strict=True):
        if by_id[answer["request_id"]]["expects_refusal"]:
            answer["hypothesis"] = "The available conversation does not establish that fact."
            judgment["answer_sha256"] = text_hash(answer["hypothesis"])
            judgment["correct"] = True
    write_lines(answers_file, answers)
    write_lines(judgments_file, judgments)
    judge_file = prepared[2] / "judge.requests.jsonl"
    prepare_judge_requests(plan_file, answers_file, judge_file)
    refusal_prompts = [
        json.loads(line)["prompt"]
        for line in judge_file.read_text().splitlines()
        if "REFUSAL-REFERENCE-SENTINEL" in line
    ]
    assert len(refusal_prompts) == 3
    assert all("cannot be determined" in prompt for prompt in refusal_prompts)
    report = score_answer_plan(plan_file, answers_file, judgments_file)
    assert report["abstention_evaluated"] is True
    assert report["summary"]["hybrid"]["count"] == (0 if only else 2)
    assert report["summary"]["hybrid"]["accuracy"] == (None if only else 0)
    assert report["abstention_summary"]["hybrid"]["count"] == 1
    assert report["abstention_summary"]["hybrid"]["accuracy"] == 1
    # Silence is a failed answer, even when the question requires abstention.
    refusal_answer = next(a for a in answers if by_id[a["request_id"]]["expects_refusal"])
    refusal_answer["hypothesis"] = "   "
    next(j for j in judgments if j["request_id"] == refusal_answer["request_id"])[
        "answer_sha256"
    ] = text_hash("   ")
    write_lines(answers_file, answers)
    write_lines(judgments_file, judgments)
    changed = score_answer_plan(plan_file, answers_file, judgments_file)
    arm = by_id[refusal_answer["request_id"]]["arm"]
    assert changed["abstention_summary"][arm]["accuracy"] == 0


def test_judgment_settings_and_failure_policy_cannot_disguise_a_completed_answer(
    bundle: tuple[Path, Path, Path],
) -> None:
    plan, answer_file, judgment_file, _, judgments = exchanges(bundle)
    judgments[0]["judgment_config"] = {"temperature": 0}
    write_lines(judgment_file, judgments)
    with pytest.raises(ValueError, match="same configuration"):
        score_answer_plan(plan, answer_file, judgment_file)
    judgments[0].pop("judgment_config")
    judgments[0]["judgment_origin"] = "failed_generation_policy"
    write_lines(judgment_file, judgments)
    with pytest.raises(ValueError, match="completed nonempty"):
        score_answer_plan(plan, answer_file, judgment_file)


def test_cli_can_export_a_refusal_only_plan(bundle: tuple[Path, Path, Path]) -> None:
    source, retrieval, output = refusal_bundle(bundle, only=True)
    result = CliRunner().invoke(
        app,
        [
            "longmemeval",
            str(source),
            str(retrieval),
            "--abstention-only",
        ],
    )
    assert result.exit_code == 0, result.output
    result = CliRunner().invoke(
        app,
        [
            "longmemeval-answer-plan",
            str(source),
            str(retrieval),
            str(output.with_name("cli-refusal")),
            "--include-abstention",
            "--arm",
            "hybrid",
            "--arm",
            "query_only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Prepared 2 requests for 1 questions" in result.output
