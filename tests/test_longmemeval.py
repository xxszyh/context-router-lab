from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.longmemeval import (
    evaluate_instance,
    evaluate_longmemeval,
    evidence_metrics,
    iter_instances,
    rerank_sessions,
    session_passages,
)
from context_router.providers.embedding import HashEmbeddingProvider


def instance() -> dict[str, Any]:
    return {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "Which archiver and exporter did I choose?",
        "answer": "zip and csv",
        "haystack_session_ids": ["archive", "export", "noise"],
        "haystack_sessions": [
            [{"role": "user", "content": "The archiver uses zip.", "has_answer": True}],
            [{"role": "user", "content": "The exporter uses csv.", "has_answer": True}],
            [{"role": "user", "content": "Today I walked to the park."}],
        ],
        "answer_session_ids": ["archive", "export"],
    }


def test_any_gold_does_not_claim_multi_session_success() -> None:
    metrics = evidence_metrics({"a", "b"}, ["a", "noise"])
    assert metrics == {"any": 1.0, "all": 0.0, "coverage": 0.5}
    with pytest.raises(ValueError, match="abstention"):
        evidence_metrics(set(), [])


def test_missing_or_duplicated_gold_session_is_not_silently_dropped() -> None:
    sample = instance()
    sample["answer_session_ids"] = ["archive", "missing"]
    with pytest.raises(ValueError, match="missing from the haystack"):
        evaluate_instance(sample, HashEmbeddingProvider())
    sample = instance()
    sample["haystack_session_ids"] = ["archive", "archive", "noise"]
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_instance(sample, HashEmbeddingProvider())


def test_answers_and_turn_annotations_cannot_change_retrieval() -> None:
    sample = instance()
    poisoned = copy.deepcopy(sample)
    poisoned["answer"] = "secret-sentinel"
    for session in poisoned["haystack_sessions"]:
        for turn in session:
            turn["has_answer"] = "secret-sentinel"
    before = evaluate_instance(sample, HashEmbeddingProvider())
    after = evaluate_instance(poisoned, HashEmbeddingProvider())
    assert before == after
    assert "secret-sentinel" not in json.dumps(after)


def test_identical_repeated_sessions_are_collapsed_like_the_cleaned_release() -> None:
    sample = instance()
    before = evaluate_instance(sample, HashEmbeddingProvider())
    sample["haystack_session_ids"].append("archive")
    sample["haystack_sessions"].append(copy.deepcopy(sample["haystack_sessions"][0]))
    after = evaluate_instance(sample, HashEmbeddingProvider())
    assert after["arms"] == before["arms"]
    assert after["haystack_sessions"] == before["haystack_sessions"]
    assert after["duplicate_sessions_collapsed"] == 1


class RecordingScorer:
    model_version = "test-pair-scorer"

    def __init__(self) -> None:
        self.passages: list[str] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.passages = passages
        return [10.0 if "right-evidence" in text else -1.0 for text in passages]


def test_joint_scoring_reaches_evidence_beyond_the_session_prefix() -> None:
    scorer = RecordingScorer()
    documents = {
        "a": "irrelevant " * 100 + "archiver right-evidence zip",
        "b": "archiver archiver archiver misleading text",
    }
    ordered, scores = rerank_sessions(
        "archiver",
        documents,
        ["b", "a"],
        scorer,
        passage_words=10,
        passage_overlap=2,
        passages_per_session=1,
    )
    assert ordered == ["a", "b"]
    assert scores["a"] > scores["b"]
    assert all(len(text.split()) <= 10 for text in scorer.passages)
    assert session_passages("") == [""]


@pytest.mark.parametrize("values", [[1.0], [1.0, float("nan")]])
def test_invalid_joint_scores_are_rejected(values: list[float]) -> None:
    class BadScorer:
        model_version = "bad"

        def score(self, query: str, passages: list[str]) -> list[float]:
            return values

    with pytest.raises(ValueError, match="one finite score"):
        rerank_sessions("archiver", {"a": "zip", "b": "csv"}, ["a", "b"], BadScorer())


def test_streaming_reader_rejects_wrong_top_level_and_truncated_data(tmp_path: Path) -> None:
    source = tmp_path / "data.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        list(iter_instances(source))
    source.write_text('[{"question":', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_instances(source))


def test_cli_writes_configured_paired_report_and_rejects_overwriting_data(tmp_path: Path) -> None:
    source = tmp_path / "data.json"
    source.write_text(json.dumps([instance()]), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["longmemeval", str(source), str(source)])
    assert result.exit_code != 0
    assert isinstance(json.loads(source.read_text()), list)
    output = tmp_path / "report.json"
    result = runner.invoke(app, ["longmemeval", str(source), str(output), "--depth", "2"])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report["answer_quality"] == "not_measured"
    assert report["primary_metric"] == "all_gold_session_recall"
    assert report["summary"]["questions"] == 1
    assert len(report["dataset_sha256"]) == 64
    for decomposition in report["summary"]["decomposition"].values():
        assert sum(decomposition.values()) == 1


def test_abstention_is_counted_separately_and_empty_selection_fails(tmp_path: Path) -> None:
    source = tmp_path / "data.json"
    abstention = instance()
    abstention["question_id"] = "q_abs"
    abstention["answer_session_ids"] = []
    source.write_text(json.dumps([abstention, instance()]), encoding="utf-8")
    report = evaluate_longmemeval(source, HashEmbeddingProvider())
    assert report["skipped_abstention"] == 1
    assert report["summary"]["questions"] == 1
    with pytest.raises(ValueError, match="no answerable questions"):
        evaluate_longmemeval(source, HashEmbeddingProvider(), types=("not-a-type",))
