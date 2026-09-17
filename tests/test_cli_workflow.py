from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from context_router.cli import app
from context_router.datasets import queries_per_session

runner = CliRunner()


def test_cli_generates_ingests_validates_and_benchmarks_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    generated = runner.invoke(app, ["generate-synthetic", str(dataset_dir), "--sessions", "2"])
    assert generated.exit_code == 0, generated.output

    database = tmp_path / "events.db"
    ingested = runner.invoke(app, ["ingest", str(database), str(dataset_dir / "records.jsonl")])
    assert ingested.exit_code == 0, ingested.output

    validated = runner.invoke(
        app,
        [
            "validate-dataset",
            str(database),
            str(dataset_dir / "benchmark.jsonl"),
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert '"valid": true' in validated.output.lower()

    output = tmp_path / "results.json"
    benchmarked = runner.invoke(
        app,
        [
            "benchmark",
            str(database),
            str(dataset_dir / "benchmark.jsonl"),
            str(output),
        ],
    )
    assert benchmarked.exit_code == 0, benchmarked.output
    results = json.loads(output.read_text(encoding="utf-8"))
    assert results["metrics"]["count"] == queries_per_session() * 2
    assert "micro_recall" in results["metrics"]

    report = tmp_path / "report.md"
    reported = runner.invoke(app, ["report", str(output), str(report)])
    assert reported.exit_code == 0, reported.output
    assert "Context Router Benchmark" in report.read_text(encoding="utf-8")


def test_judge_answers_applies_the_double_refusal_gate_by_default(tmp_path: Path) -> None:
    answers = tmp_path / "answers.json"
    answers.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": "s-01",
                        "arm": "hybrid_router",
                        "query": "What happened?",
                        "answer_requirements": ["Explain the implementation"],
                        "answer": "上下文不足，无法回答。",
                        "must_abstain": True,
                    },
                    {
                        "sample_id": "s-01",
                        "arm": "query_recent_only",
                        "query": "What happened?",
                        "answer_requirements": ["Explain the implementation"],
                        "answer": "There is not enough information to answer.",
                        "must_abstain": True,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "judge.json"

    result = runner.invoke(
        app,
        [
            "judge-answers",
            str(answers),
            str(output),
            "--judge",
            "coverage",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["refusal_gate"] is True
    assert payload["refusal_gated_pairs"] == 1
    assert payload["refusal_gate_hits"] == 1
    assert payload["agreed"] == 0
    assert payload["judged_pairs"] == 0
    assert payload["outcomes"][0]["winner"] == "tie"
    assert payload["outcomes"][0]["agreement"] is None
    assert payload["tally"] == {}, "the primary table excludes must-refuse checkpoints"
    assert "overall_tally" not in payload, "refusal checkpoints must not re-enter a mixed table"
    assert payload["strata"]["checkpoint"]["must_refuse"]["pairs"] == 1
    assert payload["strata"]["checkpoint"]["must_refuse"]["judged_pairs"] == 0
    assert payload["strata"]["checkpoint"]["must_refuse"]["order_agreement"] is None
    assert payload["strata"]["response"]["both_refuse"]["pairs"] == 1


def test_judge_answers_rejects_missing_or_inconsistent_abstain_labels(tmp_path: Path) -> None:
    answers = tmp_path / "answers.json"
    answers.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": "s-01",
                        "arm": "hybrid_router",
                        "query": "q",
                        "answer_requirements": ["r"],
                        "answer": "a",
                        "must_abstain": True,
                    },
                    {
                        "sample_id": "s-01",
                        "arm": "query_recent_only",
                        "query": "q",
                        "answer_requirements": ["r"],
                        "answer": "b",
                        "must_abstain": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["judge-answers", str(answers), str(tmp_path / "judge.json"), "--judge", "coverage"],
    )

    assert result.exit_code != 0
    assert "must_abstain labels disagree" in result.output

    payload = json.loads(answers.read_text(encoding="utf-8"))
    del payload["records"][1]["must_abstain"]
    answers.write_text(json.dumps(payload), encoding="utf-8")
    missing = runner.invoke(
        app,
        ["judge-answers", str(answers), str(tmp_path / "judge.json"), "--judge", "coverage"],
    )
    assert missing.exit_code != 0
    assert "need an explicit boolean must_abstain label" in missing.output
