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
