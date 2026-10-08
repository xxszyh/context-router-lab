from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_answer_runner import completion, read_rows, write_requests
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.answer_runner import run_requests
from context_router.external.batch_requests import read_requests
from context_router.external.context_capacity import (
    prepare_capacity_report,
    validate_capacity_report,
)
from context_router.external.longmemeval_answers import object_hash


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    backend.enable_truncation(2)
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        model_max_length=2,
        chat_template=(
            "{% for message in messages %}{{ message.role }} {{ message.content }} "
            "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
        ),
    )
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    return directory


def capacity(
    source: Path, tokenizer_dir: Path, *, window: int = 64, stage: str = "generation"
) -> Path:
    path = source.with_name(f"capacity-{stage}-{window}.json")
    prepare_capacity_report(
        source,
        path,
        model="fixture-model",
        tokenizer_directory=tokenizer_dir,
        context_window=window,
        max_output_tokens=8,
        stage="generation" if stage == "generation" else "judge",
    )
    return path


def validate(source: Path, report: Path) -> tuple[str, dict[str, int]]:
    rows, digest = read_requests(source, "generation")
    return validate_capacity_report(
        report,
        rows,
        requests_sha256=digest,
        stage="generation",
        model="fixture-model",
        max_output_tokens=8,
    )


def reseal(path: Path, data: dict[str, Any]) -> None:
    data.pop("capacity_sha256")
    data["capacity_sha256"] = object_hash(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_chat_count_includes_messages_and_assistant_prefix_without_stored_truncation(
    tmp_path: Path, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("offline counting must not construct a network client")

    monkeypatch.setattr(httpx, "Client", forbidden)
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    report = capacity(source, tokenizer_dir)
    data = json.loads(report.read_text())
    # system (1), instruction (4), user (1), question (3), assistant prefix (1).
    assert data["rows"][0]["input_tokens"] == 10
    assert data["rows"][0]["required_tokens"] == 18
    assert data["network_calls"] == 0 and data["model_weights_loaded"] is False
    assert data["service_tokenization_verified"] is False
    assert data["counter"]["truncation"] is False
    assert validate(source, report)[1] == {"request-0": 10}
    with pytest.raises(ValueError, match="new output"):
        capacity(source, tokenizer_dir)


def test_one_oversized_item_blocks_all_calls_even_outside_the_scheduled_prefix(
    tmp_path: Path, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=2)
    rows = read_rows(source)
    rows[1]["prompt"] = "long " * 100
    rows[1]["prompt_sha256"] = object_hash({k: rows[1][k] for k in ("instructions", "prompt")})
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = capacity(source, tokenizer_dir)
    assert json.loads(report.read_text())["overflow_requests"] == 1

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("blocked input constructed a client")

    monkeypatch.setattr(httpx, "Client", forbidden)
    output = tmp_path / "not-created" / "answers.jsonl"
    with pytest.raises(ValueError, match="1 requests exceed"):
        run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
            max_requests=1,
        )
    assert not output.parent.exists()


@pytest.mark.parametrize("window,fit", [(18, True), (17, False)])
def test_context_boundary_includes_output_reserve(
    tmp_path: Path, tokenizer_dir: Path, window: int, fit: bool
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    report = capacity(source, tokenizer_dir, window=window)
    assert json.loads(report.read_text())["all_fit"] is fit


@pytest.mark.parametrize("change", ["requests", "model", "reserve", "assets", "stage"])
def test_stale_capacity_binding_is_rejected_before_any_output(
    tmp_path: Path, tokenizer_dir: Path, change: str
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    report = capacity(source, tokenizer_dir)
    kwargs: dict[str, Any] = {}
    if change == "requests":
        source.write_text(source.read_text() + "\n")
    elif change == "model":
        kwargs["model"] = "other-model"
    elif change == "reserve":
        kwargs["max_output_tokens"] = 9
    elif change == "assets":
        path = tokenizer_dir / "tokenizer_config.json"
        path.write_text(path.read_text() + "\n")
    else:
        rows, digest = read_requests(source, "generation")
        with pytest.raises(ValueError, match="match"):
            validate_capacity_report(
                report,
                rows,
                requests_sha256=digest,
                stage="judge",
                model="fixture-model",
                max_output_tokens=8,
            )
        return
    output = tmp_path / "absent" / "answers.jsonl"
    with pytest.raises(ValueError, match="match|changed"):
        run_requests(
            source,
            output,
            stage="generation",
            execute=True,
            capacity_report=report,
            **{"model": "fixture-model", "max_output_tokens": 8, **kwargs},
        )
    assert not output.parent.exists()


@pytest.mark.parametrize("usage", [10, 9, None])
def test_guarded_execution_requires_observed_input_usage_to_match_preflight(
    tmp_path: Path, tokenizer_dir: Path, usage: int | None
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    report = capacity(source, tokenizer_dir)
    output = tmp_path / "answers.jsonl"

    def serve(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "system", "content": "Use only the fixture."},
            {"role": "user", "content": "Fixture question 0"},
        ]
        payload = completion(request).json()
        payload["usage"]["prompt_tokens"] = usage
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        result = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
            client=client,
        )
    row = read_rows(output)[0]
    assert result["network_calls"] == 1
    assert result["failed"] == (0 if usage == 10 else 1)
    assert row["hypothesis"] == ("fixture answer" if usage == 10 else "")
    assert row["stop_reason"] == ("stop" if usage == 10 else "error")
    assert (
        row["generation_config"]["capacity_report_sha256"]
        == json.loads(report.read_text())["capacity_sha256"]
    )


def test_capacity_configuration_cannot_change_mid_resume(
    tmp_path: Path, tokenizer_dir: Path
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    report = capacity(source, tokenizer_dir)
    other = capacity(source, tokenizer_dir, window=128)
    output = tmp_path / "answers.jsonl"
    with httpx.Client(transport=httpx.MockTransport(completion)) as client:
        run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
            client=client,
        )
    original = output.read_bytes()
    with pytest.raises(ValueError, match="checkpoint configuration"):
        run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            resume=True,
            capacity_report=other,
            max_output_tokens=8,
        )
    assert output.read_bytes() == original


def test_usage_mismatch_stops_remaining_batch_and_requires_explicit_retry(
    tmp_path: Path, tokenizer_dir: Path
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=2)
    report = capacity(source, tokenizer_dir)
    output = tmp_path / "answers.jsonl"
    calls = 0

    def serve(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = completion(request).json()
        payload["usage"]["prompt_tokens"] = 9 if calls == 1 else 10
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        first = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
            client=client,
        )
        assert first["network_calls"] == 1 and first["remaining"] == 1
        assert len(read_rows(output)) == 1 and read_rows(output)[0]["stop_reason"] == "error"
        second = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
            client=client,
            resume=True,
            retry_failures=True,
        )
        assert second["network_calls"] == 2 and second["remaining"] == second["failed"] == 0
        assert calls == 3
        assert all(row["stop_reason"] == "stop" for row in read_rows(output))


def test_missing_chat_template_has_no_fallback_or_network(
    tmp_path: Path, tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for path in tokenizer_dir.glob("chat_template*"):
        if path.is_file():
            path.unlink()
    path = tokenizer_dir / "tokenizer_config.json"
    configuration = json.loads(path.read_text())
    configuration.pop("chat_template", None)
    path.write_text(json.dumps(configuration))
    source = write_requests(tmp_path / "requests.jsonl", count=1)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("missing local template must not fetch a fallback")

    monkeypatch.setattr(httpx, "Client", forbidden)
    with pytest.raises(ValueError, match="template"):
        capacity(source, tokenizer_dir)
    assert not (tmp_path / "capacity-generation-64.json").exists()


def test_policy_zero_judging_has_no_context_demand_or_http_call(
    tmp_path: Path, tokenizer_dir: Path
) -> None:
    source = write_requests(tmp_path / "judge.jsonl", count=1, stage="judge")
    row = read_rows(source)[0]
    row["answer_completed"] = False
    row["prompt"] = "long " * 500
    source.write_text(json.dumps(row) + "\n")
    report = capacity(source, tokenizer_dir, window=9, stage="judge")
    data = json.loads(report.read_text())
    assert data["all_fit"] is True and data["rows"][0]["input_tokens"] == 0
    assert data["rows"][0]["required_tokens"] == 0

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("policy zero must not make a call")

    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        result = run_requests(
            source,
            tmp_path / "judgments.jsonl",
            stage="judge",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
            client=client,
        )
    assert result["network_calls"] == 0 and result["policy_zero_judgments"] == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "count", "required", "summary", "hash"])
def test_invalid_report_cannot_enable_execution(
    tmp_path: Path, tokenizer_dir: Path, change: str
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    report = capacity(source, tokenizer_dir)
    data = json.loads(report.read_text())
    if change == "missing":
        data["rows"] = []
    elif change == "duplicate":
        data["rows"].append(data["rows"][0])
    elif change == "count":
        data["rows"][0]["input_tokens"] = -1
    elif change == "required":
        data["rows"][0]["required_tokens"] -= 1
    elif change == "summary":
        data["overflow_requests"] = 1
    else:
        data["context_window"] += 1
    if change != "hash":
        reseal(report, data)
    else:
        report.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate(source, report)


def test_capacity_cli_writes_blocked_report_but_runner_performs_no_calls(
    tmp_path: Path, tokenizer_dir: Path
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    output = tmp_path / "cli-capacity.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "longmemeval-capacity-plan",
            str(source),
            str(output),
            "--tokenizer-directory",
            str(tokenizer_dir),
            "--model",
            "fixture-model",
            "--context-window",
            "17",
            "--max-output-tokens",
            "8",
        ],
    )
    assert result.exit_code == 1, result.output
    assert json.loads(output.read_text())["status"] == "blocked_context_overflow"
    answer = tmp_path / "no-answers.jsonl"
    result = runner.invoke(
        app,
        [
            "longmemeval-run",
            str(source),
            str(answer),
            "--model",
            "fixture-model",
            "--capacity-report",
            str(output),
            "--max-output-tokens",
            "8",
            "--execute",
        ],
    )
    assert result.exit_code != 0 and "exceed" in result.output
    assert not answer.exists()


@pytest.mark.parametrize("destination", ["answer", "checkpoint", "temporary"])
def test_capacity_evidence_cannot_be_overwritten_by_batch_files(
    tmp_path: Path, tokenizer_dir: Path, destination: str
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    answer = tmp_path / "answers.jsonl"
    report = {
        "answer": answer,
        "checkpoint": answer.with_name(answer.name + ".checkpoint.jsonl"),
        "temporary": answer.with_name(answer.name + ".tmp"),
    }[destination]
    prepare_capacity_report(
        source,
        report,
        model="fixture-model",
        tokenizer_directory=tokenizer_dir,
        context_window=64,
        max_output_tokens=8,
    )
    original = report.read_bytes()
    with pytest.raises(ValueError, match="overwrite its capacity report"):
        run_requests(
            source,
            answer,
            stage="generation",
            model="fixture-model",
            execute=True,
            capacity_report=report,
            max_output_tokens=8,
        )
    assert report.read_bytes() == original
