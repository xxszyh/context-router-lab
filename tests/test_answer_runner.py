from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_longmemeval_answers import bundle as bundle
from typer.testing import CliRunner

from context_router.cli import app
from context_router.external.answer_runner import run_requests
from context_router.external.longmemeval_answers import (
    object_hash,
    prepare_judge_requests,
    score_answer_plan,
    text_hash,
)


def write_requests(path: Path, *, stage: str = "generation", count: int = 4) -> Path:
    rows = []
    for i in range(count):
        row = {"request_id": f"request-{i}", "prompt": f"Fixture question {i}"}
        if stage == "generation":
            row["instructions"] = "Use only the fixture."
            row["prompt_sha256"] = object_hash(
                {key: row[key] for key in ("instructions", "prompt")}
            )
        else:
            row["answer_sha256"] = text_hash("fixture answer")
        rows.append(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def completion(
    request: httpx.Request, text: str = "fixture answer", *, reason: str = "stop"
) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "model": body["model"],
            "choices": [{"message": {"content": text}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        },
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_dry_run_never_constructs_client_or_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_requests(tmp_path / "requests.jsonl")
    output = tmp_path / "not-created" / "answers.jsonl"

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry run constructed a network client")

    monkeypatch.setattr(httpx, "Client", forbidden)
    result = run_requests(source, output, stage="generation", model="fixture-model", max_requests=2)
    assert result["status"] == "dry_run"
    assert result["network_calls"] == 0
    assert result["maximum_calls_this_invocation"] == 2
    assert not output.parent.exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://192.168.1.1/v1",
        "http://localhost.evil/v1",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?key=secret",
        "ftp://localhost/v1",
    ],
)
def test_nonlocal_or_credential_bearing_endpoint_is_rejected(tmp_path: Path, url: str) -> None:
    source = write_requests(tmp_path / "requests.jsonl")
    with pytest.raises(ValueError, match="loopback"):
        run_requests(
            source, tmp_path / "answers", stage="generation", model="fixture-model", base_url=url
        )


def test_interruption_resumes_only_unfinished_requests_and_repairs_only_partial_tail(
    tmp_path: Path,
) -> None:
    source = write_requests(tmp_path / "requests.jsonl")
    output = tmp_path / "answers.jsonl"
    calls = []

    def serve(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["messages"][1]["content"])
        return completion(request)

    def interrupt(count: int) -> None:
        if count == 2:
            raise RuntimeError("fixture interruption")

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        with pytest.raises(RuntimeError, match="interruption"):
            run_requests(
                source,
                output,
                stage="generation",
                model="fixture-model",
                execute=True,
                client=client,
                progress=interrupt,
            )
        assert len(read_rows(output)) == 2
        checkpoint = output.with_name(output.name + ".checkpoint.jsonl")
        with checkpoint.open("ab") as stream:
            stream.write(b'{"partial":"\xe4')
        size = checkpoint.stat().st_size
        preview = run_requests(
            source, output, stage="generation", model="fixture-model", resume=True
        )
        assert preview["pending"] == 2
        assert checkpoint.stat().st_size == size
        result = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            resume=True,
            client=client,
        )
    assert result["network_calls"] == 2
    assert len(calls) == len(set(calls)) == 4
    assert len(read_rows(output)) == 4
    assert len(read_rows(checkpoint)) == 5


def test_failed_generation_stays_in_denominator_and_retry_replaces_snapshot_not_history(
    tmp_path: Path,
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=2)
    output = tmp_path / "answers.jsonl"
    calls = 0

    def serve(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "private-error-body-sentinel"})
        return completion(request)

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        first = run_requests(
            source, output, stage="generation", model="fixture-model", execute=True, client=client
        )
        assert first["failed"] == 1
        rows = read_rows(output)
        assert len(rows) == 2
        assert rows[0]["hypothesis"] == "" and rows[0]["stop_reason"] == "error"
        skipped = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            resume=True,
            client=client,
        )
        assert skipped["network_calls"] == 0
        final = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            resume=True,
            retry_failures=True,
            client=client,
        )
    assert final["failed"] == 0 and final["network_calls"] == 1
    assert len(read_rows(output)) == 2
    checkpoint = output.with_name(output.name + ".checkpoint.jsonl")
    assert len(read_rows(checkpoint)) == 4
    assert "private-error-body-sentinel" not in checkpoint.read_text()


def test_chunk_limit_and_changed_config_or_requests_are_enforced(tmp_path: Path) -> None:
    source = write_requests(tmp_path / "requests.jsonl")
    output = tmp_path / "answers.jsonl"
    with httpx.Client(transport=httpx.MockTransport(completion)) as client:
        result = run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            max_requests=1,
            client=client,
        )
    assert result["status"] == "partial" and result["remaining"] == 3
    with pytest.raises(ValueError, match="configuration"):
        run_requests(source, output, stage="generation", model="changed-model", resume=True)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="configuration"):
        run_requests(source, output, stage="generation", model="fixture-model", resume=True)


@pytest.mark.parametrize(
    "text,reason",
    [("yesterday", "stop"), ("yes and no", "stop"), ("yes.", "stop"), ("yes", "length")],
)
def test_ambiguous_or_cutoff_judgment_is_not_silently_false_or_true(
    tmp_path: Path, text: str, reason: str
) -> None:
    source = write_requests(tmp_path / "judge.jsonl", stage="judge", count=1)
    output = tmp_path / "judgments.jsonl"
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: completion(request, text, reason=reason))
    ) as client:
        result = run_requests(
            source, output, stage="judge", model="fixture-judge", execute=True, client=client
        )
    assert result["failed"] == 1
    assert read_rows(output) == []


def test_failed_or_empty_answer_uses_explicit_policy_zero_without_a_judge_call(
    tmp_path: Path,
) -> None:
    source = write_requests(tmp_path / "judge.jsonl", stage="judge", count=2)
    rows = read_rows(source)
    rows[0]["answer_completed"] = False
    rows[1]["answer_empty"] = True
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("failed generation should not require a judge request")

    output = tmp_path / "judgments.jsonl"
    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        result = run_requests(
            source, output, stage="judge", model="fixture-judge", execute=True, client=client
        )
    assert result["network_calls"] == 0 and result["policy_zero_judgments"] == 2
    assert all(
        row["correct"] is False and row["judge_call_performed"] is False
        for row in read_rows(output)
    )


def test_returned_model_mismatch_is_recorded_as_failure(tmp_path: Path) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)

    def wrong(request: httpx.Request) -> httpx.Response:
        response = completion(request)
        payload = response.json()
        payload["model"] = "wrong-model"
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(wrong)) as client:
        result = run_requests(
            source,
            tmp_path / "answers",
            stage="generation",
            model="fixture-model",
            execute=True,
            client=client,
        )
    assert result["failed"] == 1


@pytest.mark.parametrize("text,reason", [("partial answer", "length"), ("   ", "stop")])
def test_incomplete_generation_is_preserved_and_can_be_explicitly_retried(
    tmp_path: Path, text: str, reason: str
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    output = tmp_path / "answers.jsonl"
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: completion(request, text, reason=reason))
    ) as client:
        result = run_requests(
            source, output, stage="generation", model="fixture-model", execute=True, client=client
        )
    assert result["failed"] == 1
    answer = read_rows(output)[0]
    assert answer["hypothesis"] == text and answer["stop_reason"] == reason
    assert answer["output_tokens"] == 5
    retry = run_requests(
        source,
        output,
        stage="generation",
        model="fixture-model",
        resume=True,
        retry_failures=True,
    )
    assert retry["pending"] == 1 and retry["network_calls"] == 0


def test_mock_service_completes_generation_judging_and_paired_scoring(
    bundle: tuple[Path, Path, Path],
) -> None:
    directory = bundle[2]
    generation_bodies, judging_bodies = [], []

    def generate(request: httpx.Request) -> httpx.Response:
        generation_bodies.append(json.loads(request.content))
        return completion(request)

    answers = directory / "executed.answers.jsonl"
    with httpx.Client(transport=httpx.MockTransport(generate)) as client:
        run_requests(
            directory / "generation.requests.jsonl",
            answers,
            stage="generation",
            model="fixture-generator",
            execute=True,
            client=client,
        )
    assert "REFERENCE-ONLY-SENTINEL" not in json.dumps(generation_bodies)
    requests = directory / "judge.requests.jsonl"
    prepare_judge_requests(directory / "plan.private.json", answers, requests)

    def judge(request: httpx.Request) -> httpx.Response:
        judging_bodies.append(json.loads(request.content))
        return completion(request, " YES ")

    judgments = directory / "executed.judgments.jsonl"
    with httpx.Client(transport=httpx.MockTransport(judge)) as client:
        run_requests(
            requests, judgments, stage="judge", model="fixture-judge", execute=True, client=client
        )
    assert "REFERENCE-ONLY-SENTINEL" in json.dumps(judging_bodies)
    report = score_answer_plan(directory / "plan.private.json", answers, judgments)
    assert report["status"] == "complete"
    assert report["summary"]["joint"]["reported_usage_rows"] == 2
    assert report["summary"]["joint"]["total_reported_tokens"] == 50
    assert report["summary"]["joint"]["reported_latency_rows"] == 2


def test_cli_preview_is_default_and_has_no_output_file(tmp_path: Path) -> None:
    source = write_requests(tmp_path / "requests.jsonl")
    output = tmp_path / "answers.jsonl"
    result = CliRunner().invoke(
        app, ["longmemeval-run", str(source), str(output), "--model", "fixture-model"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["network_calls"] == 0
    assert not output.exists()


def test_completed_checkpoint_corruption_is_not_repaired_or_overwritten(tmp_path: Path) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    output = tmp_path / "answers.jsonl"
    with httpx.Client(transport=httpx.MockTransport(completion)) as client:
        run_requests(
            source, output, stage="generation", model="fixture-model", execute=True, client=client
        )
    checkpoint = output.with_name(output.name + ".checkpoint.jsonl")
    events = read_rows(checkpoint)
    events[1]["row"]["hypothesis"] = "changed after completion"
    checkpoint.write_text("".join(json.dumps(event) + "\n" for event in events))
    corrupted = checkpoint.read_bytes()
    snapshot = output.read_bytes()
    with pytest.raises(ValueError, match="integrity"):
        run_requests(source, output, stage="generation", model="fixture-model", resume=True)
    assert checkpoint.read_bytes() == corrupted
    assert output.read_bytes() == snapshot


@pytest.mark.parametrize("legacy", [False, True])
def test_token_limit_field_is_explicit_and_missing_usage_stays_unknown(
    tmp_path: Path, legacy: bool
) -> None:
    source = write_requests(tmp_path / "requests.jsonl", count=1)
    output = tmp_path / "answers.jsonl"

    def serve(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        field = "max_tokens" if legacy else "max_completion_tokens"
        assert body[field] == 128
        assert ("max_completion_tokens" if legacy else "max_tokens") not in body
        response = completion(request)
        payload = response.json()
        payload.pop("usage")
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        run_requests(
            source,
            output,
            stage="generation",
            model="fixture-model",
            execute=True,
            max_output_tokens=128,
            token_limit_field="max_tokens" if legacy else "max_completion_tokens",
            client=client,
        )
    assert read_rows(output)[0]["input_tokens"] is None
    assert read_rows(output)[0]["output_tokens"] is None
