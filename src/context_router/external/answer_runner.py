"""Bounded local batch execution with a durable per-request journal.

Dry runs never construct a client or write files. Execution is opt-in, accepts
loopback endpoints only, and makes exactly one HTTP attempt per scheduled item.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from context_router.external.batch_requests import Stage as Stage
from context_router.external.batch_requests import policy_zero, read_requests, request_messages
from context_router.external.context_capacity import validate_capacity_report
from context_router.external.longmemeval_answers import object_hash
from context_router.providers.pinned import require_pinned_model

TokenLimitField = Literal["max_completion_tokens", "max_tokens"]


class ResponseError(ValueError):
    """A provider replied, but not with the declared model/response contract."""


class CapacityMismatchError(ResponseError):
    """Observed token usage contradicts the preflight; stop the remaining batch."""


def local_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.hostname not in ("localhost", "127.0.0.1", "::1")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("batch execution requires a loopback URL without credentials or query")
    # Accessing port also validates its range and representation.
    _ = parsed.port
    return value.rstrip("/")


def _response(response: httpx.Response, model: str) -> tuple[str, str, int | None, int | None]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("model") != model:
        raise ResponseError("returned model does not match the declared model")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ResponseError("expected one completion choice")
    message = choices[0].get("message")
    reason = choices[0].get("finish_reason")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ResponseError("expected a text answer")
    if not isinstance(reason, str) or not reason:
        raise ResponseError("finish_reason is required")
    try:
        message["content"].encode("utf-8")
    except UnicodeError as error:
        raise ResponseError("response contains invalid Unicode") from error
    usage = payload.get("usage")
    if usage is None:
        return message["content"], reason, None, None
    if not isinstance(usage, dict):
        raise ResponseError("invalid usage object")
    counts = [usage.get(key) for key in ("prompt_tokens", "completion_tokens")]
    if any(value is not None and (type(value) is not int or value < 0) for value in counts):
        raise ResponseError("invalid token usage")
    return message["content"], reason, counts[0], counts[1]


def _read_journal(
    path: Path, configuration: dict[str, Any], expected: set[str]
) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open("rb") as stream:
        if json.loads(stream.readline()) != {"configuration": configuration}:
            raise ValueError("checkpoint configuration or request file changed")
        complete_end = stream.tell()
        for line in stream:
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("invalid journal event")
            digest = event.pop("event_sha256", None)
            key = event.get("request_id")
            if not isinstance(key, str) or object_hash(event) != digest or key not in expected:
                raise ValueError("journal integrity check failed")
            if (
                event.get("status") not in ("success", "failed")
                or type(event.get("attempt")) is not int
                or event["attempt"] != latest.get(key, {}).get("attempt", 0) + 1
            ):
                raise ValueError("invalid journal attempt sequence")
            latest[key] = event
            complete_end += len(line)
    return latest, complete_end


def _snapshot(
    output: Path, requests: list[dict[str, Any]], latest: dict[str, dict[str, Any]]
) -> None:
    rows = [
        latest[row["request_id"]]["row"]
        for row in requests
        if row["request_id"] in latest and latest[row["request_id"]]["row"] is not None
    ]
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(output)


def run_requests(
    source: Path,
    output: Path,
    *,
    stage: Stage,
    model: str,
    base_url: str = "http://127.0.0.1:11434/v1",
    execute: bool = False,
    resume: bool = False,
    retry_failures: bool = False,
    max_requests: int = 20,
    max_output_tokens: int = 256,
    temperature: float = 0.0,
    token_limit_field: TokenLimitField = "max_completion_tokens",
    timeout: float = 60.0,
    client: httpx.Client | None = None,
    progress: Callable[[int], None] | None = None,
    capacity_report: Path | None = None,
) -> dict[str, Any]:
    if stage not in ("generation", "judge") or token_limit_field not in (
        "max_tokens",
        "max_completion_tokens",
    ):
        raise ValueError("unsupported stage or token-limit field")
    if max_requests < 1 or max_output_tokens < 1 or not 0 <= temperature <= 2:
        raise ValueError("request/output limits must be positive and temperature within 0..2")
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be within 0..60 seconds")
    if retry_failures and not resume:
        raise ValueError("retry-failures requires resume")
    base_url = local_base_url(base_url)
    model = require_pinned_model(model)
    requests, digest = read_requests(source, stage)
    checkpoint = output.with_name(output.name + ".checkpoint.jsonl")
    if source.resolve() in {
        output.resolve(),
        checkpoint.resolve(),
        output.with_name(output.name + ".tmp").resolve(),
    }:
        raise ValueError("batch output must not overwrite its request source")
    capacity_digest = None
    token_counts: dict[str, int] = {}
    if capacity_report is not None:
        if capacity_report.resolve() in {
            output.resolve(),
            checkpoint.resolve(),
            output.with_name(output.name + ".tmp").resolve(),
        }:
            raise ValueError("batch output must not overwrite its capacity report")
        capacity_digest, token_counts = validate_capacity_report(
            capacity_report,
            requests,
            requests_sha256=digest,
            stage=stage,
            model=model,
            max_output_tokens=max_output_tokens,
        )
    settings = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "token_limit_field": token_limit_field,
        "n": 1,
        "stream": False,
        "protocol": "chat_completions",
    }
    if capacity_digest is not None:
        settings["capacity_report_sha256"] = capacity_digest
    configuration = {
        "implementation": "local-answer-batch-v1",
        "requests_sha256": digest,
        "stage": stage,
        "model": model,
        "base_url": base_url,
        "generation_config": settings,
        "timeout": timeout,
    }
    latest: dict[str, dict[str, Any]] = {}
    complete_end = 0
    if resume:
        if not checkpoint.exists():
            raise ValueError("resume checkpoint does not exist")
        latest, complete_end = _read_journal(
            checkpoint, configuration, {r["request_id"] for r in requests}
        )
    elif execute and (checkpoint.exists() or output.exists()):
        raise ValueError("batch output exists; resume or choose a new path")
    pending = [
        row
        for row in requests
        if row["request_id"] not in latest
        or (retry_failures and latest[row["request_id"]]["status"] == "failed")
    ]
    if not execute:
        return {
            "status": "dry_run",
            "stage": stage,
            "requests": len(requests),
            "pending": len(pending),
            "network_calls": 0,
            "maximum_calls_this_invocation": min(max_requests, len(pending)),
            "maximum_output_tokens_this_invocation": min(max_requests, len(pending))
            * max_output_tokens,
            "configuration": configuration,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume and complete_end != checkpoint.stat().st_size:
        with checkpoint.open("r+b") as repair:
            repair.truncate(complete_end)
    calls = 0
    policy_zeros = 0
    owned_client = client is None
    active_client = client or httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False)
    try:
        with checkpoint.open("a" if resume else "x", encoding="utf-8", newline="\n") as journal:
            if not resume:
                journal.write(json.dumps({"configuration": configuration}) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
            try:
                for request in pending:
                    key = request["request_id"]
                    zero = policy_zero(request, stage)
                    if not zero and calls >= max_requests:
                        break
                    started = time.perf_counter()
                    record: dict[str, Any] | None = None
                    failure: dict[str, Any] | None = None
                    try:
                        if zero:
                            policy_zeros += 1
                            record = {
                                "request_id": key,
                                "answer_sha256": request["answer_sha256"],
                                "correct": False,
                                "judge_model": model,
                                "judgment_origin": "failed_generation_policy",
                                "judge_call_performed": False,
                            }
                        else:
                            calls += 1
                            response = active_client.post(
                                base_url + "/chat/completions",
                                json={
                                    "model": model,
                                    "messages": request_messages(request),
                                    "temperature": temperature,
                                    token_limit_field: max_output_tokens,
                                    "n": 1,
                                    "stream": False,
                                },
                            )
                            text, reason, input_tokens, output_tokens = _response(response, model)
                            if capacity_digest is not None and input_tokens != token_counts[key]:
                                raise CapacityMismatchError(
                                    "service input-token usage does not match capacity preflight"
                                )
                            if stage == "generation":
                                record = {
                                    "request_id": key,
                                    "prompt_sha256": request["prompt_sha256"],
                                    "hypothesis": text,
                                    "model": model,
                                    "generation_config": settings,
                                    "stop_reason": reason,
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                }
                                if reason not in ("stop", "end_turn", "completed"):
                                    failure = {"kind": "IncompleteGeneration"}
                                elif not text.strip():
                                    failure = {"kind": "EmptyGeneration"}
                            else:
                                verdict = text.strip().lower()
                                if reason != "stop" or verdict not in ("yes", "no"):
                                    raise ResponseError("judge must finish with exactly yes or no")
                                record = {
                                    "request_id": key,
                                    "answer_sha256": request["answer_sha256"],
                                    "correct": verdict == "yes",
                                    "judge_model": model,
                                    "judgment_origin": "model",
                                    "judge_call_performed": True,
                                }
                    except (httpx.HTTPError, ValueError) as error:
                        # Never persist URL/header/body strings from provider exceptions.
                        failure = {"kind": type(error).__name__}
                        if isinstance(error, httpx.HTTPStatusError):
                            failure["http_status"] = error.response.status_code
                        if stage == "generation":
                            record = {
                                "request_id": key,
                                "prompt_sha256": request["prompt_sha256"],
                                "hypothesis": "",
                                "model": model,
                                "generation_config": settings,
                                "stop_reason": "error",
                                "input_tokens": None,
                                "output_tokens": None,
                            }
                    latency = time.perf_counter() - started
                    if record is not None:
                        record["latency_seconds"] = latency
                        if stage == "judge":
                            record["judgment_config"] = settings
                    event = {
                        "request_id": key,
                        "attempt": latest.get(key, {}).get("attempt", 0) + 1,
                        "status": "failed" if failure else "success",
                        "failure": failure,
                        "row": record,
                        "latency_seconds": latency,
                    }
                    journal.write(
                        json.dumps(
                            {**event, "event_sha256": object_hash(event)}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    journal.flush()
                    os.fsync(journal.fileno())
                    latest[key] = event
                    if progress is not None:
                        progress(len(latest))
                    if failure is not None and failure["kind"] == "CapacityMismatchError":
                        break
            finally:
                _snapshot(output, requests, latest)
    finally:
        if owned_client:
            active_client.close()
    failed = sum(event["status"] == "failed" for event in latest.values())
    remaining = len(requests) - len(latest)
    return {
        "status": "partial" if remaining else ("failed" if failed else "complete"),
        "stage": stage,
        "requests": len(requests),
        "recorded": len(latest),
        "remaining": remaining,
        "failed": failed,
        "network_calls": calls,
        "policy_zero_judgments": policy_zeros,
        "output": str(output),
        "checkpoint": str(checkpoint),
    }
