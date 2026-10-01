"""A long run must survive a hiccup, and must not survive a refusal.

The two halves are equally important. Retrying a read timeout is what stops a 51-minute
264-call judge run from being discarded at call 200 -- which is what happened on 2026-10-01.
Retrying a 400 would delete a finding: `full_history` failing to run against a documented
model's ceiling is counted in HTTP 400s, and a retry loop would turn those into timeouts and
report the arm as merely slow.
"""

from __future__ import annotations

import httpx
import pytest

from context_router.providers.anthropic import AnthropicCompatibleVerdictModel
from context_router.providers.transport import post_with_retry

URL = "https://gateway.test/v1/messages"


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_a_transport_failure_is_retried_and_the_later_response_returned() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"ok": True})

    response = post_with_retry(_client(handler), URL, headers={}, body={}, sleep=lambda _: None)

    assert response.status_code == 200
    assert len(calls) == 2


def test_a_status_error_is_answered_once_and_not_retried() -> None:
    """The server replying is the end of the exchange, whatever it replied."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "context length exceeded"}})

    with pytest.raises(httpx.HTTPStatusError):
        post_with_retry(_client(handler), URL, headers={}, body={}, sleep=lambda _: None)

    assert len(calls) == 1, "a 400 is a result, not a hiccup"


def test_exhausting_the_attempts_raises_the_last_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(httpx.ReadTimeout):
        post_with_retry(
            _client(handler), URL, headers={}, body={}, attempts=3, sleep=lambda _: None
        )


def test_the_backoff_doubles_and_is_not_paid_after_the_last_attempt() -> None:
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(httpx.ConnectError):
        post_with_retry(
            _client(handler),
            URL,
            headers={},
            body={},
            attempts=4,
            backoff=2.0,
            sleep=waits.append,
        )

    assert waits == [2.0, 4.0, 8.0], "sleeping after the final failure just delays the error"


def test_a_zero_attempt_call_is_rejected_rather_than_returning_nothing() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        post_with_retry(
            _client(lambda request: httpx.Response(200)),
            URL,
            headers={},
            body={},
            attempts=0,
        )


def test_the_verdict_model_itself_survives_a_transient_failure() -> None:
    """The helper is only worth having if the thing that failed is wired to it."""

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(
            200,
            json={
                "model": "kimi-k2.5",
                "content": [{"type": "text", "text": "WINNER: a"}],
                "usage": {"input_tokens": 11, "output_tokens": 3},
                "stop_reason": "end_turn",
            },
        )

    model = AnthropicCompatibleVerdictModel(
        base_url="https://gateway.test",
        api_key="not-a-real-key",
        model="kimi-k2.5",
        client=_client(handler),
    )
    result = model.run(prompt="p", instructions="i")

    assert result.text == "WINNER: a"
    assert len(calls) == 2
