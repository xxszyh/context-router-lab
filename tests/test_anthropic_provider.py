from __future__ import annotations

import json

import httpx
import pytest

from context_router.providers.anthropic import AnthropicCompatibleAnswerProvider, AuthStyle


def make_provider(
    requests: list[httpx.Request], *, auth_style: AuthStyle = "bearer"
) -> AnthropicCompatibleAnswerProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "some-pinned-model",
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "回退到上一个可用版本。"},
                ],
                "usage": {"input_tokens": 321, "output_tokens": 12},
            },
        )

    return AnthropicCompatibleAnswerProvider(
        base_url="https://example.invalid",
        api_key="test-key",
        model="some-pinned-model",
        auth_style=auth_style,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_provider_sends_only_the_manually_assembled_context() -> None:
    requests: list[httpx.Request] = []
    provider = make_provider(requests)

    result = provider.answer(
        query="回到 SQLite migration，锁升级怎么定？",
        working_context="[Context: c1 | SQLite migration]\nGoal: 修迁移",
        instructions="Answer from the context only.",
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path.endswith("/v1/messages")
    body = json.loads(request.content)
    assert body["model"] == "some-pinned-model"
    assert body["system"] == "Answer from the context only."
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert "回到 SQLite migration" in body["messages"][0]["content"]
    # No server-side conversation state may leak in: this adapter is stateless.
    assert "previous_response_id" not in body
    assert "conversation" not in body
    assert result.text == "回退到上一个可用版本。"
    assert result.input_tokens == 321
    assert result.output_tokens == 12
    assert result.total_tokens == 333


def test_provider_uses_the_anthropic_version_header() -> None:
    requests: list[httpx.Request] = []
    make_provider(requests).answer(query="q", working_context="c", instructions="i")

    assert requests[0].headers["anthropic-version"] == "2023-06-01"


def test_provider_counts_every_prompt_cache_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "some-pinned-model",
                "content": [{"type": "text", "text": "done"}],
                "usage": {
                    "input_tokens": 59,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 71000,
                    "output_tokens": 12,
                },
            },
        )

    provider = AnthropicCompatibleAnswerProvider(
        base_url="https://example.invalid",
        api_key="test-key",
        model="some-pinned-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = provider.answer(query="q", working_context="memory", instructions="rules")

    assert result.input_tokens == 71159
    assert result.total_tokens == 71171


def test_provider_sends_the_token_as_a_bearer_authorization() -> None:
    requests: list[httpx.Request] = []
    make_provider(requests, auth_style="bearer").answer(
        query="q", working_context="c", instructions="i"
    )

    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert "x-api-key" not in requests[0].headers


def test_provider_can_send_the_token_as_an_api_key_header() -> None:
    requests: list[httpx.Request] = []
    make_provider(requests, auth_style="x-api-key").answer(
        query="q", working_context="c", instructions="i"
    )

    assert requests[0].headers["x-api-key"] == "test-key"
    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize("model", ["", "latest", "default", "AUTO"])
def test_provider_refuses_an_unpinned_model(model: str) -> None:
    """A run whose model is not fixed cannot be reproduced, so it must not start."""

    with pytest.raises(ValueError, match="explicit pinned model"):
        AnthropicCompatibleAnswerProvider(
            base_url="https://example.invalid", api_key="k", model=model
        )
