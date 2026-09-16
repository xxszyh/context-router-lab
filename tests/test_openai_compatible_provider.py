from __future__ import annotations

import json

import httpx

from context_router.providers.openai_compatible import OpenAICompatibleAnswerProvider


def test_answer_provider_sends_only_the_manually_assembled_context() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "answer"}],
                    }
                ],
                "usage": {"input_tokens": 17, "output_tokens": 2, "total_tokens": 19},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleAnswerProvider(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="pinned-model-2026-01-01",
        client=client,
    )

    result = provider.answer(
        query="Why?",
        working_context="[Context: ctx-a] evidence",
        instructions="Use only supplied evidence.",
    )

    assert result.text == "answer"
    assert captured["model"] == "pinned-model-2026-01-01"
    assert captured["store"] is False
    assert "conversation" not in captured
    assert "previous_response_id" not in captured
    assert "[Context: ctx-a] evidence" in str(captured["input"])
