"""Explicit-model adapter for Anthropic-compatible ``/v1/messages`` endpoints.

Kept separate from the Responses adapter because the two protocols differ in endpoint,
auth header, request shape and response shape. As with the other adapters, the caller owns
all cross-turn context assembly: this sends exactly the working context it is given and
never relies on server-side conversation state.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from context_router.providers.openai_compatible import AnswerResult
from context_router.providers.pinned import require_pinned_model

AuthStyle = Literal["bearer", "x-api-key"]

ANTHROPIC_VERSION = "2023-06-01"


def _blocks_of_type(payload: dict[str, Any], kind: str, field: str) -> str:
    parts: list[str] = []
    for block in payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == kind:
            value = block.get(field)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _extract_text(payload: dict[str, Any], *, include_thinking: bool = False) -> str:
    """Join the text blocks of a messages response.

    ``include_thinking`` is the fallback for a reply with no text block at all: thinking
    blocks count against the same output budget, so a model that thinks for long enough
    emits none, and the answer -- or the verdict -- is left inside the thinking. A caller
    that parses a specific field out of the reply can use this safely; one that shows the
    text to a human cannot, because the thinking is deliberation rather than an answer.
    """

    text = _blocks_of_type(payload, "text", "text")
    if text or not include_thinking:
        return text
    return _blocks_of_type(payload, "thinking", "thinking")


def _input_tokens(usage: dict[str, Any]) -> int:
    """Count uncached, cache-written and cache-read input reported separately."""

    return sum(
        int(usage.get(field, 0) or 0)
        for field in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )


class AnthropicCompatibleVerdictModel:
    """Runs a prompt and returns the raw reply, for callers that parse their own output.

    Separate from the answer provider because a judge wants a different token budget and a
    deterministic sampler, and because the caller -- not this adapter -- owns the prompt.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        auth_style: AuthStyle = "bearer",
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        max_tokens: int = 512,
    ) -> None:
        self.model_version = require_pinned_model(model)
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.max_tokens = max_tokens
        auth = (
            {"Authorization": f"Bearer {api_key}"}
            if auth_style == "bearer"
            else {"x-api-key": api_key}
        )
        self.headers = {
            **auth,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def run(self, *, prompt: str, instructions: str) -> AnswerResult:
        body = {
            "model": self.model_version,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "system": instructions,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = self.client.post(f"{self.base_url}/v1/messages", headers=self.headers, json=body)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        input_tokens = _input_tokens(usage)
        output_tokens = int(usage.get("output_tokens", 0))
        return AnswerResult(
            # A verdict is machine-parsed, so recovering it from the thinking is safe and
            # better than losing it. The answer provider below deliberately does not do
            # this: a human should never be shown deliberation as if it were an answer.
            text=_extract_text(payload, include_thinking=True),
            model=str(payload.get("model", self.model_version)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            stop_reason=payload.get("stop_reason"),
            raw_response=payload,
        )


class AnthropicCompatibleAnswerProvider:
    """Stateless messages adapter; callers own all cross-turn context assembly."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        auth_style: AuthStyle = "bearer",
        client: httpx.Client | None = None,
        timeout: float = 120.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = require_pinned_model(model)
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout)
        self.max_tokens = max_tokens
        auth = (
            {"Authorization": f"Bearer {api_key}"}
            if auth_style == "bearer"
            else {"x-api-key": api_key}
        )
        self.headers = {
            **auth,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def answer(
        self,
        *,
        query: str,
        working_context: str,
        instructions: str,
    ) -> AnswerResult:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": instructions,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use the source-labelled working context below. "
                        "Do not assume omitted history.\n\n"
                        f"{working_context}\n\n[Current Query]\n{query}"
                    ),
                }
            ],
        }
        response = self.client.post(f"{self.base_url}/v1/messages", headers=self.headers, json=body)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        input_tokens = _input_tokens(usage)
        output_tokens = int(usage.get("output_tokens", 0))
        return AnswerResult(
            text=_extract_text(payload),
            model=str(payload.get("model", self.model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            stop_reason=payload.get("stop_reason"),
            raw_response=payload,
        )
