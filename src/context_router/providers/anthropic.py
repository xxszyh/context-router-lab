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


def _extract_text(payload: dict[str, Any]) -> str:
    """Join the text blocks of a messages response, ignoring thinking blocks."""

    parts: list[str] = []
    for block in payload.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


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
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        return AnswerResult(
            text=_extract_text(payload),
            model=str(payload.get("model", self.model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            raw_response=payload,
        )
