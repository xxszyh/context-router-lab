"""Explicit-model adapters for OpenAI-compatible endpoints."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import Field

from context_router.domain import Contract
from context_router.providers.pinned import require_pinned_model


class AnswerResult(Contract):
    text: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    #: Provider stop reason, so a length-truncated answer is visible as such.
    stop_reason: str | None = None
    raw_response: dict[str, Any]


def _extract_output_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    parts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if content.get("type") in {"output_text", "text"} and isinstance(text, str):
                parts.append(text)
    return "".join(parts)


class OpenAICompatibleAnswerProvider:
    """Stateless Responses adapter; callers own all cross-turn context assembly."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = require_pinned_model(model)
        self.client = client or httpx.Client(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def answer(
        self,
        *,
        query: str,
        working_context: str,
        instructions: str,
    ) -> AnswerResult:
        body = {
            "model": self.model,
            "store": False,
            "instructions": instructions,
            "input": [
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
        response = self.client.post(f"{self.base_url}/responses", headers=self.headers, json=body)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return AnswerResult(
            text=_extract_output_text(payload),
            model=self.model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            stop_reason=payload.get("stop_reason") or payload.get("status"),
            raw_response=payload,
        )


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not model or model in {"latest", "default"}:
            raise ValueError("an explicit pinned embedding model identifier is required")
        self.base_url = base_url.rstrip("/")
        self.model_version = model
        self.client = client or httpx.Client(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": self.model_version, "input": texts},
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [[float(value) for value in item["embedding"]] for item in data]


class OpenAICompatibleRelationClassifier:
    """Optional structured-output relation adapter for ambiguous queries."""

    labels = ["continue", "switch_or_return", "cross_context", "new_context", "unknown"]

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not model or model in {"latest", "default"}:
            raise ValueError("an explicit pinned relation model identifier is required")
        self.base_url = base_url.rstrip("/")
        self.model_version = model
        self.client = client or httpx.Client(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def classify_text(self, prompt: str) -> tuple[str, dict[str, float]]:
        schema = {
            "type": "object",
            "properties": {
                "relation": {"type": "string", "enum": self.labels},
                "probabilities": {
                    "type": "object",
                    "properties": {
                        label: {"type": "number", "minimum": 0, "maximum": 1}
                        for label in self.labels
                    },
                    "required": self.labels,
                    "additionalProperties": False,
                },
            },
            "required": ["relation", "probabilities"],
            "additionalProperties": False,
        }
        body = {
            "model": self.model_version,
            "store": False,
            "instructions": (
                "Classify conversational context relation only; do not answer the query."
            ),
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "context_relation",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        response = self.client.post(f"{self.base_url}/responses", headers=self.headers, json=body)
        response.raise_for_status()
        parsed = json.loads(_extract_output_text(response.json()))
        probabilities = {key: float(value) for key, value in parsed["probabilities"].items()}
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("relation probabilities must have positive mass")
        probabilities = {key: value / total for key, value in probabilities.items()}
        return str(parsed["relation"]), probabilities
