from __future__ import annotations

import hashlib
import math
from typing import Protocol

from context_router.retrieval import LexicalAnalyzer


class EmbeddingProvider(Protocol):
    model_version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic offline embedding adapter for tests and reproducible smoke runs."""

    model_version = "hash-embedding-v1"

    def __init__(self, dimensions: int = 256, analyzer: LexicalAnalyzer | None = None) -> None:
        self.dimensions = dimensions
        self.analyzer = analyzer or LexicalAnalyzer()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self.analyzer.tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
