"""Optional joint query-passage scoring. Scores are ranks, never probabilities."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class PairScorer(Protocol):
    model_version: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderScorer:
    """An explicitly pinned, local, CPU cross-encoder; no API key is used.

    The optional dependency and weights load only when requested. MS MARCO is a
    passage-ranking training task, not a validation of conversation answer quality.
    """

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        *,
        revision: str = DEFAULT_RERANK_REVISION,
        max_length: int = 384,
        batch_size: int = 16,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("reranker revision must be an immutable 40-character commit")
        if max_length < 32 or batch_size < 1:
            raise ValueError("max_length must be >=32 and batch_size must be positive")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:  # pragma: no cover - optional extra
            raise ImportError("install the rerank extra: pip install -e '.[rerank]'") from error
        self._model: Any = CrossEncoder(
            model, revision=revision, device="cpu", max_length=max_length, trust_remote_code=False
        )
        self.batch_size = batch_size
        self.model_version = f"cross-encoder:{model}@{revision}:max_length={max_length}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = [float(value) for value in raw]
        if len(scores) != len(passages) or not all(math.isfinite(score) for score in scores):
            raise ValueError("reranker must return one finite score per passage")
        return scores
