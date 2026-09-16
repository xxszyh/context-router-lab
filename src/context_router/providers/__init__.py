"""Replaceable provider adapters used at real seams."""

from context_router.providers.anthropic import AnthropicCompatibleAnswerProvider
from context_router.providers.embedding import EmbeddingProvider, HashEmbeddingProvider
from context_router.providers.openai_compatible import (
    AnswerResult,
    OpenAICompatibleAnswerProvider,
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleRelationClassifier,
)

__all__ = [
    "AnthropicCompatibleAnswerProvider",
    "AnswerResult",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "OpenAICompatibleAnswerProvider",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleRelationClassifier",
]
