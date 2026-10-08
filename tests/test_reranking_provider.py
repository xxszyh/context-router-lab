from __future__ import annotations

import pytest

from context_router.providers.reranking import CrossEncoderScorer


def test_joint_model_rejects_a_mutable_revision_before_loading_dependencies() -> None:
    with pytest.raises(ValueError, match="immutable"):
        CrossEncoderScorer(revision="main")


def test_joint_model_rejects_invalid_limits_before_loading_dependencies() -> None:
    with pytest.raises(ValueError, match="max_length"):
        CrossEncoderScorer(max_length=0)
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderScorer(batch_size=0)
