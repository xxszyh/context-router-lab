"""Synthetic data generation and causal benchmark validation."""

from context_router.datasets.synthetic import (
    DEFAULT_CONTEXTS,
    DEFAULT_EPISODES,
    SyntheticDataset,
    generate_synthetic_dataset,
    queries_per_session,
)
from context_router.datasets.validation import validate_dataset

__all__ = [
    "DEFAULT_CONTEXTS",
    "DEFAULT_EPISODES",
    "SyntheticDataset",
    "generate_synthetic_dataset",
    "queries_per_session",
    "validate_dataset",
]
