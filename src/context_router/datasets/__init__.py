"""Synthetic data generation and causal benchmark validation."""

from context_router.datasets.synthetic import SyntheticDataset, generate_synthetic_dataset
from context_router.datasets.validation import validate_dataset

__all__ = ["SyntheticDataset", "generate_synthetic_dataset", "validate_dataset"]
