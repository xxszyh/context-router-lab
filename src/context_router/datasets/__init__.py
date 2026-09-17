"""Synthetic data generation and causal benchmark validation."""

from context_router.datasets.real_replay import (
    AnnotatedCheckpoint,
    AnnotationCoverage,
    RealReplayAnnotation,
    RealReplayValidation,
    ScrubError,
    annotation_coverage,
    assert_clean,
    export_sanitized,
    scrub,
    to_benchmark_queries,
    validate_real_replay,
)
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
    "AnnotatedCheckpoint",
    "AnnotationCoverage",
    "RealReplayAnnotation",
    "RealReplayValidation",
    "ScrubError",
    "annotation_coverage",
    "assert_clean",
    "DEFAULT_EPISODES",
    "SyntheticDataset",
    "export_sanitized",
    "generate_synthetic_dataset",
    "scrub",
    "to_benchmark_queries",
    "validate_real_replay",
    "queries_per_session",
    "validate_dataset",
]
