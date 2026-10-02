"""Adapters to external benchmarks this project measures itself against.

Kept apart from `datasets/` on purpose. `datasets/` holds benchmarks this project owns and
labels; a module here consumes somebody else's, and its job is to say honestly what the external
benchmark does and does not decide -- see `scale_qa` for the arrangement problem.
"""

from context_router.external.scale_qa import (
    Arrangement,
    ArrangementReachability,
    RetrievalMode,
    ScaleQaBlock,
    ScaleQaPackage,
    ScaleQaQuestion,
    ScaleQaWindow,
    WindowRetriever,
    arrange,
    arrangement_reachability,
    default_embedder,
    evaluate,
    evidence_recall,
    load_package,
    retrieve,
    window_blocks,
)

__all__ = [
    "Arrangement",
    "ArrangementReachability",
    "RetrievalMode",
    "ScaleQaBlock",
    "ScaleQaPackage",
    "ScaleQaQuestion",
    "ScaleQaWindow",
    "WindowRetriever",
    "arrange",
    "arrangement_reachability",
    "default_embedder",
    "evaluate",
    "evidence_recall",
    "load_package",
    "retrieve",
    "window_blocks",
]
