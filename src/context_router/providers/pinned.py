"""The one rule every model adapter shares: the model must be named, not implied."""

from __future__ import annotations

#: Identifiers that name no model, so a run pinned to one is not reproducible.
UNPINNED = frozenset({"", "latest", "default", "auto", "current"})


def require_pinned_model(model: str) -> str:
    """Reject an unpinned model identifier.

    The plan requires every experiment configuration to fix its model name explicitly.
    A run whose model is "latest" cannot be re-run to the same numbers, and a research
    repo whose adapter silently accepts one is one careless config away from publishing
    an unreproducible result.
    """

    if model.strip().lower() in UNPINNED:
        raise ValueError("an explicit pinned model identifier is required")
    return model
