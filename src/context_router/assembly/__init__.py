"""Token-budgeted, provenance-preserving context assembly."""

from context_router.assembly.builder import (
    ContextBuilder,
    assemble_context,
    assemble_indexed_context,
)

__all__ = ["ContextBuilder", "assemble_context", "assemble_indexed_context"]
