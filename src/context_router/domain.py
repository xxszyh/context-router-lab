"""Versioned domain contracts shared by routing, storage, assembly and evaluation."""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Actor = Literal["user", "assistant", "tool", "system"]
EventKind = Literal["message", "tool_call", "tool_result", "file_change", "decision", "artifact"]
Relation = Literal["continue", "switch_or_return", "cross_context", "new_context", "unknown"]
Decision = Literal["route", "abstain", "new_context_candidate"]


class Contract(BaseModel):
    """Base for immutable, JSON-stable public contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def new_uuid7() -> str:
    """Generate an RFC 9562-compatible UUIDv7 on Python versions without uuid.uuid7."""

    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    random_bits = uuid.uuid4().int & ((1 << 74) - 1)
    value = timestamp_ms << 80
    value |= 0x7 << 76
    value |= ((random_bits >> 62) & 0xFFF) << 64
    value |= 0b10 << 62
    value |= random_bits & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


class RawEvent(Contract):
    schema_version: Literal["1.0"] = "1.0"
    event_id: str
    session_id: str
    sequence: int = Field(ge=0)
    occurred_at: datetime
    ingested_at: datetime
    actor: Actor
    kind: EventKind
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None
    content_sha256: str

    @field_validator("occurred_at", "ingested_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_content_hash(self) -> RawEvent:
        expected = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match content")
        return self

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        sequence: int,
        occurred_at: datetime,
        actor: Actor,
        kind: EventKind,
        content: str,
        event_id: str | None = None,
        ingested_at: datetime | None = None,
        payload: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
    ) -> RawEvent:
        return cls(
            event_id=event_id or new_uuid7(),
            session_id=session_id,
            sequence=sequence,
            occurred_at=occurred_at,
            ingested_at=ingested_at or datetime.now(UTC),
            actor=actor,
            kind=kind,
            content=content,
            payload=payload or {},
            parent_event_id=parent_event_id,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )


class EventContextAssignment(Contract):
    event_id: str
    context_id: str
    source: Literal["oracle", "human", "router"]
    relevance: float = Field(ge=0.0, le=1.0)
    annotation_version: int = Field(ge=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("assignment timestamps must be timezone-aware")
        return value


class FlatContext(Contract):
    context_id: str
    name: str
    goal: str
    summary: str
    entities: list[str] = Field(default_factory=list)
    lexical_terms: list[str] = Field(default_factory=list)
    status: Literal["active", "paused", "archived"]
    created_at_event: str
    last_active_sequence: int = Field(ge=0)
    version: int = Field(ge=1)

    def searchable_text(self) -> str:
        return "\n".join(
            [
                self.name,
                self.goal,
                self.summary,
                " ".join(self.entities),
                " ".join(self.lexical_terms),
            ]
        )


class RouteRequest(Contract):
    query_event_id: str
    query: str
    recent_events: list[RawEvent] = Field(default_factory=list)
    primary_context_id: str | None = None
    recent_context_ids: list[str] = Field(default_factory=list)
    context_catalog: list[FlatContext]
    as_of_sequence: int = Field(ge=0)
    max_selected_contexts: int = Field(default=3, ge=1, le=20)
    routing_profile: str = "default"

    @model_validator(mode="after")
    def reject_future_recent_events(self) -> RouteRequest:
        if any(event.sequence > self.as_of_sequence for event in self.recent_events):
            raise ValueError("recent_events contains a future event")
        return self


class ContextCandidate(Contract):
    context_id: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    entity_rank: int | None = None
    rrf_score: float = Field(ge=0.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    feature_values: dict[str, float]
    reasons: list[str]


class RouteDecision(Contract):
    decision: Decision
    relation: Relation
    relation_probabilities: dict[str, float]
    candidates: list[ContextCandidate]
    selected_context_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_level: int = Field(ge=0, le=3)
    trace_id: str
    index_version: str
    model_versions: dict[str, str]


class AssemblyRequest(Contract):
    query: str
    recent_events: list[RawEvent]
    event_pool: list[RawEvent]
    context_catalog: list[FlatContext]
    assignments: list[EventContextAssignment]
    as_of_sequence: int = Field(ge=0)
    token_budget: int = Field(default=2048, ge=32)
    system_rules: str = ""


class SourceSpan(Contract):
    event_id: str | None
    context_id: str | None
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    section: Literal["recent", "context", "evidence"]


class WorkingContext(Contract):
    rendered_text: str
    selected_context_ids: list[str]
    included_event_ids: list[str]
    source_spans: list[SourceSpan]
    memory_tokens: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    dropped_candidates: list[dict[str, Any]]
    causal_cutoff: int
    routing_trace_id: str


class BenchmarkQuery(Contract):
    sample_id: str
    session_id: str
    query_event_id: str
    as_of_sequence: int
    language: Literal["zh", "en", "mixed"]
    query_type: Literal[
        "continue",
        "return",
        "switch",
        "cross_context",
        "short_coreference",
        "new_context",
        "unanswerable",
    ]
    required_context_ids: list[str]
    acceptable_evidence_sets: list[list[str]]
    forbidden_future_event_ids: list[str]
    relation_label: Relation
    answer_requirements: list[str]
    must_abstain: bool
    difficulty: Literal["easy", "medium", "hard"]
    primary_context_id: str | None = None
    recent_context_ids: list[str] = Field(default_factory=list)
