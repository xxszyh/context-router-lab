"""Real-replay annotation format: label real conversations, publish only a scrubbed subset.

The synthetic benchmark is templated, so it cannot show whether routing survives paraphrases,
ambiguous identifiers or conclusions that a later turn overturns. Real replay supplies those,
at the cost of two hazards this module exists to contain.

The first is privacy. The imported Claude histories contain the user's phone number, their
real email address and the absolute paths of their machine, so nothing derived from them may
be committed unscrubbed. Redaction is applied on the way out and a leftover match is a **hard
failure**, because a partially scrubbed conversation is worse than none: it looks safe.

The second is provenance. An annotation is a claim about specific events in a specific
conversation at a specific point in time, so the format records the source session, the
annotator, and whether the checkpoint was independently re-annotated and adjudicated. A label
nobody signed for is an opinion, and the disagreement rate between annotators is the only
evidence about how much the labels can be trusted.

Only the events an annotation actually references are exported, so the published derivative is
a subset of the conversation rather than the conversation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from context_router.domain import (
    BenchmarkQuery,
    Contract,
    EventContextAssignment,
    FlatContext,
    QueryType,
    RawEvent,
    Relation,
)
from context_router.storage import SQLiteEventStore

SCHEMA_VERSION = "1.0"

#: Redactions applied to every exported string. The goal is to remove what identifies the
#: person or the machine without removing what identifies the task, because file and symbol
#: names are the signal the routing benchmark is built on. An absolute path therefore keeps
#: its tail and loses its prefix.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "<private-key>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "<secret>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<secret>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<secret>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "<phone>"),
    (re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+"), "<home>"),
    (re.compile(r"/(?:home|Users)/[^/\s\"']+"), "<home>"),
    # A drive letter, not any letter followed by a colon: without the lookbehind this also
    # rewrites the `s:/` inside `https://`, and the export gate then fails on every URL.
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"), "<abs>/"),
)

#: What must not survive scrubbing. A hit refuses the export rather than writing a file that
#: leaks, so the failure is loud and happens before anything reaches disk.
_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("secret", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    # Anchored the same way, so `https://` is not mistaken for a leaked Windows path. The
    # first version of this pattern flagged 172 of 7 705 real events, none of them actually
    # a path, which is how the lookbehind got here.
    ("drive-path", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")),
)


class ScrubError(RuntimeError):
    """Raised when a redaction left something that must not be published."""


def scrub(text: str) -> str:
    """Redact personal and machine-specific content from one string."""

    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _mask(value: str) -> str:
    """Describe a match without reproducing it. A gate that logs the leak is part of the leak."""

    return f"{value[:2]}…({len(value)} chars)"


def assert_clean(text: str, *, where: str) -> None:
    """Fail loudly if anything publishable is still present.

    The offending value is masked rather than echoed: this runs in CI and in transcripts, and
    printing the phone number it just found would put it somewhere new.

    Callers must pass the fields that were actually redacted, not a document that also carries
    importer-generated identifiers. A UUID's hex groups are runs of digits, so a phone pattern
    matches them, and checking identifiers turns a real guard into a false alarm -- which is
    exactly what happened on the first export.
    """

    for label, pattern in _FORBIDDEN:
        match = pattern.search(text)
        if match:
            raise ScrubError(f"{label} survived scrubbing in {where}: {_mask(match.group(0))}")


class AnnotatedCheckpoint(Contract):
    """One query position in a real conversation, with its labels.

    Mirrors `BenchmarkQuery` but keyed to real event ids and carrying who labelled it.
    """

    sample_id: str
    query_event_id: str
    as_of_sequence: int = Field(ge=0)
    query_type: QueryType
    relation_label: Relation
    required_context_ids: list[str] = Field(default_factory=list)
    acceptable_evidence_sets: list[list[str]] = Field(default_factory=list)
    must_abstain: bool = False
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    language: Literal["zh", "en", "mixed"] = "mixed"
    annotator: str
    notes: str = ""
    #: Set only when a *different* person labelled this checkpoint independently. A second
    #: pass by the same annotator is a recheck and goes in the fields below, because one
    #: person twice cannot measure how much a label depends on who is labelling.
    second_annotator: str | None = None
    #: Set when the two disagreed and someone resolved it.
    adjudicated_by: str | None = None
    #: A same-annotator re-derivation: what it used, and whether it landed on the same label.
    recheck_annotator: str | None = None
    recheck_method: str = ""
    recheck_contexts_agree: bool | None = None
    recheck_type_agrees: bool | None = None


class RealReplayAnnotation(Contract):
    """A whole annotation file: the contexts it declares and the checkpoints it labels.

    `context_members` maps each declared context to the real events that belong to it. It is
    separate from `FlatContext` because membership is an annotation, not part of the domain
    contract the router consumes.
    """

    schema_version: str = SCHEMA_VERSION
    source_session_id: str
    source_note: str = ""
    contexts: list[FlatContext]
    context_members: dict[str, list[str]] = Field(default_factory=dict)
    checkpoints: list[AnnotatedCheckpoint]

    def referenced_event_ids(self) -> set[str]:
        referenced: set[str] = set()
        for checkpoint in self.checkpoints:
            referenced.add(checkpoint.query_event_id)
            for option in checkpoint.acceptable_evidence_sets:
                referenced.update(option)
        for members in self.context_members.values():
            referenced.update(members)
        return referenced


@dataclass(frozen=True)
class ExportReport:
    events: int
    contexts: int
    assignments: int
    queries: int
    redacted_events: int
    directory: Path


def _scrub_value(value: Any) -> Any:
    """Redact a parsed payload's strings, leaving its structure alone.

    Scrubbing the *serialised* payload instead is wrong, and not subtly: a JSON string holds
    `C:\\\\Users\\\\x`, so a pattern matching `C:\\` consumes one backslash of an escaped pair and
    leaves `\\x` behind, which is not a legal escape. Real events carry paths in their payload,
    so the export produced invalid JSON on the first real run while the unit tests, whose
    payloads had no paths, stayed green.
    """

    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {key: _scrub_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    return value


def _scrub_event(event: RawEvent) -> tuple[RawEvent, bool]:
    """Rebuild an event with its content and payload redacted."""

    content = scrub(event.content)
    payload: dict[str, Any] = _scrub_value(event.payload)
    # `create` recomputes `content_sha256`, which the contract requires to match the content.
    return (
        RawEvent.create(
            event_id=event.event_id,
            session_id=event.session_id,
            sequence=event.sequence,
            occurred_at=event.occurred_at,
            ingested_at=event.ingested_at,
            actor=event.actor,
            kind=event.kind,
            content=content,
            payload=payload,
            parent_event_id=event.parent_event_id,
        ),
        content != event.content or payload != event.payload,
    )


def export_sanitized(
    store: SQLiteEventStore,
    annotations: list[RealReplayAnnotation],
    directory: str | Path,
) -> ExportReport:
    """Write a publishable subset: only referenced events, every string scrubbed and checked."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)

    events: list[RawEvent] = []
    redacted = 0
    for annotation in annotations:
        referenced = annotation.referenced_event_ids()
        for event in store.list_events(annotation.source_session_id):
            if event.event_id not in referenced:
                continue
            cleaned, changed = _scrub_event(event)
            redacted += int(changed)
            events.append(cleaned)

    contexts = [context for annotation in annotations for context in annotation.contexts]
    members = {
        c.context_id: set(a.context_members.get(c.context_id, ()))
        for a in annotations
        for c in a.contexts
    }
    assignments = [
        EventContextAssignment(
            event_id=event.event_id,
            context_id=context.context_id,
            source="human",
            relevance=1.0,
            annotation_version=1,
            created_at=event.ingested_at,
        )
        for context in contexts
        for event in events
        if event.event_id in members.get(context.context_id, set())
    ]

    # Check the redacted fields, not the whole document: identifiers are importer-generated
    # UUIDs and are deliberately not redacted, so scanning them only produces false alarms.
    for event in events:
        where = f"{destination}/sanitized.json:{event.event_id}"
        assert_clean(event.content, where=where)
        assert_clean(json.dumps(event.payload, ensure_ascii=False), where=where)
    for context in contexts:
        assert_clean(context.model_dump_json(), where=f"{destination}/sanitized.json")
    for item in assignments:
        assert_clean(item.model_dump_json(), where=f"{destination}/sanitized.json")

    blob = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "contexts": [context.model_dump(mode="json") for context in contexts],
            "assignments": [item.model_dump(mode="json") for item in assignments],
        },
        ensure_ascii=False,
    )
    (destination / "sanitized.json").write_text(blob + "\n", encoding="utf-8")

    query_rows = [
        checkpoint.model_dump(mode="json")
        for annotation in annotations
        for checkpoint in annotation.checkpoints
    ]
    # `notes` is the only free text here; the rest is ids and labels.
    for checkpoint in (c for a in annotations for c in a.checkpoints):
        assert_clean(
            checkpoint.notes,
            where=f"{destination}/benchmark.jsonl:{checkpoint.sample_id}",
        )
    queries_blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in query_rows)
    (destination / "benchmark.jsonl").write_text(queries_blob, encoding="utf-8")

    return ExportReport(
        events=len(events),
        contexts=len(contexts),
        assignments=len(assignments),
        queries=len(query_rows),
        redacted_events=redacted,
        directory=destination,
    )


def to_benchmark_queries(
    annotation: RealReplayAnnotation, events: list[RawEvent]
) -> list[BenchmarkQuery]:
    """Convert an annotation into the contract the router already consumes."""

    by_id = {event.event_id: event for event in events}
    queries: list[BenchmarkQuery] = []
    for checkpoint in annotation.checkpoints:
        if checkpoint.query_event_id not in by_id:
            raise KeyError(f"checkpoint {checkpoint.sample_id} has no query event in the pool")
        queries.append(
            BenchmarkQuery(
                sample_id=checkpoint.sample_id,
                session_id=annotation.source_session_id,
                query_event_id=checkpoint.query_event_id,
                as_of_sequence=checkpoint.as_of_sequence,
                language=checkpoint.language,
                query_type=checkpoint.query_type,
                required_context_ids=list(checkpoint.required_context_ids),
                acceptable_evidence_sets=[
                    list(option) for option in checkpoint.acceptable_evidence_sets
                ],
                forbidden_future_event_ids=[
                    event.event_id for event in events if event.sequence > checkpoint.as_of_sequence
                ][:3],
                relation_label=checkpoint.relation_label,
                answer_requirements=[],
                must_abstain=checkpoint.must_abstain,
                difficulty=checkpoint.difficulty,
            )
        )
    return queries


class AnnotationCoverage(Contract):
    """Independent double annotation, and the weaker same-annotator recheck beside it.

    These are kept apart deliberately. The plan's 20% floor is about independent annotators,
    because the number it produces is how much the labels depend on *who* is labelling. A
    recheck by the same annotator measures test-retest stability instead, which is a weaker
    property, and folding it into `double_rate` would report the weaker thing as the stronger.
    """

    checkpoints: int = Field(ge=0)
    double_annotated: int = Field(ge=0)
    adjudicated: int = Field(ge=0)
    double_rate: float = Field(ge=0.0, le=1.0)
    rechecked: int = Field(ge=0)
    recheck_context_agreement: float = Field(ge=0.0, le=1.0)
    recheck_type_agreement: float = Field(ge=0.0, le=1.0)


def annotation_coverage(annotations: list[RealReplayAnnotation]) -> AnnotationCoverage:
    """Double-annotation coverage and the recheck rates, reported separately."""

    checkpoints = [item for annotation in annotations for item in annotation.checkpoints]
    double = sum(1 for item in checkpoints if item.second_annotator)
    adjudicated = sum(1 for item in checkpoints if item.adjudicated_by)
    rechecked = [item for item in checkpoints if item.recheck_annotator]
    return AnnotationCoverage(
        checkpoints=len(checkpoints),
        double_annotated=double,
        adjudicated=adjudicated,
        double_rate=double / len(checkpoints) if checkpoints else 0.0,
        rechecked=len(rechecked),
        recheck_context_agreement=(
            sum(1 for item in rechecked if item.recheck_contexts_agree) / len(rechecked)
            if rechecked
            else 0.0
        ),
        recheck_type_agreement=(
            sum(1 for item in rechecked if item.recheck_type_agrees) / len(rechecked)
            if rechecked
            else 0.0
        ),
    )


class RealReplayValidation(Contract):
    """A typed verdict, so callers do not have to index into `object` to read the errors."""

    valid: bool
    errors: list[str]
    coverage: AnnotationCoverage


def validate_real_replay(
    store: SQLiteEventStore,
    annotations: list[RealReplayAnnotation],
    *,
    enforce_double_annotation: bool = True,
) -> RealReplayValidation:
    """Check every label against the real conversation it claims to be about."""

    errors: list[str] = []
    for annotation in annotations:
        by_id = {event.event_id: event for event in store.list_events(annotation.source_session_id)}
        known_contexts = {context.context_id for context in annotation.contexts}
        for checkpoint in annotation.checkpoints:
            event = by_id.get(checkpoint.query_event_id)
            if event is None:
                errors.append(f"unknown query event: {checkpoint.sample_id}")
                continue
            if event.sequence != checkpoint.as_of_sequence:
                errors.append(f"as_of_sequence mismatch: {checkpoint.sample_id}")
            missing = set(checkpoint.required_context_ids) - known_contexts
            if missing:
                errors.append(f"undeclared contexts in {checkpoint.sample_id}: {sorted(missing)}")
            # A context can be declared and still not exist yet at this checkpoint. The first
            # real checkpoint required `ctx-t1-algo` while its own first event *was* that
            # checkpoint, so nothing of it was visible -- a label referring to the future that
            # the declaration check alone let through.
            for context_id in checkpoint.required_context_ids:
                members = [
                    by_id[event_id].sequence
                    for event_id in annotation.context_members.get(context_id, [])
                    if event_id in by_id
                ]
                # Strictly before: a context whose only event at this point *is* the checkpoint
                # has nothing to retrieve, so requiring it asks for material that cannot exist.
                # Testing "no member is later" instead passes that case, which is how the first
                # real checkpoint slipped through.
                if members and not any(
                    sequence < checkpoint.as_of_sequence for sequence in members
                ):
                    errors.append(
                        f"required context has no visible material at "
                        f"{checkpoint.sample_id}: {context_id}"
                    )
            if bool(checkpoint.required_context_ids) == checkpoint.must_abstain:
                errors.append(
                    f"must_abstain conflicts with required contexts: {checkpoint.sample_id}"
                )
            for option in checkpoint.acceptable_evidence_sets:
                if not option:
                    errors.append(f"empty evidence option in {checkpoint.sample_id}")
                for event_id in option:
                    evidence = by_id.get(event_id)
                    if evidence is None:
                        errors.append(f"unknown evidence {event_id} in {checkpoint.sample_id}")
                    elif evidence.sequence > checkpoint.as_of_sequence:
                        errors.append(f"non-causal evidence {event_id} in {checkpoint.sample_id}")
        for context_id, member_ids in annotation.context_members.items():
            if context_id not in known_contexts:
                errors.append(f"members declared for undeclared context: {context_id}")
            for event_id in member_ids:
                if event_id not in by_id:
                    errors.append(f"unknown member {event_id} of {context_id}")

    coverage = annotation_coverage(annotations)
    if enforce_double_annotation and coverage.double_rate < 0.2:
        errors.append(f"double annotation covers {coverage.double_rate:.0%}, below the 20% floor")
    return RealReplayValidation(valid=not errors, errors=errors, coverage=coverage)
