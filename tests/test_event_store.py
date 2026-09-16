from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from context_router.domain import EventContextAssignment, FlatContext, RawEvent
from context_router.storage import DuplicateEventError, SequenceConflictError, SQLiteEventStore


def make_event(event_id: str, sequence: int, content: str = "hello") -> RawEvent:
    occurred = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=sequence)
    return RawEvent.create(
        event_id=event_id,
        session_id="session-1",
        sequence=sequence,
        occurred_at=occurred,
        actor="user",
        kind="message",
        content=content,
    )


def test_event_store_appends_and_replays_at_a_causal_cutoff(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    first = make_event("evt-1", 1, "first")
    future = make_event("evt-2", 2, "future")

    store.append_event(first)
    store.append_event(future)

    replay = store.list_events("session-1", as_of_sequence=1)
    assert replay == [first]


def test_event_store_rejects_duplicate_identity_and_sequence(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    store.append_event(make_event("evt-1", 1))

    with pytest.raises(DuplicateEventError):
        store.append_event(make_event("evt-1", 2))
    with pytest.raises(SequenceConflictError):
        store.append_event(make_event("evt-other", 1))


def test_context_assignments_are_versioned_without_mutating_events(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.db")
    event = make_event("evt-1", 1)
    store.append_event(event)
    store.upsert_context(
        FlatContext(
            context_id="ctx-a",
            name="Router",
            goal="Build routing",
            summary="Phase zero",
            entities=["RouteDecision"],
            lexical_terms=["router"],
            status="active",
            created_at_event="evt-1",
            last_active_sequence=1,
            version=1,
        )
    )
    store.append_assignment(
        EventContextAssignment(
            event_id="evt-1",
            context_id="ctx-a",
            source="human",
            relevance=0.8,
            annotation_version=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    store.append_assignment(
        EventContextAssignment(
            event_id="evt-1",
            context_id="ctx-a",
            source="human",
            relevance=1.0,
            annotation_version=2,
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
    )

    assignments = store.list_assignments(session_id="session-1", latest_only=True)
    assert len(assignments) == 1
    assert assignments[0].annotation_version == 2
    assert store.get_event("evt-1") == event


def test_jsonl_round_trip_preserves_events_contexts_and_assignments(tmp_path: Path) -> None:
    source = SQLiteEventStore(tmp_path / "source.db")
    event = make_event("evt-1", 1, "路径 src/router.py")
    context = FlatContext(
        context_id="ctx-a",
        name="Router",
        goal="Build routing",
        summary="Bilingual context",
        entities=["src/router.py"],
        lexical_terms=["路由"],
        status="active",
        created_at_event="evt-1",
        last_active_sequence=1,
        version=1,
    )
    assignment = EventContextAssignment(
        event_id="evt-1",
        context_id="ctx-a",
        source="oracle",
        relevance=1.0,
        annotation_version=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    source.append_event(event)
    source.upsert_context(context)
    source.append_assignment(assignment)

    export_path = tmp_path / "export.jsonl"
    source.export_jsonl(export_path)
    target = SQLiteEventStore(tmp_path / "target.db")
    counts = target.import_jsonl(export_path)

    assert counts == {"events": 1, "contexts": 1, "assignments": 1}
    assert target.get_event("evt-1") == event
    assert target.list_contexts() == [context]
    assert target.list_assignments(session_id="session-1") == [assignment]
