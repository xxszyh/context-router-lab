"""SQLite source-of-truth storage and JSONL exchange."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from context_router.domain import EventContextAssignment, FlatContext, RawEvent


class EventStoreError(RuntimeError):
    pass


class DuplicateEventError(EventStoreError):
    pass


class SequenceConflictError(EventStoreError):
    pass


class SQLiteEventStore:
    """Append-only raw events with rebuildable context projections."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    UNIQUE(session_id, sequence)
                );
                CREATE TRIGGER IF NOT EXISTS events_no_update
                BEFORE UPDATE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                BEFORE DELETE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                END;
                CREATE TABLE IF NOT EXISTS contexts (
                    context_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assignments (
                    event_id TEXT NOT NULL REFERENCES events(event_id),
                    context_id TEXT NOT NULL REFERENCES contexts(context_id),
                    source TEXT NOT NULL,
                    annotation_version INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    PRIMARY KEY(event_id, context_id, source, annotation_version)
                );
                CREATE INDEX IF NOT EXISTS assignments_context_idx
                    ON assignments(context_id, event_id);
                """
            )

    def append_event(self, event: RawEvent) -> None:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM events WHERE event_id = ?", (event.event_id,)
            ).fetchone():
                raise DuplicateEventError(event.event_id)
            if connection.execute(
                "SELECT 1 FROM events WHERE session_id = ? AND sequence = ?",
                (event.session_id, event.sequence),
            ).fetchone():
                raise SequenceConflictError(f"{event.session_id}:{event.sequence}")
            connection.execute(
                "INSERT INTO events(event_id, session_id, sequence, data) VALUES (?, ?, ?, ?)",
                (event.event_id, event.session_id, event.sequence, event.model_dump_json()),
            )

    def get_event(self, event_id: str) -> RawEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return RawEvent.model_validate_json(row["data"]) if row else None

    def list_events(self, session_id: str, as_of_sequence: int | None = None) -> list[RawEvent]:
        sql = "SELECT data FROM events WHERE session_id = ?"
        params: list[Any] = [session_id]
        if as_of_sequence is not None:
            sql += " AND sequence <= ?"
            params.append(as_of_sequence)
        sql += " ORDER BY sequence"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [RawEvent.model_validate_json(row["data"]) for row in rows]

    def upsert_context(self, context: FlatContext) -> None:
        with self._connect() as connection:
            current = connection.execute(
                "SELECT version FROM contexts WHERE context_id = ?", (context.context_id,)
            ).fetchone()
            if current and context.version < int(current["version"]):
                raise EventStoreError("cannot replace a context with an older version")
            connection.execute(
                """
                INSERT INTO contexts(context_id, version, data) VALUES (?, ?, ?)
                ON CONFLICT(context_id) DO UPDATE SET version=excluded.version, data=excluded.data
                """,
                (context.context_id, context.version, context.model_dump_json()),
            )

    def list_contexts(self) -> list[FlatContext]:
        with self._connect() as connection:
            rows = connection.execute("SELECT data FROM contexts ORDER BY context_id").fetchall()
        return [FlatContext.model_validate_json(row["data"]) for row in rows]

    def append_assignment(self, assignment: EventContextAssignment) -> None:
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO assignments(
                        event_id, context_id, source, annotation_version, data
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        assignment.event_id,
                        assignment.context_id,
                        assignment.source,
                        assignment.annotation_version,
                        assignment.model_dump_json(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise EventStoreError(str(error)) from error

    def list_assignments(
        self, *, session_id: str | None = None, latest_only: bool = False
    ) -> list[EventContextAssignment]:
        sql = """
            SELECT a.data FROM assignments a
            JOIN events e ON e.event_id = a.event_id
        """
        params: tuple[str, ...] = ()
        if session_id is not None:
            sql += " WHERE e.session_id = ?"
            params = (session_id,)
        sql += " ORDER BY e.sequence, a.context_id, a.source, a.annotation_version"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        values = [EventContextAssignment.model_validate_json(row["data"]) for row in rows]
        if not latest_only:
            return values
        latest: dict[tuple[str, str, str], EventContextAssignment] = {}
        for value in values:
            key = (value.event_id, value.context_id, value.source)
            if key not in latest or value.annotation_version > latest[key].annotation_version:
                latest[key] = value
        return list(latest.values())

    def _records(self) -> Iterator[dict[str, Any]]:
        with self._connect() as connection:
            for row in connection.execute("SELECT data FROM events ORDER BY session_id, sequence"):
                yield {"record_type": "event", "data": json.loads(row["data"])}
            for row in connection.execute("SELECT data FROM contexts ORDER BY context_id"):
                yield {"record_type": "context", "data": json.loads(row["data"])}
            for row in connection.execute(
                "SELECT data FROM assignments ORDER BY event_id, context_id, annotation_version"
            ):
                yield {"record_type": "assignment", "data": json.loads(row["data"])}

    def export_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for record in self._records():
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def import_jsonl(self, path: str | Path) -> dict[str, int]:
        parsed: dict[str, list[dict[str, Any]]] = {
            "event": [],
            "context": [],
            "assignment": [],
        }
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_type = record.get("record_type")
                if record_type not in parsed:
                    raise EventStoreError(f"unknown record_type at line {line_number}")
                parsed[record_type].append(record["data"])
        for data in parsed["event"]:
            self.append_event(RawEvent.model_validate(data))
        for data in parsed["context"]:
            self.upsert_context(FlatContext.model_validate(data))
        for data in parsed["assignment"]:
            self.append_assignment(EventContextAssignment.model_validate(data))
        return {
            "events": len(parsed["event"]),
            "contexts": len(parsed["context"]),
            "assignments": len(parsed["assignment"]),
        }


__all__ = [
    "DuplicateEventError",
    "EventStoreError",
    "SQLiteEventStore",
    "SequenceConflictError",
]
