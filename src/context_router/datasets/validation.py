from __future__ import annotations

import re

from context_router.domain import BenchmarkQuery
from context_router.storage import SQLiteEventStore

_SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN .+ PRIVATE KEY-----)")


def validate_dataset(store: SQLiteEventStore, queries: list[BenchmarkQuery]) -> dict[str, object]:
    errors: list[str] = []
    contexts = {context.context_id for context in store.list_contexts()}
    seen_samples: set[str] = set()
    for query in queries:
        if query.sample_id in seen_samples:
            errors.append(f"duplicate sample_id: {query.sample_id}")
        seen_samples.add(query.sample_id)
        query_event = store.get_event(query.query_event_id)
        if query_event is None:
            errors.append(f"missing query event: {query.query_event_id}")
            continue
        if query_event.session_id != query.session_id:
            errors.append(f"session mismatch: {query.sample_id}")
        if query_event.sequence != query.as_of_sequence:
            errors.append(f"as_of_sequence mismatch: {query.sample_id}")
        missing_contexts = set(query.required_context_ids) - contexts
        if missing_contexts:
            errors.append(f"missing contexts for {query.sample_id}: {sorted(missing_contexts)}")
        if bool(query.required_context_ids) == query.must_abstain:
            errors.append(f"must_abstain conflicts with required contexts: {query.sample_id}")
        for evidence_set in query.acceptable_evidence_sets:
            for event_id in evidence_set:
                event = store.get_event(event_id)
                if event is None or event.sequence > query.as_of_sequence:
                    errors.append(f"non-causal evidence {event_id} in {query.sample_id}")
        for event_id in query.forbidden_future_event_ids:
            event = store.get_event(event_id)
            if event is None or event.sequence <= query.as_of_sequence:
                errors.append(f"invalid future event {event_id} in {query.sample_id}")
        session_events = store.list_events(query.session_id)
        for event in session_events:
            if _SECRET.search(event.content):
                errors.append(f"possible secret in event {event.event_id}")
    return {"valid": not errors, "samples": len(queries), "errors": errors}
