from __future__ import annotations

from pathlib import Path

import pytest

from context_router.datasets.real_replay import (
    AnnotatedCheckpoint,
    RealReplayAnnotation,
    ScrubError,
    annotation_coverage,
    assert_clean,
    export_sanitized,
    scrub,
    validate_real_replay,
)
from context_router.domain import EventContextAssignment, FlatContext, RawEvent
from context_router.storage import SQLiteEventStore

# Placeholder values only. A test file is committed, so a real number must never appear here.
PHONE = "13800138000"
EMAIL = "someone@example.com"
HOME = r"C:\Users\somebody\Desktop\proj\p2.py"


def test_scrub_removes_pii_but_keeps_the_task_identifier() -> None:
    """File and symbol names are the signal the benchmark runs on, so the tail must survive."""

    text = f"联系 {PHONE} 或 {EMAIL}，代码在 {HOME}"
    cleaned = scrub(text)

    assert PHONE not in cleaned
    assert EMAIL not in cleaned
    assert r"C:\Users\somebody" not in cleaned
    assert "p2.py" in cleaned, "the file name is what the query is about"
    assert "<phone>" in cleaned and "<email>" in cleaned


def test_assert_clean_refuses_anything_that_survived() -> None:
    """A partially scrubbed conversation is worse than none, because it looks safe."""

    with pytest.raises(ScrubError, match="phone"):
        assert_clean(f"call {PHONE}", where="test")
    with pytest.raises(ScrubError, match="email"):
        assert_clean(f"mail {EMAIL}", where="test")
    with pytest.raises(ScrubError, match="drive-path"):
        assert_clean(r"open C:\temp\x.py", where="test")


def test_clean_text_passes() -> None:
    assert_clean("回到 p2_user_new.py，北极海冰那一步怎么定？", where="test")


def event(event_id: str, sequence: int, content: str) -> RawEvent:
    from datetime import UTC, datetime

    return RawEvent.create(
        event_id=event_id,
        session_id="real-1",
        sequence=sequence,
        occurred_at=datetime(2026, 9, 1, sequence, tzinfo=UTC),
        ingested_at=datetime(2026, 9, 1, sequence, tzinfo=UTC),
        actor="user",
        kind="message",
        content=content,
    )


def context() -> FlatContext:
    return FlatContext(
        context_id="ctx-ice",
        name="Arctic sea ice",
        goal="修正 p2 的 Picard 迭代",
        summary="p2_user_new.py 收敛慢",
        entities=["p2_user_new.py"],
        lexical_terms=[],
        status="active",
        created_at_event="evt-1",
        last_active_sequence=1,
        version=1,
    )


def annotation(
    *,
    second: str | None = None,
    required: list[str] | None = None,
    evidence: list[list[str]] | None = None,
) -> RealReplayAnnotation:
    """Built with parameters rather than mutated: the contracts are frozen by design."""

    return RealReplayAnnotation(
        source_session_id="real-1",
        contexts=[context()],
        context_members={"ctx-ice": ["evt-1", "evt-2"]},
        checkpoints=[
            AnnotatedCheckpoint(
                sample_id="real-1-q-01",
                query_event_id="evt-3",
                as_of_sequence=3,
                query_type="return",
                relation_label="switch_or_return",
                required_context_ids=["ctx-ice"] if required is None else required,
                acceptable_evidence_sets=[["evt-2"]] if evidence is None else evidence,
                annotator="a1",
                second_annotator=second,
            )
        ],
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteEventStore:
    s = SQLiteEventStore(tmp_path / "events.sqlite")
    for item in (
        event("evt-1", 1, f"p2 收敛慢，路径 {HOME}"),
        event("evt-2", 2, "Picard 迭代次数不够"),
        event("evt-3", 3, "回到之前那个迭代问题"),
        event("evt-4", 4, "未来的事"),
    ):
        s.append_event(item)
    s.upsert_context(context())
    s.append_assignment(
        EventContextAssignment(
            event_id="evt-1",
            context_id="ctx-ice",
            source="human",
            relevance=1.0,
            annotation_version=1,
            created_at=item.ingested_at,
        )
    )
    return s


def test_export_publishes_only_referenced_events_and_scrubs_them(
    store: SQLiteEventStore, tmp_path: Path
) -> None:
    report = export_sanitized(store, [annotation()], tmp_path / "out")

    assert report.events == 3, "evt-4 is unreferenced and must not be exported"
    assert report.queries == 1
    assert report.redacted_events == 1
    blob = (tmp_path / "out" / "sanitized.json").read_text(encoding="utf-8")
    assert "somebody" not in blob and PHONE not in blob
    assert "p2.py" in blob, "the identifier the query turns on must survive"


def test_export_refuses_when_something_survives_scrubbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must fire before anything reaches disk."""

    import context_router.datasets.real_replay as module

    monkeypatch.setattr(module, "_REDACTIONS", ())
    store = SQLiteEventStore(tmp_path / "e.sqlite")
    store.append_event(event("evt-1", 1, f"call {PHONE}"))
    store.append_event(event("evt-2", 2, "x"))
    store.append_event(event("evt-3", 3, "q"))
    with pytest.raises(ScrubError):
        export_sanitized(store, [annotation()], tmp_path / "out")
    assert not (tmp_path / "out" / "sanitized.json").exists()


def test_validate_rejects_a_non_causal_evidence_label(store: SQLiteEventStore) -> None:
    """Labelled evidence must be visible at the checkpoint, or the replay is cheating."""

    report = validate_real_replay(
        store, [annotation(evidence=[["evt-4"]])], enforce_double_annotation=False
    )

    assert report.valid is False
    assert any("non-causal evidence" in error for error in report.errors)


def test_validate_rejects_an_undeclared_required_context(store: SQLiteEventStore) -> None:
    report = validate_real_replay(
        store, [annotation(required=["ctx-nope"])], enforce_double_annotation=False
    )

    assert report.valid is False
    assert any("undeclared contexts" in error for error in report.errors)


def test_validate_accepts_a_causally_sound_annotation(store: SQLiteEventStore) -> None:
    report = validate_real_replay(store, [annotation()], enforce_double_annotation=False)

    assert report.valid is True, report.errors


def test_double_annotation_floor_is_enforced(store: SQLiteEventStore) -> None:
    """The plan's 20% floor is a gate, not a suggestion."""

    report = validate_real_replay(store, [annotation()])

    assert report.valid is False
    assert any("below the 20% floor" in error for error in report.errors)
    assert annotation_coverage([annotation(second="a2")]).double_rate == 1.0


def test_annotation_round_trips_through_json(store: SQLiteEventStore) -> None:
    restored = RealReplayAnnotation.model_validate_json(annotation().model_dump_json())

    assert restored == annotation()


def test_a_url_is_not_mistaken_for_a_windows_path() -> None:
    """The first version of the guard flagged 172 of 7 705 real events, none a real path.

    `[A-Za-z]:[\\/]` matches the `s:/` inside `https://`, so the lookbehind that requires a
    real drive letter is load-bearing rather than cosmetic.
    """

    url = "see https://example.invalid/docs for the upstream reference"

    assert scrub(url) == url, "a URL must survive untouched"
    assert_clean(url, where="test")


def test_a_real_drive_path_is_still_caught_and_scrubbed() -> None:
    assert scrub(r"D:/data/input.csv").startswith("<abs>/")
    with pytest.raises(ScrubError, match="drive-path"):
        assert_clean(r"open D:/data/input.csv", where="test")
