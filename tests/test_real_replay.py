from __future__ import annotations

import json
from datetime import UTC, datetime
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
    to_benchmark_queries,
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
    requirements: list[str] | None = None,
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
                answer_requirements=[] if requirements is None else requirements,
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


def test_validate_rejects_a_requirement_with_nothing_to_match_on(
    store: SQLiteEventStore,
) -> None:
    """An unscoreable requirement is not a strict one -- it can never be satisfied at all.

    `requirement_satisfied` returns False outright when a requirement carries no quoted span
    and no ASCII identifier, so a checkpoint holding one is capped below 1.0 forever, and the
    failure reads as a weak answer rather than as a broken label. Six real checkpoints carried
    one apiece ("必须实际生成...图", "必须交付...文件") before this check existed.
    """

    report = validate_real_replay(
        store,
        [annotation(requirements=["必须实际重新生成并导出最终图"])],
        enforce_double_annotation=False,
    )

    assert report.valid is False
    assert any("nothing to match on" in error for error in report.errors)


def test_validate_accepts_requirements_that_carry_a_quoted_span_or_identifier(
    store: SQLiteEventStore,
) -> None:
    report = validate_real_replay(
        store,
        [annotation(requirements=["指出「15 格」对不上", "提到 p2_user_new.py 收敛慢"])],
        enforce_double_annotation=False,
    )

    assert report.valid is True, report.errors


def test_answer_requirements_survive_conversion_to_the_router_contract(
    store: SQLiteEventStore,
) -> None:
    """They used to be dropped on the floor here, so labelling them changed nothing."""

    labelled = annotation(requirements=["指出「15 格」对不上"])
    events = store.list_events("real-1")

    queries = to_benchmark_queries(labelled, events)

    assert queries[0].answer_requirements == ["指出「15 格」对不上"]


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


def test_a_path_inside_the_payload_is_scrubbed_without_breaking_the_json() -> None:
    r"""Redaction must run on parsed values, not on serialised JSON.

    A JSON string holds `C:\\Users\\x`, so a pattern that eats `C:\` consumes one backslash of
    an escaped pair and leaves `\x`, which is not a legal escape. The first real export died
    on exactly this, because the fixture payloads had no paths in them.
    """

    from datetime import UTC, datetime

    from context_router.datasets.real_replay import _scrub_event

    original = RawEvent.create(
        event_id="evt-9",
        session_id="real-1",
        sequence=9,
        occurred_at=datetime(2026, 9, 1, 9, tzinfo=UTC),
        ingested_at=datetime(2026, 9, 1, 9, tzinfo=UTC),
        actor="tool",
        kind="tool_call",
        content="read the file",
        payload={"source_file": HOME, "nested": {"list": [HOME, "keep p3.py"]}},
    )

    cleaned, changed = _scrub_event(original)

    assert changed is True
    assert "somebody" not in json.dumps(cleaned.payload)
    assert "p3.py" in json.dumps(cleaned.payload), "the task identifier survives"
    # The round trip is the assertion that matters: this is what raised on real data.
    assert json.loads(cleaned.model_dump_json())["payload"]["source_file"].startswith("<home>")


def test_a_uuids_digit_run_is_not_reported_as_a_phone_number() -> None:
    """Importer-generated ids are not redacted, so scanning them only raises false alarms.

    A UUID hex group can be eleven digits starting with 1, which is exactly a mainland mobile
    number's shape. The first real export failed on one, and the guard's own message then
    printed the digits -- a gate that logs the leak is part of the leak.
    """

    uuid_like = "claude:d22f2593-21d1-4c36-8156-737656f86e98:187208235433:0:message"

    assert_clean(uuid_like, where="identifier")  # must not raise


def test_the_guard_does_not_echo_what_it_caught() -> None:
    secret_value = "sk-" + "a" * 24

    with pytest.raises(ScrubError) as excinfo:
        assert_clean(f"token {secret_value}", where="test")

    assert secret_value not in str(excinfo.value)
    assert "chars)" in str(excinfo.value), "report the shape, not the value"


def test_a_same_annotator_recheck_does_not_count_as_double_annotation() -> None:
    """The two measure different things and must not be conflated.

    Independent annotators measure how much a label depends on *who* is labelling. One person
    twice measures test-retest stability, which is weaker. Folding the recheck into the
    double-annotation rate would report the weaker property as the stronger one, and the 20%
    floor exists precisely to force the stronger one.
    """

    from context_router.datasets.real_replay import annotation_coverage

    rechecked = annotation().model_copy(
        update={
            "checkpoints": [
                annotation()
                .checkpoints[0]
                .model_copy(
                    update={
                        "recheck_annotator": "a1",
                        "recheck_method": "same annotator, different route",
                        "recheck_contexts_agree": True,
                        "recheck_type_agrees": False,
                    }
                )
            ]
        }
    )

    coverage = annotation_coverage([rechecked])

    assert coverage.rechecked == 1
    assert coverage.recheck_context_agreement == 1.0
    assert coverage.recheck_type_agreement == 0.0
    assert coverage.double_annotated == 0, "a recheck is not an independent annotator"
    assert coverage.double_rate == 0.0


def test_validate_rejects_a_required_context_that_does_not_exist_yet(
    store: SQLiteEventStore,
) -> None:
    """Declared is not the same as visible.

    The first real checkpoint required `ctx-t1-algo` while that context's own first event was
    the checkpoint itself, so none of its material was in scope. The declaration check passed
    it; only the ablation exposed it.
    """

    early = annotation(required=["ctx-ice"]).model_copy(
        update={
            "checkpoints": [annotation().checkpoints[0].model_copy(update={"as_of_sequence": 0})]
        }
    )

    report = validate_real_replay(store, [early], enforce_double_annotation=False)

    assert report.valid is False
    assert any("no visible material" in error for error in report.errors), report.errors


def test_answer_after_skips_tool_traffic_and_reads_the_real_reply() -> None:
    """The first assistant event after a query is usually a tool call, not an answer.

    Measured on the first real batch: that mistake gave a median of 72 characters where the
    actual answers are a median of 1 520, which invalidated every judgement about what a reply
    said.
    """

    from context_router.datasets.real_replay import answer_after

    def built(event_id: str, sequence: int, actor: str, kind: str, content: str) -> RawEvent:
        return RawEvent.create(
            event_id=event_id,
            session_id="real-1",
            sequence=sequence,
            occurred_at=datetime(2026, 9, 1, min(sequence, 59), tzinfo=UTC),
            ingested_at=datetime(2026, 9, 1, min(sequence, 59), tzinfo=UTC),
            actor=actor,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content,
        )

    events = [
        built("q", 1, "user", "message", "这个问题怎么解？"),
        built("call", 2, "assistant", "tool_call", 'Read {"file_path": "x.py"}'),
        built("res", 3, "tool", "tool_result", "file contents"),
        built("a1", 4, "assistant", "message", "先看代码本身："),
        built("a2", 5, "assistant", "message", "对不上，35 格里 15 格四位小数有差异。"),
        built("q2", 6, "user", "message", "那怎么办？"),
        built("a3", 7, "assistant", "message", "这一段属于下一轮，不该被算进来。"),
    ]

    answer = answer_after(events, 1)

    assert "Read {" not in answer, "tool calls are not prose"
    assert "file contents" not in answer, "tool results are not the assistant's reply"
    assert "35 格里 15 格" in answer
    assert "下一轮" not in answer, "the reply stops at the next user turn"
