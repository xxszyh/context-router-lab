from __future__ import annotations

from datetime import UTC, datetime

from context_router.assembly import assemble_context
from context_router.domain import (
    AssemblyRequest,
    ContextCandidate,
    EventContextAssignment,
    FlatContext,
    RawEvent,
    RouteDecision,
)


def event(
    event_id: str,
    sequence: int,
    content: str,
    *,
    actor: str = "user",
    kind: str = "message",
    parent: str | None = None,
) -> RawEvent:
    return RawEvent.create(
        event_id=event_id,
        session_id="s1",
        sequence=sequence,
        occurred_at=datetime(2026, 1, 1, sequence, tzinfo=UTC),
        actor=actor,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        content=content,
        parent_event_id=parent,
    )


def context() -> FlatContext:
    return FlatContext(
        context_id="ctx-db",
        name="Database migration",
        goal="修复数据库迁移",
        summary="migration.py 报 SQLITE_BUSY",
        entities=["migration.py", "SQLITE_BUSY"],
        lexical_terms=["SQLite"],
        status="active",
        created_at_event="evt-1",
        last_active_sequence=4,
        version=1,
    )


def assignment(event_id: str) -> EventContextAssignment:
    return EventContextAssignment(
        event_id=event_id,
        context_id="ctx-db",
        source="oracle",
        relevance=1.0,
        annotation_version=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def decision() -> RouteDecision:
    return RouteDecision(
        decision="route",
        relation="continue",
        relation_probabilities={"continue": 1.0},
        candidates=[
            ContextCandidate(
                context_id="ctx-db",
                rrf_score=1.0,
                calibrated_probability=0.95,
                feature_values={},
                reasons=["oracle"],
            )
        ],
        selected_context_ids=["ctx-db"],
        confidence=0.95,
        fallback_level=0,
        trace_id="trace-1",
        index_version="index-1",
        model_versions={},
    )


def test_builder_is_causal_and_preserves_source_spans() -> None:
    past = event("evt-1", 1, "migration.py 使用旧 schema")
    evidence = event("evt-2", 2, "SQLITE_BUSY 是锁竞争导致")
    future = event("evt-9", 9, "未来才知道应该启用 WAL")
    request = AssemblyRequest(
        query="SQLITE_BUSY 为什么发生？",
        recent_events=[evidence],
        event_pool=[past, evidence, future],
        context_catalog=[context()],
        assignments=[assignment("evt-1"), assignment("evt-2"), assignment("evt-9")],
        as_of_sequence=3,
        token_budget=256,
    )

    working = assemble_context(request, decision())

    assert "SQLITE_BUSY 是锁竞争导致" in working.rendered_text
    assert "未来才知道" not in working.rendered_text
    assert "evt-9" not in working.included_event_ids
    assert "[Context: ctx-db | Database migration]" in working.rendered_text
    for span in working.source_spans:
        assert working.rendered_text[span.start_char : span.end_char]


def test_builder_keeps_tool_call_and_result_atomic() -> None:
    call = event("call-1", 2, "python migrate.py", actor="assistant", kind="tool_call")
    result = event(
        "result-1",
        3,
        "OperationalError: SQLITE_BUSY",
        actor="tool",
        kind="tool_result",
        parent="call-1",
    )
    request = AssemblyRequest(
        query="迁移命令报了什么错误？",
        recent_events=[],
        event_pool=[call, result],
        context_catalog=[context()],
        assignments=[assignment("call-1"), assignment("result-1")],
        as_of_sequence=3,
        token_budget=256,
    )

    working = assemble_context(request, decision())

    assert {"call-1", "result-1"}.issubset(working.included_event_ids)
    assert "python migrate.py" in working.rendered_text
    assert "OperationalError: SQLITE_BUSY" in working.rendered_text


def plot_context() -> FlatContext:
    return FlatContext(
        context_id="ctx-plot",
        name="Plot rendering",
        goal="修复论文图例",
        summary="plot.py 的 legend 遮挡坐标轴",
        entities=["plot.py", "legend"],
        lexical_terms=["matplotlib"],
        status="active",
        created_at_event="evt-1",
        last_active_sequence=4,
        version=1,
    )


def plot_assignment(event_id: str) -> EventContextAssignment:
    return EventContextAssignment(
        event_id=event_id,
        context_id="ctx-plot",
        source="oracle",
        relevance=1.0,
        annotation_version=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def two_context_decision() -> RouteDecision:
    return RouteDecision(
        decision="route",
        relation="cross_context",
        relation_probabilities={"cross_context": 1.0},
        candidates=[],
        selected_context_ids=["ctx-db", "ctx-plot"],
        confidence=0.95,
        fallback_level=0,
        trace_id="trace-2",
        index_version="index-1",
        model_versions={},
    )


def test_every_selected_context_gets_evidence_before_any_context_gets_depth() -> None:
    """A tight budget must be shared across selected contexts, not spent in rank order.

    A cross-context query needs evidence from both contexts. Ranking every candidate
    event globally and admitting them while tokens remain lets one context's long tail
    crowd out the other context's only evidence, which is how a correct routing decision
    still loses the answer. A minimal sufficient context must cover each selected context
    before deepening any of them.
    """

    database_events = [
        event(f"db-{index}", index, "SQLITE_BUSY SQLITE_BUSY 事务范围 细节确认")
        for index in range(1, 9)
    ]
    plot_events = [event("plot-1", 9, "legend 结论确定"), event("plot-2", 10, "legend 复核完成")]
    request = AssemblyRequest(
        query="SQLITE_BUSY SQLITE_BUSY 事务范围，另外 legend 的结论呢？",
        recent_events=[],
        event_pool=[*database_events, *plot_events],
        context_catalog=[context(), plot_context()],
        assignments=[
            *(assignment(item.event_id) for item in database_events),
            *(plot_assignment(item.event_id) for item in plot_events),
        ],
        as_of_sequence=10,
        token_budget=200,
    )

    working = assemble_context(request, two_context_decision())

    assert "legend 结论确定" in working.rendered_text
    assert {"plot-1", "plot-2"} <= set(working.included_event_ids)
    database_included = [i for i in working.included_event_ids if i.startswith("db-")]
    plot_included = [i for i in working.included_event_ids if i.startswith("plot-")]
    assert database_included, "the first context must still contribute evidence"
    assert len(plot_included) == len(plot_events)
    assert working.memory_tokens <= 200


def test_builder_stops_at_minimal_sufficiency_instead_of_filling_the_budget() -> None:
    """Low-relevance evidence is pollution, not a bonus for having tokens left.

    Admitting every group that fits spends most of the context on turns that a strong
    query match already made unnecessary. The floor uses only the system's own scores.
    """

    strong = event("strong-1", 1, "SQLITE_BUSY 锁竞争 根因确认")
    weak = [event(f"weak-{index}", index, f"迁移步骤 {index} 的普通记录") for index in range(2, 8)]
    request = AssemblyRequest(
        query="SQLITE_BUSY 锁竞争 根因确认",
        recent_events=[],
        event_pool=[strong, *weak],
        context_catalog=[context()],
        assignments=[assignment(item.event_id) for item in [strong, *weak]],
        as_of_sequence=8,
        token_budget=2048,
    )

    working = assemble_context(request, decision())

    assert "strong-1" in working.included_event_ids
    dropped_reasons = {str(d.get("reason")) for d in working.dropped_candidates}
    assert "minimal_sufficiency_floor" in dropped_reasons
    assert working.memory_tokens < 2048, "the budget must not be treated as a target"


def test_builder_honours_a_per_context_depth_cap() -> None:
    """A selected context may be capped so no single context dominates the budget."""

    from context_router.assembly import builder as builder_module

    events = [event(f"e-{index}", index, "SQLITE_BUSY 记录") for index in range(1, 10)]
    request = AssemblyRequest(
        query="SQLITE_BUSY",
        recent_events=[],
        event_pool=events,
        context_catalog=[context()],
        assignments=[assignment(item.event_id) for item in events],
        as_of_sequence=10,
        token_budget=2048,
    )
    original = builder_module.EVIDENCE_MAX_GROUPS_PER_CONTEXT
    try:
        builder_module.EVIDENCE_MAX_GROUPS_PER_CONTEXT = 1
        capped = assemble_context(request, decision())
    finally:
        builder_module.EVIDENCE_MAX_GROUPS_PER_CONTEXT = original

    reasons = {str(d.get("reason")) for d in capped.dropped_candidates}
    assert "minimal_sufficiency_depth" in reasons
    assert len(capped.included_event_ids) < len(events)


def test_builder_never_breaks_its_memory_budget_or_truncates_an_event() -> None:
    events = [
        event(f"evt-{index}", index, f"完整事件-{index} " + "细节 " * 20) for index in range(1, 8)
    ]
    request = AssemblyRequest(
        query="完整事件如何处理？",
        recent_events=[],
        event_pool=events,
        context_catalog=[context()],
        assignments=[assignment(item.event_id) for item in events],
        as_of_sequence=7,
        token_budget=80,
    )

    working = assemble_context(request, decision())

    assert working.memory_tokens <= 80
    assert working.dropped_candidates
    for included in working.included_event_ids:
        original = next(item for item in events if item.event_id == included)
        assert original.content in working.rendered_text
