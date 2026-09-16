from __future__ import annotations

from datetime import UTC, datetime

from context_router.domain import FlatContext, RawEvent, RouteRequest
from context_router.routing import ContextRouter, route


def event(event_id: str, sequence: int, content: str) -> RawEvent:
    return RawEvent.create(
        event_id=event_id,
        session_id="s1",
        sequence=sequence,
        occurred_at=datetime(2026, 1, sequence, tzinfo=UTC),
        actor="user",
        kind="message",
        content=content,
    )


def context(
    context_id: str,
    name: str,
    goal: str,
    summary: str,
    entities: list[str],
    last_active: int,
) -> FlatContext:
    return FlatContext(
        context_id=context_id,
        name=name,
        goal=goal,
        summary=summary,
        entities=entities,
        lexical_terms=[],
        status="active",
        created_at_event="evt-0",
        last_active_sequence=last_active,
        version=1,
    )


def request(query: str, *, primary: str | None = "ctx-router") -> RouteRequest:
    contexts = [
        context(
            "ctx-router",
            "Context Router",
            "实现上下文路由和 soft routing",
            "RouteDecision 使用 RRF 和 calibrated probability",
            ["RouteDecision", "src/context_router/routing.py"],
            9,
        ),
        context(
            "ctx-database",
            "Database migration",
            "修复 SQLite schema migration",
            "数据库迁移在 migration.py 中失败，错误码 SQLITE_BUSY",
            ["migration.py", "SQLITE_BUSY", "SQLite"],
            7,
        ),
        context(
            "ctx-visual",
            "Plot rendering",
            "Render paper figures",
            "Matplotlib output and chart styles",
            ["matplotlib", "plot.py"],
            4,
        ),
    ]
    return RouteRequest(
        query_event_id="query",
        query=query,
        recent_events=[event("evt-9", 9, "我们正在实现 Router 的概率校准")],
        primary_context_id=primary,
        recent_context_ids=["ctx-database", "ctx-router"],
        context_catalog=contexts,
        as_of_sequence=10,
        max_selected_contexts=3,
        routing_profile="default",
    )


def test_route_short_coreference_continues_primary_context() -> None:
    decision = route(request("继续这个，刚才的阈值怎么调？"))

    assert decision.decision == "route"
    assert decision.relation == "continue"
    assert decision.selected_context_ids[0] == "ctx-router"


def test_route_preserves_code_identifiers_across_languages() -> None:
    decision = ContextRouter().route(request("migration.py 的 SQLITE_BUSY 怎么修？"))

    assert decision.selected_context_ids[0] == "ctx-database"
    database = next(c for c in decision.candidates if c.context_id == "ctx-database")
    assert database.entity_rank == 1
    assert "entity" in database.reasons


def test_route_cross_context_selects_two_relevant_contexts() -> None:
    decision = route(request("把 SQLite migration 的检索方法应用到 Context Router，可以吗？"))

    assert decision.relation == "cross_context"
    assert {"ctx-router", "ctx-database"}.issubset(decision.selected_context_ids)


def test_route_does_not_fake_confidence_for_an_unrelated_query() -> None:
    decision = route(request("我想开始研究一个全新的量子化学项目", primary=None))

    assert decision.decision in {"new_context_candidate", "abstain"}
    assert decision.confidence < 0.8


def test_route_widens_when_the_relation_is_unknown_but_two_contexts_are_relevant() -> None:
    """A relation-rule miss must widen the selection, never collapse it to top-1.

    "把 ... 思路用到 ... 上" is a natural cross-context paraphrase that the literal
    rules do not cover, so the classifier reports `unknown`. Treating `unknown` as a
    confident single-context continuation silently drops the second context the query
    actually needs, turning a classifier miss into lost evidence.
    """

    decision = route(
        request("把 Database migration 的 SQLITE_BUSY 思路用到 Context Router 上，可行吗？")
    )

    # Not a single-context continuation, whichever way the rules classify it.
    assert decision.relation in {"unknown", "cross_context"}, decision.relation
    assert {"ctx-router", "ctx-database"} <= set(decision.selected_context_ids)


def test_unknown_relation_never_selected_as_a_lone_confident_context() -> None:
    """`unknown` is an admission of ignorance, so it must not route a single context."""

    decision = route(request("SQLITE_BUSY 的思路和 Context Router 的 RRF 相比，哪个更稳？"))

    assert decision.relation == "unknown", decision.relation
    if decision.decision == "route":
        assert len(decision.selected_context_ids) != 1


def test_recent_window_text_does_not_drive_retrieval_away_from_the_named_target() -> None:
    """The recent window resolves coreference; it must not become the retrieval query.

    Concatenating recent turns into the retrieval query lets the context being left
    dominate BM25 and the dense channel, which is the context stickiness the plan warns
    about. A query that names its target must win on the query alone.
    """

    route_request = request("回到 Database migration，之前那个索引重建的结论是什么？").model_copy(
        update={
            "recent_events": [
                event("evt-7", 7, "Context Router 的 RouteDecision 校准已经完成"),
                event("evt-8", 8, "Context Router 的 soft routing 阈值还要复核"),
                event("evt-9", 9, "Context Router 的候选重排继续调整"),
            ]
        }
    )

    decision = route(route_request)

    assert decision.selected_context_ids[0] == "ctx-database"
    assert "ctx-router" not in decision.selected_context_ids


def test_a_return_target_is_not_penalised_for_being_cold() -> None:
    """`relation_match` must not reward the contexts a return query is leaving.

    A return reaches a context that is by construction not among the recent ones, so a
    bonus for recent contexts is a bonus against the correct answer.
    """

    route_request = request("回到 Database migration，之前那个索引重建的结论是什么？").model_copy(
        update={
            "primary_context_id": "ctx-router",
            "recent_context_ids": ["ctx-router"],
            "recent_events": [event("evt-9", 9, "Context Router 的候选重排继续调整")],
        }
    )

    decision = route(route_request)

    assert decision.selected_context_ids[0] == "ctx-database"
    database = next(c for c in decision.candidates if c.context_id == "ctx-database")
    router_candidate = next(c for c in decision.candidates if c.context_id == "ctx-router")
    assert database.calibrated_probability > router_candidate.calibrated_probability


def test_route_never_returns_more_than_requested_contexts() -> None:
    route_request = request("Router、SQLite 和 plot.py 三者如何组合？")
    route_request = route_request.model_copy(update={"max_selected_contexts": 2})

    decision = route(route_request)

    assert len(decision.selected_context_ids) <= 2
