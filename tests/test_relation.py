from __future__ import annotations

import pytest

from context_router.domain import FlatContext, Relation, RouteRequest
from context_router.routing.relation import RuleRelationClassifier


def request(query: str) -> RouteRequest:
    return RouteRequest(
        query_event_id="query",
        query=query,
        recent_events=[],
        primary_context_id=None,
        recent_context_ids=[],
        context_catalog=[
            FlatContext(
                context_id="ctx-router",
                name="Context Router",
                goal="实现上下文路由",
                summary="RouteDecision 使用 RRF",
                entities=["routing.py"],
                lexical_terms=[],
                status="active",
                created_at_event="evt-0",
                last_active_sequence=1,
                version=1,
            )
        ],
        as_of_sequence=10,
    )


def classify(query: str) -> Relation:
    relation, probabilities = RuleRelationClassifier().classify(request(query))
    assert pytest.approx(1.0) == sum(probabilities.values())
    return relation


@pytest.mark.parametrize(
    "query",
    [
        "把 SQLite migration 的检索方法应用到 Context Router，可以吗？",  # the plan's own example
        "把 Context Router 的 RRF 思路用到 SQLite migration 上，可行吗？",
        "把 plot.py 的做法用在 Context Router 里行不行？",
        "借鉴 migration.py 的思路改 Context Router",
        "Can we apply the WAL approach to the matplotlib figure?",
        "Router、SQLite 和 plot.py 三者如何组合？",
    ],
)
def test_cross_context_paraphrases_are_recognized(query: str) -> None:
    assert classify(query) == "cross_context"


def test_english_cross_context_without_an_explicit_cue_is_unknown_not_a_guess() -> None:
    """Keyword rules cannot read cross-context intent out of every English phrasing.

    This is exactly the case the plan reserves for a small model. It is safe to leave as
    `unknown` only because the selection policy widens on `unknown` instead of collapsing
    to one confident context.
    """

    assert classify("Can the WAL result be shown in the matplotlib figure?") == "unknown"


@pytest.mark.parametrize(
    "query",
    [
        "现在切换到 migration.py 的 SQLITE_BUSY。",
        "现在转到 plot.py 的图例问题",
        "switch to the migration context",
        "换个话题，看 migration.py",
    ],
)
def test_switching_is_recognized(query: str) -> None:
    assert classify(query) == "switch_or_return"


@pytest.mark.parametrize(
    "query",
    [
        "回到之前的 Context Router，RRF 怎么算？",
        "回到 RouteDecision，继续完成 soft routing。",
        "Go back to plot.py and fix the legend.",
    ],
)
def test_returning_is_recognized(query: str) -> None:
    assert classify(query) == "switch_or_return"


def test_continuing_the_immediate_context_is_recognized() -> None:
    assert classify("继续这个，刚才的阈值怎么调？") == "continue"


def test_a_new_topic_is_recognized() -> None:
    assert classify("我想开始研究一个全新的量子化学项目") == "new_context"


@pytest.mark.parametrize(
    "query",
    [
        "那个尚未讨论的部署密钥是多少？",
        "SQLITE_BUSY 会不会影响最终结论？",
    ],
)
def test_unmatched_queries_stay_unknown(query: str) -> None:
    """`unknown` is the honest fallback: the router must widen, not guess."""
    assert classify(query) == "unknown"
