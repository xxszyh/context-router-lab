from __future__ import annotations

import pytest

from context_router.datasets import generate_synthetic_dataset
from context_router.evaluation.arms import (
    ARM_NAMES,
    ArmCase,
    ArmCaseResult,
    assemble_arm,
    build_arm_cases,
    evaluate_arms,
    first_gate,
    oracle_decision,
    run_arm,
)
from context_router.storage import SQLiteEventStore


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> SQLiteEventStore:
    workspace = tmp_path_factory.mktemp("arms")
    dataset = generate_synthetic_dataset(session_count=2)
    dataset.write(workspace / "dataset")
    event_store = SQLiteEventStore(workspace / "events.sqlite")
    event_store.import_jsonl(workspace / "dataset" / "records.jsonl")
    return event_store


@pytest.fixture(scope="module")
def cases(store: SQLiteEventStore) -> list[ArmCase]:
    dataset = generate_synthetic_dataset(session_count=2)
    return build_arm_cases(store, dataset.queries)


@pytest.fixture(scope="module")
def results(cases: list[ArmCase]) -> list[ArmCaseResult]:
    return [run_arm(name, case) for case in cases for name in ARM_NAMES]


def test_arm_names_cover_the_planned_baselines() -> None:
    assert set(ARM_NAMES) == {
        "query_recent_only",
        "full_history",
        "sliding_window",
        "summary_recent",
        "global_bm25",
        "global_dense",
        "global_hybrid",
        "hybrid_router",
        "indexed_router",
        "oracle_router",
    }


def test_build_arm_cases_is_strictly_causal(cases: list[ArmCase]) -> None:
    assert cases
    for case in cases:
        assert case.events
        assert all(event.sequence <= case.as_of_sequence for event in case.events)
        assert all(event.sequence <= case.as_of_sequence for event in case.recent_events)


def test_arm_cases_preserve_the_benchmark_labels(
    cases: list[ArmCase], store: SQLiteEventStore
) -> None:
    dataset = generate_synthetic_dataset(session_count=2)
    assert [case.sample_id for case in cases] == [query.sample_id for query in dataset.queries]
    assert all(
        case.required_context_ids == query.required_context_ids
        for case, query in zip(cases, dataset.queries, strict=True)
    )


def test_every_arm_never_leaks_a_future_event(
    cases: list[ArmCase], results: list[ArmCaseResult]
) -> None:
    for case in cases:
        leaked = [result for result in results if result.sample_id == case.sample_id]
        assert leaked
        for result in leaked:
            assert result.future_leakage == 0, (result.arm, case.sample_id)


def test_future_events_exist_in_the_fixture_and_are_excluded(
    cases: list[ArmCase], store: SQLiteEventStore
) -> None:
    case = cases[0]
    future = {
        event.event_id
        for event in store.list_events(case.session_id)
        if event.sequence > case.as_of_sequence
    }
    assert future
    for name in ARM_NAMES:
        assert future.isdisjoint(run_arm(name, case).included_event_ids), name


def test_no_arm_exceeds_its_token_budget_except_full_history(results: list[ArmCaseResult]) -> None:
    budgets = {result.sample_id: result.token_budget for result in results}
    for result in results:
        if result.arm == "full_history":
            continue
        assert result.memory_tokens <= budgets[result.sample_id], (result.arm, result.sample_id)


def test_indexed_router_differs_from_hybrid_router_only_in_form(
    cases: list[ArmCase],
) -> None:
    """The pair is an ablation of rendering, and that is the only thing it may vary.

    If the selection drifted -- a different context, a different event, a different order --
    then a difference between the two arms would be unattributable: it could be the form or it
    could be the routing, and nothing in the result would say which. So the identity of the
    selection is asserted here rather than assumed, and the rendering is asserted to differ.
    """

    differed = 0
    for case in cases:
        prose = assemble_arm("hybrid_router", case)
        index = assemble_arm("indexed_router", case)

        assert index.selected_context_ids == prose.selected_context_ids, case.sample_id
        assert index.included_event_ids == prose.included_event_ids, case.sample_id
        assert index.decision == prose.decision, case.sample_id

        if prose.rendered_text:
            assert index.rendered_text != prose.rendered_text, case.sample_id
            differed += 1

    assert differed, "no checkpoint rendered differently, so the arm is not doing anything"


def test_the_indexed_form_costs_no_more_than_the_prose_it_replaces(
    cases: list[ArmCase],
) -> None:
    """Truncating bodies to a head cannot cost more than keeping them whole.

    This is the property that makes the arm worth running: if the compact form is also more
    expensive, the ablation has no case at all.
    """

    for case in cases:
        prose = assemble_arm("hybrid_router", case)
        index = assemble_arm("indexed_router", case)
        assert index.memory_tokens <= prose.memory_tokens, case.sample_id


def test_a_short_turn_falls_back_to_prose_rather_than_growing(cases: list[ArmCase]) -> None:
    """The indexed form is a compression, so it must never expand.

    A short turn costs less to keep whole than to re-render with its metadata attached. If a row
    grew, the arm would exceed the budget its prose twin respected -- and, because the budget fit
    measures the prose cost, the admitted set would stop matching too. Both invariants hold only
    while every row is no larger than the prose it replaces, so the renderer falls back rather
    than trusting the data to be long.
    """

    from context_router.assembly.builder import ContextBuilder, _Block  # noqa: PLC0415

    builder = ContextBuilder(render_mode="index")
    short = _Block(text="[Recent][evt-1][user/message] 好", section="recent", context_id="c1")

    assert builder._row(0, short) == short.text


def test_the_indexed_form_keeps_every_events_identity(cases: list[ArmCase]) -> None:
    """A truncated body must not become an anonymous row: the index still says which turn it is.

    Losing the sequence number would leave the model unable to refer to a turn it can see, which
    is the whole reason the form exists.

    The synthetic fixture's turns are short enough that every row falls back to prose, so this
    inflates one checkpoint's turns to give the renderer something worth compressing.
    """

    from datetime import UTC, datetime

    from context_router.assembly.builder import (  # noqa: PLC0415
        INDEX_HEAD_CHARS,
        ContextBuilder,
        _Block,
    )
    from context_router.domain import RawEvent  # noqa: PLC0415

    # Tested at the renderer rather than through an arm: the synthetic fixture's turns are short
    # enough that every row falls back to prose, and inflating them far enough to compress
    # changes what the router selects, which would make this a test of the fixture.
    long_event = RawEvent.create(
        event_id="evt-7",
        session_id="syn-000",
        sequence=7,
        occurred_at=datetime(2026, 9, 1, tzinfo=UTC),
        ingested_at=datetime(2026, 9, 1, tzinfo=UTC),
        actor="assistant",
        kind="message",
        content="上下文填充 " * 200,
    )
    prose = _Block(
        text=f"[Evidence:c1][evt-7][assistant/message] {long_event.content}",
        section="evidence",
        context_id="c1",
        event=long_event,
    )

    row = ContextBuilder(render_mode="index")._row(3, prose)

    assert row.startswith("[3] evidence ctx=c1 seq=7 assistant/message: "), row[:60]
    assert row.endswith("…"), "a long body must be truncated, not carried whole"
    assert str(INDEX_HEAD_CHARS) not in row, "the head limit is characters, not a printed count"
    assert len(row) < len(prose.text), "the row must be smaller than the prose it replaces"


def test_full_history_contains_the_whole_causal_session(cases: list[ArmCase]) -> None:
    case = cases[0]
    result = run_arm("full_history", case)
    assert set(result.included_event_ids) == {event.event_id for event in case.events}


def test_oracle_router_selects_exactly_the_required_contexts(cases: list[ArmCase]) -> None:
    for case in cases:
        decision = oracle_decision(case)
        assert decision.selected_context_ids == case.required_context_ids
        assert decision.decision == "route"
        assert decision.confidence == 1.0
        assert decision.model_versions["oracle"] == "ground-truth-labels"
        assert run_arm("oracle_router", case).selected_context_ids == case.required_context_ids


def test_oracle_router_recovers_the_labeled_evidence_sets(cases: list[ArmCase]) -> None:
    scored = [case for case in cases if case.acceptable_evidence_sets]
    assert scored
    for case in scored:
        included = set(run_arm("oracle_router", case).included_event_ids)
        assert included >= set(case.acceptable_evidence_sets[0]), case.sample_id
        assert run_arm("oracle_router", case).evidence_set_recall == 1.0


def test_global_lexical_retrieval_beats_the_recent_window(
    cases: list[ArmCase], results: list[ArmCaseResult]
) -> None:
    summary = evaluate_arms(results)
    assert (
        summary["global_bm25"]["evidence_set_recall"]
        > summary["query_recent_only"]["evidence_set_recall"]
    )
    assert summary["query_recent_only"]["evidence_set_recall"] < 1.0
    # The dataset must contain evidence the recent window cannot reach.
    far = [
        case
        for case in cases
        if case.acceptable_evidence_sets
        and not set(case.acceptable_evidence_sets[0])
        <= {event.event_id for event in case.recent_events}
    ]
    assert far


def test_router_arms_route_through_the_context_builder(cases: list[ArmCase]) -> None:
    oracle = run_arm("oracle_router", cases[0])
    assert oracle.selected_context_ids == cases[0].required_context_ids
    assert "context" in oracle.sections
    assert oracle.routing_trace_id
    hybrid = run_arm("hybrid_router", cases[0])
    assert hybrid.decision in {"route", "abstain", "new_context_candidate"}
    assert hybrid.routing_trace_id
    assert 0.0 <= hybrid.confidence <= 1.0


def test_hybrid_router_abstains_like_the_domain_contract(cases: list[ArmCase]) -> None:
    every = [run_arm("hybrid_router", case) for case in cases]
    assert all(result.decision in {"route", "abstain", "new_context_candidate"} for result in every)
    assert all(0.0 <= result.confidence <= 1.0 for result in every)


def test_evaluate_arms_reports_a_row_per_arm(results: list[ArmCaseResult]) -> None:
    summary = evaluate_arms(results)
    assert set(summary) == set(ARM_NAMES)
    for name in ARM_NAMES:
        row = summary[name]
        assert row["count"] == len(results) / len(ARM_NAMES)
        assert row["future_leakage_total"] == 0.0
        assert 0.0 <= row["evidence_set_recall"] <= 1.0
        assert row["mean_memory_tokens"] > 0
        assert "median_memory_tokens" in row


def test_oracle_reduces_memory_tokens_on_the_synthetic_dataset(
    results: list[ArmCaseResult],
) -> None:
    summary = evaluate_arms(results)
    reduction = summary["oracle_router"]["token_reduction_vs_full_history"]
    assert 0.0 < reduction < 1.0
    assert (
        summary["full_history"]["mean_memory_tokens"]
        > summary["oracle_router"]["mean_memory_tokens"]
    )


def test_hybrid_arm_reports_a_configured_router(cases: list[ArmCase]) -> None:
    """The arm must be able to measure a trained ranker / tuned policy, not just defaults."""

    from context_router.routing import ContextRouter
    from context_router.routing.router import RoutingPolicy

    default = run_arm("hybrid_router", cases[0])
    assert "ranker=heuristic-v1" in default.router_profile
    assert "policy(t_low=0.35" in default.router_profile
    assert run_arm("oracle_router", cases[0]).router_profile == "ground-truth"
    assert run_arm("full_history", cases[0]).router_profile == "deterministic"

    # A policy that refuses to settle for one context must actually change the outcome.
    wide = ContextRouter(policy=RoutingPolicy(t_high=0.99, margin=0.99))
    configured = run_arm("hybrid_router", cases[0], router=wide)
    assert configured.router_profile != default.router_profile
    assert "t_high=0.99" in configured.router_profile
    assert len(configured.selected_context_ids) >= len(default.selected_context_ids)


def test_generator_session_start_produces_disjoint_sessions() -> None:
    """Train and test splits must not share a conversation."""

    train = generate_synthetic_dataset(session_count=2, session_start=0)
    test = generate_synthetic_dataset(session_count=2, session_start=2)
    train_ids = {event.session_id for event in train.events}
    test_ids = {event.session_id for event in test.events}
    assert train_ids == {"syn-000", "syn-001"}
    assert test_ids == {"syn-002", "syn-003"}
    assert train_ids.isdisjoint(test_ids)


def test_full_history_outgrows_the_memory_budget(results: list[ArmCaseResult]) -> None:
    """The dataset must be large enough that the budget actually constrains the arms."""

    summary = evaluate_arms(results)
    assert summary["full_history"]["max_memory_tokens"] > 2048


def test_arm_cases_only_expose_contexts_that_already_exist(
    cases: list[ArmCase], store: SQLiteEventStore
) -> None:
    early = cases[0]
    created = {event.event_id for event in early.events}
    assert all(context.created_at_event in created for context in early.contexts)
    session_contexts = [
        context
        for context in store.list_contexts()
        if context.context_id.startswith(early.session_id)
    ]
    assert len(early.contexts) < len(session_contexts)


def test_first_gate_passes_on_reduced_tokens_and_preserved_recall() -> None:
    gate = first_gate(
        {
            "full_history": {"mean_memory_tokens": 1000.0, "evidence_set_recall": 1.0},
            "oracle_router": {"mean_memory_tokens": 600.0, "evidence_set_recall": 1.0},
        }
    )
    assert gate["oracle_token_reduction"] == pytest.approx(0.4)
    assert gate["oracle_recall_not_worse_than_full_history"] is True
    assert gate["verdict"] == "continue"
    assert gate["answer_quality_verified"] is False


def test_first_gate_stops_when_oracle_cannot_reduce_tokens() -> None:
    gate = first_gate(
        {
            "full_history": {"mean_memory_tokens": 1000.0, "evidence_set_recall": 0.9},
            "oracle_router": {"mean_memory_tokens": 900.0, "evidence_set_recall": 0.9},
        }
    )
    assert gate["verdict"] == "stop"
    assert gate["oracle_token_reduction"] == pytest.approx(0.1)


def test_first_gate_stops_when_oracle_recall_degrades() -> None:
    gate = first_gate(
        {
            "full_history": {"mean_memory_tokens": 1000.0, "evidence_set_recall": 0.95},
            "oracle_router": {"mean_memory_tokens": 400.0, "evidence_set_recall": 0.50},
        }
    )
    assert gate["oracle_token_reduction"] == pytest.approx(0.6)
    assert gate["oracle_recall_not_worse_than_full_history"] is False
    assert gate["verdict"] == "stop"
