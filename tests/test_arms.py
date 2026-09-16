from __future__ import annotations

import pytest

from context_router.datasets import generate_synthetic_dataset
from context_router.evaluation.arms import (
    ARM_NAMES,
    ArmCase,
    ArmCaseResult,
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
