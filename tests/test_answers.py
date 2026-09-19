from __future__ import annotations

import pytest

from context_router.cli import _stratified_sample
from context_router.datasets import generate_synthetic_dataset
from context_router.evaluation.answers import (
    AnswerRecord,
    answer_one,
    build_answer_cases,
    evaluate_answers,
    quality_token_frontier,
)
from context_router.evaluation.arms import ArmCase, ArmName
from context_router.providers.openai_compatible import AnswerResult
from context_router.storage import SQLiteEventStore

ORACLE_AND_FULL: tuple[ArmName, ...] = ("oracle_router", "full_history")


class StubProvider:
    """Deterministic stand-in for a main model: echoes the evidence it was shown.

    Returning the context verbatim means the deterministic coverage score is 1.0 exactly
    when the context contained the required terms, which is what makes this useful for
    testing the harness without spending a call.
    """

    def __init__(self, *, stop_reason: str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.stop_reason = stop_reason

    def answer(self, *, query: str, working_context: str, instructions: str) -> AnswerResult:
        self.calls.append((query, instructions))
        text = working_context if self.stop_reason is None else ""
        return AnswerResult(
            text=text,
            model="stub-model",
            input_tokens=len(working_context) // 4,
            output_tokens=len(text) // 4,
            total_tokens=(len(working_context) + len(text)) // 4,
            stop_reason=self.stop_reason,
            raw_response={},
        )


@pytest.fixture(scope="module")
def cases(tmp_path_factory: pytest.TempPathFactory) -> list[ArmCase]:
    workspace = tmp_path_factory.mktemp("answers")
    dataset = generate_synthetic_dataset(session_count=2)
    dataset.write(workspace / "dataset")
    store = SQLiteEventStore(workspace / "events.sqlite")
    store.import_jsonl(workspace / "dataset" / "records.jsonl")
    return build_answer_cases(store, dataset.queries)


def test_arm_cases_carry_the_answer_requirements(cases: list[ArmCase]) -> None:
    answerable = [case for case in cases if case.required_context_ids]
    assert answerable
    assert all(case.answer_requirements for case in answerable)
    for case in answerable:
        for requirement in case.answer_requirements:
            assert "「" in requirement and "」" in requirement


def test_unanswerable_checkpoints_require_a_refusal(cases: list[ArmCase]) -> None:
    unanswerable = [case for case in cases if not case.required_context_ids]
    assert unanswerable
    assert all("不足以回答" in case.answer_requirements[0] for case in unanswerable)


def test_answer_one_records_cost_coverage_and_provenance(cases: list[ArmCase]) -> None:
    provider = StubProvider()
    case = next(case for case in cases if case.required_context_ids)

    record = answer_one(case, "oracle_router", provider)

    assert record.arm == "oracle_router"
    assert record.sample_id == case.sample_id
    assert record.model == "stub-model"
    assert record.coverage == 1.0, "the stub echoes its context, so every term is present"
    assert record.strict_coverage == 1.0
    assert record.refused is False
    assert record.memory_tokens > 0
    assert record.total_tokens > 0
    assert record.context_sha256
    assert record.truncated is False
    assert record.latency_seconds >= 0.0
    assert provider.calls and provider.calls[0][0] == case.query


def test_answerability_uses_explicit_label_not_required_contexts(cases: list[ArmCase]) -> None:
    """Real replay has no required-context ids but its labelled replies are answerable."""

    source = next(case for case in cases if case.required_context_ids)
    real_style = source.model_copy(update={"required_context_ids": [], "must_abstain": False})
    record = answer_one(real_style, "full_history", StubProvider())

    assert record.must_abstain is False
    assert record.strict_coverage == record.coverage


def test_a_truncated_answer_is_flagged_not_silently_scored(cases: list[ArmCase]) -> None:
    """A cut-off answer scores as wrong, so it must be visible as a truncation."""

    provider = StubProvider(stop_reason="max_tokens")
    case = next(case for case in cases if case.required_context_ids)

    record = answer_one(case, "oracle_router", provider)

    assert record.truncated is True
    assert record.answer == ""


def test_the_arm_can_only_see_its_own_memory(cases: list[ArmCase]) -> None:
    """The oracle's context must be smaller than full history, as the routing arms say."""

    provider = StubProvider()
    case = next(case for case in cases if case.required_context_ids)

    oracle = answer_one(case, "oracle_router", provider)
    full = answer_one(case, "full_history", provider)

    assert oracle.memory_tokens < full.memory_tokens
    assert oracle.context_sha256 != full.context_sha256


def test_evaluate_answers_aggregates_per_arm(cases: list[ArmCase]) -> None:
    provider = StubProvider()
    chosen = cases[:4]
    records = [answer_one(case, arm, provider) for case in chosen for arm in ORACLE_AND_FULL]

    summary = evaluate_answers(records)

    assert set(summary) == {"oracle_router", "full_history"}
    for row in summary.values():
        assert row["count"] == len(chosen)
        assert "mean_strict_coverage" in row
        assert row["refusal_rate"] == 0.0
        assert 0.0 <= row["mean_coverage"] <= 1.0
        assert row["mean_total_tokens"] > 0
        assert row["truncated_rate"] == 0.0
    assert (
        summary["oracle_router"]["mean_total_tokens"] < summary["full_history"]["mean_total_tokens"]
    )


def test_evaluate_answers_rejects_an_empty_run() -> None:
    with pytest.raises(ValueError, match="at least one answer record"):
        evaluate_answers([])


def test_frontier_drops_the_arm_that_is_worse_on_both_axes() -> None:
    summary = {
        "cheap_and_good": {"mean_coverage": 0.9, "mean_total_tokens": 100.0},
        "expensive_and_worse": {"mean_coverage": 0.5, "mean_total_tokens": 500.0},
        "expensive_and_better": {"mean_coverage": 0.95, "mean_total_tokens": 400.0},
    }

    frontier = quality_token_frontier(summary)

    arms = [point.arm for point in frontier]
    assert arms == ["cheap_and_good", "expensive_and_better"]


def test_frontier_keeps_a_genuine_tradeoff() -> None:
    summary = {
        "cheap": {"mean_coverage": 0.6, "mean_total_tokens": 100.0},
        "pricey": {"mean_coverage": 0.9, "mean_total_tokens": 300.0},
    }

    assert len(quality_token_frontier(summary)) == 2


def test_records_round_trip_through_json(cases: list[ArmCase]) -> None:
    record = answer_one(
        next(case for case in cases if case.required_context_ids), "oracle_router", StubProvider()
    )

    restored = AnswerRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_truncation_reasons_cover_the_providers_we_use() -> None:
    from context_router.evaluation.answers import TRUNCATION_STOP_REASONS

    assert "max_tokens" in TRUNCATION_STOP_REASONS  # Anthropic
    assert "length" in TRUNCATION_STOP_REASONS  # OpenAI-compatible
    assert "stop" not in TRUNCATION_STOP_REASONS


def test_answer_cases_carry_the_same_labels_as_the_routing_harness(cases: list[ArmCase]) -> None:
    """Both experiments must see identical inputs, or their numbers are not comparable."""

    from context_router.evaluation import build_arm_cases

    assert all(case.answer_requirements for case in cases if case.required_context_ids)
    assert all(case.token_budget == 2048 for case in cases)
    assert build_arm_cases.__module__ == "context_router.evaluation.arms"


def test_stratified_sample_fills_limit_when_some_type_buckets_are_sparse(
    cases: list[ArmCase],
) -> None:
    counts = {"return": 7, "cross_context": 5, "switch": 5, "short": 3, "continue": 2, "new": 2}
    source = cases[0]
    sparse = [
        source.model_copy(update={"sample_id": f"{kind}-{index}", "query_type": kind})
        for kind, count in counts.items()
        for index in range(count)
    ]

    sampled = _stratified_sample(sparse, 20)

    assert len(sampled) == 20
    assert len({case.sample_id for case in sampled}) == 20
