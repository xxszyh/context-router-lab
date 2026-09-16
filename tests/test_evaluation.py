from __future__ import annotations

from context_router.evaluation import (
    RouteCaseResult,
    evaluate_routes,
    evidence_set_recall,
    paired_bootstrap_delta,
)


def number(metrics: dict[str, object], key: str) -> float:
    value = metrics[key]
    assert isinstance(value, float)
    return value


def risk_coverage(metrics: dict[str, object]) -> list[dict[str, float]]:
    value = metrics["risk_coverage"]
    assert isinstance(value, list)
    return value


def test_route_metrics_cover_multicontext_calibration_and_relation() -> None:
    cases = [
        RouteCaseResult(
            sample_id="a",
            session_id="s1",
            required_context_ids=["ctx-a"],
            selected_context_ids=["ctx-a"],
            candidate_order=["ctx-a", "ctx-b"],
            decision="route",
            confidence=0.9,
            relation_expected="continue",
            relation_predicted="continue",
        ),
        RouteCaseResult(
            sample_id="b",
            session_id="s2",
            required_context_ids=["ctx-a", "ctx-b"],
            selected_context_ids=["ctx-a"],
            candidate_order=["ctx-a", "ctx-c", "ctx-b"],
            decision="route",
            confidence=0.8,
            relation_expected="cross_context",
            relation_predicted="cross_context",
        ),
    ]

    metrics = evaluate_routes(cases)

    assert metrics["exact_context_set_accuracy"] == 0.5
    assert metrics["micro_precision"] == 1.0
    assert metrics["micro_recall"] == 2 / 3
    assert metrics["cross_context_recall"] == 0.5
    assert metrics["relation_accuracy"] == 1.0
    assert 0.0 <= number(metrics, "brier_score") <= 1.0
    assert len(risk_coverage(metrics)) == 9


def test_evidence_recall_accepts_any_complete_equivalent_set() -> None:
    acceptable = [["evt-1", "evt-2"], ["evt-3"]]

    assert evidence_set_recall(acceptable, ["evt-3"]) == 1.0
    assert evidence_set_recall(acceptable, ["evt-1"]) == 0.0


def test_paired_bootstrap_is_grouped_and_reproducible() -> None:
    baseline = {"s1": [0.8, 0.6], "s2": [0.7]}
    candidate = {"s1": [0.9, 0.7], "s2": [0.8]}

    result = paired_bootstrap_delta(baseline, candidate, iterations=500, seed=7)

    assert round(result["mean_delta"], 6) == 0.1
    assert result["ci_low"] <= result["mean_delta"] <= result["ci_high"]
