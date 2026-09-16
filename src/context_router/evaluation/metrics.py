from __future__ import annotations

import math
import random
from statistics import mean
from typing import Literal

from pydantic import Field

from context_router.domain import Contract, Relation


class RouteCaseResult(Contract):
    sample_id: str
    session_id: str
    required_context_ids: list[str]
    selected_context_ids: list[str]
    candidate_order: list[str]
    decision: Literal["route", "abstain", "new_context_candidate"]
    confidence: float = Field(ge=0.0, le=1.0)
    relation_expected: Relation
    relation_predicted: Relation


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _ndcg(required: set[str], ranking: list[str]) -> float:
    if not required:
        return 1.0 if not ranking else 0.0
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, context_id in enumerate(ranking)
        if context_id in required
    )
    ideal = sum(1.0 / math.log2(index + 2) for index in range(len(required)))
    return _safe_div(dcg, ideal)


def evaluate_routes(cases: list[RouteCaseResult]) -> dict[str, object]:
    if not cases:
        raise ValueError("at least one route case is required")
    true_positive = false_positive = false_negative = 0
    exact: list[float] = []
    macro_precision: list[float] = []
    macro_recall: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    cross_recalls: list[float] = []
    correctness: list[float] = []
    relation_correct: list[float] = []

    for case in cases:
        required = set(case.required_context_ids)
        selected = set(case.selected_context_ids)
        intersection = required & selected
        tp = len(intersection)
        fp = len(selected - required)
        fn = len(required - selected)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        is_exact = float(required == selected)
        exact.append(is_exact)
        correctness.append(is_exact)
        macro_precision.append(_safe_div(tp, tp + fp) if selected else float(not required))
        macro_recall.append(_safe_div(tp, tp + fn) if required else float(not selected))
        relevant_ranks = [
            index
            for index, context_id in enumerate(case.candidate_order, start=1)
            if context_id in required
        ]
        reciprocal_ranks.append(1.0 / min(relevant_ranks) if relevant_ranks else 0.0)
        ndcgs.append(_ndcg(required, case.candidate_order))
        if len(required) >= 2:
            cross_recalls.append(_safe_div(tp, len(required)))
        relation_correct.append(float(case.relation_expected == case.relation_predicted))

    micro_precision = _safe_div(true_positive, true_positive + false_positive)
    micro_recall = _safe_div(true_positive, true_positive + false_negative)
    macro_p = mean(macro_precision)
    macro_r = mean(macro_recall)
    brier = mean(
        (case.confidence - target) ** 2 for case, target in zip(cases, correctness, strict=True)
    )
    ece = _expected_calibration_error(cases, correctness)
    high_confidence = [
        target for case, target in zip(cases, correctness, strict=True) if case.confidence >= 0.8
    ]
    return {
        "count": len(cases),
        "exact_context_set_accuracy": mean(exact),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": _safe_div(2 * macro_p * macro_r, macro_p + macro_r),
        "mrr": mean(reciprocal_ranks),
        "ndcg": mean(ndcgs),
        "cross_context_recall": mean(cross_recalls) if cross_recalls else 0.0,
        "relation_accuracy": mean(relation_correct),
        "brier_score": brier,
        "ece": ece,
        "high_confidence_error_rate": 1.0 - mean(high_confidence) if high_confidence else 0.0,
        "risk_coverage": _risk_coverage(cases, correctness),
    }


def _expected_calibration_error(
    cases: list[RouteCaseResult], correctness: list[float], bins: int = 10
) -> float:
    total = len(cases)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, case in enumerate(cases)
            if lower <= case.confidence < upper or (upper == 1.0 and case.confidence == 1.0)
        ]
        if not indices:
            continue
        confidence = mean(cases[index].confidence for index in indices)
        accuracy = mean(correctness[index] for index in indices)
        error += len(indices) / total * abs(confidence - accuracy)
    return error


def _risk_coverage(
    cases: list[RouteCaseResult], correctness: list[float]
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for threshold_integer in range(1, 10):
        threshold = threshold_integer / 10
        included = [
            target
            for case, target in zip(cases, correctness, strict=True)
            if case.confidence >= threshold
        ]
        points.append(
            {
                "threshold": threshold,
                "coverage": len(included) / len(cases),
                "risk": 1.0 - mean(included) if included else 0.0,
            }
        )
    return points


def evidence_set_recall(
    acceptable_evidence_sets: list[list[str]], included_event_ids: list[str]
) -> float:
    included = set(included_event_ids)
    if not acceptable_evidence_sets:
        return 1.0
    return float(any(set(option) <= included for option in acceptable_evidence_sets))


def paired_bootstrap_delta(
    baseline_by_session: dict[str, list[float]],
    candidate_by_session: dict[str, list[float]],
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    sessions = sorted(set(baseline_by_session) & set(candidate_by_session))
    if not sessions:
        raise ValueError("baseline and candidate have no shared sessions")
    deltas = {
        session: mean(candidate_by_session[session]) - mean(baseline_by_session[session])
        for session in sessions
    }
    observed = mean(deltas.values())
    randomizer = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sampled_sessions = [randomizer.choice(sessions) for _ in sessions]
        samples.append(mean(deltas[session] for session in sampled_sessions))
    samples.sort()
    low_index = max(0, math.floor(0.025 * (iterations - 1)))
    high_index = min(iterations - 1, math.ceil(0.975 * (iterations - 1)))
    return {
        "mean_delta": observed,
        "ci_low": samples[low_index],
        "ci_high": samples[high_index],
    }
