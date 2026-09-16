from __future__ import annotations

import itertools
from dataclasses import dataclass

from context_router.routing.router import RoutingPolicy


@dataclass(frozen=True)
class PolicyObservation:
    probabilities: dict[str, float]
    required_context_ids: frozenset[str]
    relation: str


@dataclass(frozen=True)
class PolicyTuningResult:
    policy: RoutingPolicy
    feasible: bool
    required_context_recall: float
    high_confidence_error_rate: float
    cross_context_recall: float
    average_selected_contexts: float


class SelectionPolicyTuner:
    """Choose the smallest soft-routing policy satisfying recall and risk constraints."""

    def tune(self, observations: list[PolicyObservation]) -> PolicyTuningResult:
        if not observations:
            raise ValueError("at least one policy observation is required")
        results: list[PolicyTuningResult] = []
        for t_low, t_high, margin, maximum, confidence in itertools.product(
            (0.20, 0.30, 0.40, 0.50, 0.60),
            (0.50, 0.60, 0.70, 0.80, 0.90),
            (0.05, 0.10, 0.20, 0.30, 0.40),
            (1, 2, 3),
            (0.50, 0.60, 0.70, 0.80, 0.90),
        ):
            if t_high < t_low:
                continue
            policy = RoutingPolicy(
                t_low=t_low,
                t_high=t_high,
                margin=margin,
                max_candidates=20,
                confidence_threshold=confidence,
            )
            results.append(self._evaluate(policy, maximum, observations))
        feasible = [
            result
            for result in results
            if result.required_context_recall >= 0.95
            and result.high_confidence_error_rate <= 0.02
            and result.cross_context_recall >= 0.90
        ]
        if feasible:
            chosen = min(
                feasible,
                key=lambda result: (
                    result.average_selected_contexts,
                    -result.required_context_recall,
                    result.high_confidence_error_rate,
                ),
            )
            return PolicyTuningResult(
                policy=chosen.policy,
                feasible=True,
                required_context_recall=chosen.required_context_recall,
                high_confidence_error_rate=chosen.high_confidence_error_rate,
                cross_context_recall=chosen.cross_context_recall,
                average_selected_contexts=chosen.average_selected_contexts,
            )
        return max(
            results,
            key=lambda result: (
                result.required_context_recall,
                result.cross_context_recall,
                -result.high_confidence_error_rate,
                -result.average_selected_contexts,
            ),
        )

    @staticmethod
    def _evaluate(
        policy: RoutingPolicy,
        maximum: int,
        observations: list[PolicyObservation],
    ) -> PolicyTuningResult:
        recalls: list[float] = []
        cross_recalls: list[float] = []
        selected_counts: list[int] = []
        high_confidence_errors: list[float] = []
        for observation in observations:
            ranked = sorted(observation.probabilities.items(), key=lambda item: (-item[1], item[0]))
            first = ranked[0][1] if ranked else 0.0
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            if (
                ranked
                and first >= policy.t_high
                and first - second >= policy.margin
                and observation.relation != "cross_context"
            ):
                selected = {ranked[0][0]}
            else:
                selected = {key for key, value in ranked if value >= policy.t_low}
                selected = set(list(selected)[:maximum])
            if observation.relation == "cross_context":
                selected.update(key for key, _ in ranked[: min(2, maximum)])
            selected_counts.append(len(selected))
            required = observation.required_context_ids
            recall = len(required & selected) / len(required) if required else float(not selected)
            recalls.append(recall)
            if len(required) >= 2:
                cross_recalls.append(recall)
            if first >= policy.confidence_threshold:
                high_confidence_errors.append(float(not required <= selected))
        return PolicyTuningResult(
            policy=policy,
            feasible=False,
            required_context_recall=sum(recalls) / len(recalls),
            high_confidence_error_rate=(
                sum(high_confidence_errors) / len(high_confidence_errors)
                if high_confidence_errors
                else 0.0
            ),
            cross_context_recall=(
                sum(cross_recalls) / len(cross_recalls) if cross_recalls else 1.0
            ),
            average_selected_contexts=sum(selected_counts) / len(selected_counts),
        )
