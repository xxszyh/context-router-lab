"""Leakage-aware routing, retrieval and statistical evaluation."""

from context_router.evaluation.arms import (
    ARM_NAMES,
    ArmCase,
    ArmCaseResult,
    ArmName,
    build_arm_cases,
    describe_router,
    evaluate_arms,
    first_gate,
    oracle_decision,
    run_arm,
)
from context_router.evaluation.metrics import (
    RouteCaseResult,
    evaluate_routes,
    evidence_set_recall,
    paired_bootstrap_delta,
)

__all__ = [
    "ARM_NAMES",
    "ArmCase",
    "ArmCaseResult",
    "ArmName",
    "RouteCaseResult",
    "build_arm_cases",
    "describe_router",
    "evaluate_arms",
    "evaluate_routes",
    "evidence_set_recall",
    "first_gate",
    "oracle_decision",
    "paired_bootstrap_delta",
    "run_arm",
]
