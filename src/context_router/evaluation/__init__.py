"""Leakage-aware routing, retrieval and statistical evaluation."""

from context_router.evaluation.answers import (
    AnswerProvider,
    AnswerRecord,
    answer_one,
    build_answer_cases,
    evaluate_answers,
    quality_token_frontier,
)
from context_router.evaluation.arms import (
    ARM_NAMES,
    ArmAssembly,
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
from context_router.evaluation.judge import (
    CoverageJudge,
    Judge,
    JudgeOutcome,
    JudgePair,
    build_judge_prompt,
    run_pairwise_judging,
    summarise_wins,
)
from context_router.evaluation.metrics import (
    RouteCaseResult,
    evaluate_routes,
    evidence_set_recall,
    paired_bootstrap_delta,
)
from context_router.evaluation.scoring import deterministic_coverage, requirement_terms

__all__ = [
    "ARM_NAMES",
    "AnswerProvider",
    "AnswerRecord",
    "CoverageJudge",
    "Judge",
    "JudgeOutcome",
    "JudgePair",
    "ArmAssembly",
    "ArmCase",
    "ArmCaseResult",
    "ArmName",
    "RouteCaseResult",
    "answer_one",
    "assemble_arm",
    "build_answer_cases",
    "build_arm_cases",
    "build_judge_prompt",
    "describe_router",
    "deterministic_coverage",
    "evaluate_answers",
    "evaluate_arms",
    "evaluate_routes",
    "evidence_set_recall",
    "first_gate",
    "oracle_decision",
    "paired_bootstrap_delta",
    "quality_token_frontier",
    "requirement_terms",
    "run_pairwise_judging",
    "summarise_wins",
    "run_arm",
]
