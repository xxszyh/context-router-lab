"""Run a main model over the comparison arms and score what comes back.

The routing arms measure whether the right evidence was *selected*. This module measures
whether the model could *use* it, which is a different question and the one the project's
claim actually rests on: recall of a labelled evidence set is a proxy for answer quality,
not a substitute for it. Two ways the proxy can fail, and only this module can see either --
a model that answers correctly from parametric knowledge without the evidence, and a model
that is handed the evidence and still answers wrongly.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from pydantic import Field

from context_router.domain import BenchmarkQuery, Contract
from context_router.evaluation.arms import (
    ArmCase,
    ArmCaseSource,
    ArmName,
    assemble_arm,
    build_arm_cases,
)
from context_router.evaluation.scoring import deterministic_coverage, is_refusal
from context_router.providers.openai_compatible import AnswerResult
from context_router.routing import ContextRouter

DEFAULT_INSTRUCTIONS = (
    "Answer the current query using only the supplied working context. Be concise: at most "
    "three sentences. Include the file or identifier the answer turns on. If the context "
    "does not contain the answer, say so plainly instead of guessing."
)

#: A provider that stopped for this reason produced a cut-off answer. Such an answer scores
#: as wrong, so it must be surfaced rather than silently charged to whichever arm was
#: unlucky enough to make the model verbose.
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "length", "incomplete"})


class AnswerProvider(Protocol):
    """Anything that can answer a query from a pre-assembled working context."""

    def answer(self, *, query: str, working_context: str, instructions: str) -> AnswerResult: ...


class AnswerRecord(Contract):
    """One arm's answer to one checkpoint, with the cost it took to produce."""

    arm: ArmName
    sample_id: str
    session_id: str
    query: str
    #: Carried on the record so an answer can be judged from the record alone, without
    #: rebuilding the dataset the requirements came from.
    answer_requirements: list[str]
    answer: str
    model: str
    memory_tokens: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    #: Coverage with a declining answer counted as zero on an answerable checkpoint, so a
    #: fluent "I cannot tell from this" cannot pass as an answer by echoing the rubric.
    strict_coverage: float = Field(ge=0.0, le=1.0)
    refused: bool
    must_abstain: bool
    context_sha256: str
    truncated: bool
    latency_seconds: float = Field(ge=0.0)


def answer_one(
    case: ArmCase,
    arm: ArmName,
    provider: AnswerProvider,
    *,
    instructions: str = DEFAULT_INSTRUCTIONS,
    router: ContextRouter | None = None,
) -> AnswerRecord:
    """Assemble one arm's context for one checkpoint and ask the model to answer it.

    The memory is built by ``assemble_arm`` rather than here, so the answer experiment
    sends exactly what the routing benchmark scored. Budgeting the two separately is how a
    harness ends up measuring something other than the thing it reports.
    """

    built = assemble_arm(arm, case, router=router)
    started = time.perf_counter()
    result = provider.answer(
        query=case.query, working_context=built.rendered_text, instructions=instructions
    )
    latency = time.perf_counter() - started
    coverage = deterministic_coverage(case.answer_requirements, result.text)
    refused = is_refusal(result.text)
    answerable = bool(case.required_context_ids)
    return AnswerRecord(
        arm=arm,
        sample_id=case.sample_id,
        session_id=case.session_id,
        query=case.query,
        answer_requirements=list(case.answer_requirements),
        answer=result.text,
        model=result.model,
        memory_tokens=built.memory_tokens,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
        coverage=coverage,
        strict_coverage=coverage if (answerable or not refused) else 0.0,
        refused=refused,
        must_abstain=not answerable,
        context_sha256=hashlib.sha256(built.rendered_text.encode("utf-8")).hexdigest()[:16],
        truncated=result.stop_reason in TRUNCATION_STOP_REASONS,
        latency_seconds=latency,
    )


def build_answer_cases(
    source: ArmCaseSource, queries: list[BenchmarkQuery], *, budget: int = 2048
) -> list[ArmCase]:
    """Reuse the arm harness's causal projection so both experiments agree on the inputs."""

    return build_arm_cases(source, queries, token_budget=budget)


def _mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def evaluate_answers(records: list[AnswerRecord]) -> dict[str, dict[str, float]]:
    """Aggregate answer records into one comparable row per arm."""

    if not records:
        raise ValueError("at least one answer record is required")
    grouped: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        grouped[record.arm].append(record)
    summary: dict[str, dict[str, float]] = {}
    for arm, rows in grouped.items():
        summary[arm] = {
            "count": float(len(rows)),
            "mean_coverage": fmean(row.coverage for row in rows),
            "mean_strict_coverage": fmean(row.strict_coverage for row in rows),
            "refusal_rate": _mean_or_zero([float(row.refused) for row in rows]),
            # Coverage that only counts answers the model actually attempted.
            "mean_attempted_coverage": _mean_or_zero(
                [row.coverage for row in rows if not row.must_abstain and not row.refused]
            ),
            "mean_answerable_coverage": _mean_or_zero(
                [row.coverage for row in rows if not row.must_abstain]
            ),
            # For an unanswerable checkpoint the coverage *is* the refusal score, because
            # its only requirement is to decline rather than invent an answer.
            "unanswerable_refusal_score": _mean_or_zero(
                [row.coverage for row in rows if row.must_abstain]
            ),
            "mean_memory_tokens": fmean(float(row.memory_tokens) for row in rows),
            "mean_input_tokens": fmean(float(row.input_tokens) for row in rows),
            "mean_output_tokens": fmean(float(row.output_tokens) for row in rows),
            "mean_total_tokens": fmean(float(row.total_tokens) for row in rows),
            "mean_latency_seconds": fmean(row.latency_seconds for row in rows),
            "truncated_rate": _mean_or_zero([float(row.truncated) for row in rows]),
            "empty_answer_rate": _mean_or_zero([float(not row.answer.strip()) for row in rows]),
        }
    return summary


@dataclass(frozen=True)
class FrontierPoint:
    """One arm's position on the quality/cost plane."""

    arm: str
    quality: float
    tokens: float


def quality_token_frontier(
    summary: dict[str, dict[str, float]], *, quality_key: str = "mean_coverage"
) -> list[FrontierPoint]:
    """Arms nothing else beats on both axes at once.

    Cost is provider-reported total tokens rather than the local estimate, because that is
    what a run actually spends and because output tokens vary by arm too.
    """

    points = [
        FrontierPoint(arm=arm, quality=row[quality_key], tokens=row["mean_total_tokens"])
        for arm, row in summary.items()
    ]
    frontier: list[FrontierPoint] = []
    for point in points:
        dominated = any(
            other.tokens <= point.tokens
            and other.quality >= point.quality
            and (other.tokens < point.tokens or other.quality > point.quality)
            for other in points
        )
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda item: item.tokens)
    return frontier
