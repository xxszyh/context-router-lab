"""Blinded pairwise judging, with position bias controlled by swapping the order.

Two properties make a judge's verdict usable, and both are enforced here rather than left
to whoever writes the prompt:

* the judge must not be able to tell which method produced which answer, so the arms' names
  and their token counts never enter the prompt;
* a judge that prefers whichever answer it reads first will manufacture a result out of
  nothing, so every pair is judged twice in both orders and a disagreement is recorded as
  a tie with ``agreement=False`` rather than silently averaged away.

The deterministic ``CoverageJudge`` exists so the harness can be tested and so a real
judge's verdicts have something to be compared against; it is not a substitute for one.
"""

from __future__ import annotations

from statistics import fmean
from typing import Literal, Protocol

from context_router.domain import Contract
from context_router.evaluation.scoring import deterministic_coverage

Verdict = Literal["a", "b", "tie"]

JUDGE_INSTRUCTIONS = (
    "You are comparing two candidate answers to the same question about a software project. "
    "You are not told how either answer was produced, and that information is deliberately "
    "withheld: judge only what the answers say. Use the requirement list as the rubric. "
    'Reply with JSON only: {"winner": "a" | "b" | "tie", "reason": "<one sentence>"}.'
)


class Judge(Protocol):
    model_version: str

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict: ...


class JudgePair(Contract):
    sample_id: str
    query: str
    requirements: list[str]
    arm_a: str
    arm_b: str
    answer_a: str
    answer_b: str


class JudgeOutcome(Contract):
    sample_id: str
    arm_a: str
    arm_b: str
    #: Resolved back onto arm_a / arm_b, never the position the judge happened to see.
    winner: Verdict
    swapped: bool
    #: None when only one order was judged; False means the two orders disagreed.
    agreement: bool | None


def build_judge_prompt(*, query: str, requirements: list[str], answer_a: str, answer_b: str) -> str:
    """The judge's entire view of the comparison.

    Deliberately takes no arm name and no token count: blinding that depends on the caller
    remembering to omit something is not blinding. ``tests/test_judge.py`` asserts that an
    arm name planted in the caller's data cannot reach the prompt.
    """

    rubric = "\n".join(f"- {requirement}" for requirement in requirements) or "- (no rubric)"
    return (
        f"[Question]\n{query}\n\n"
        f"[Requirements]\n{rubric}\n\n"
        f"[Answer A]\n{answer_a}\n\n"
        f"[Answer B]\n{answer_b}\n"
    )


class CoverageJudge:
    """Deterministic stand-in: prefers the answer that satisfies more requirements.

    Useful for exercising the swap logic and for checking that a paid judge is not simply
    reproducing a lexical heuristic. It is blind to arms by construction -- it only ever
    sees the two answer strings.
    """

    model_version = "coverage-judge-v1"

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        del query
        score_a = deterministic_coverage(requirements, answer_a)
        score_b = deterministic_coverage(requirements, answer_b)
        if score_a > score_b:
            return "a"
        if score_b > score_a:
            return "b"
        return "tie"


def _flip(verdict: Verdict) -> Verdict:
    if verdict == "a":
        return "b"
    if verdict == "b":
        return "a"
    return "tie"


def run_pairwise_judging(
    judge: Judge, pairs: list[JudgePair], *, swap: bool = True
) -> list[JudgeOutcome]:
    """Judge every pair, optionally in both orders, and resolve verdicts onto the arms."""

    outcomes: list[JudgeOutcome] = []
    for pair in pairs:
        forward = judge.compare(
            query=pair.query,
            requirements=pair.requirements,
            answer_a=pair.answer_a,
            answer_b=pair.answer_b,
        )
        if not swap:
            outcomes.append(
                JudgeOutcome(
                    sample_id=pair.sample_id,
                    arm_a=pair.arm_a,
                    arm_b=pair.arm_b,
                    winner=forward,
                    swapped=False,
                    agreement=None,
                )
            )
            continue
        # Judging the same pair with the answers exchanged: a verdict that flips with the
        # position is a verdict about the position, not about the answers.
        reversed_verdict = judge.compare(
            query=pair.query,
            requirements=pair.requirements,
            answer_a=pair.answer_b,
            answer_b=pair.answer_a,
        )
        consistent = forward == _flip(reversed_verdict)
        outcomes.append(
            JudgeOutcome(
                sample_id=pair.sample_id,
                arm_a=pair.arm_a,
                arm_b=pair.arm_b,
                winner=forward if consistent else "tie",
                swapped=True,
                agreement=consistent,
            )
        )
    return outcomes


def summarise_wins(outcomes: list[JudgeOutcome]) -> dict[str, dict[str, float]]:
    """Win / loss / tie counts per arm, plus how often the two orders agreed."""

    tally: dict[str, dict[str, float]] = {}
    for outcome in outcomes:
        for arm in (outcome.arm_a, outcome.arm_b):
            tally.setdefault(arm, {"wins": 0.0, "losses": 0.0, "ties": 0.0, "comparisons": 0.0})
        tally[outcome.arm_a]["comparisons"] += 1
        tally[outcome.arm_b]["comparisons"] += 1
        if outcome.winner == "tie":
            tally[outcome.arm_a]["ties"] += 1
            tally[outcome.arm_b]["ties"] += 1
        else:
            winner = outcome.arm_a if outcome.winner == "a" else outcome.arm_b
            loser = outcome.arm_b if outcome.winner == "a" else outcome.arm_a
            tally[winner]["wins"] += 1
            tally[loser]["losses"] += 1
    agreements = [float(o.agreement) for o in outcomes if o.agreement is not None]
    order_agreement = fmean(agreements) if agreements else 0.0
    for row in tally.values():
        row["win_rate"] = row["wins"] / row["comparisons"] if row["comparisons"] else 0.0
        row["order_agreement"] = order_agreement
    return tally
