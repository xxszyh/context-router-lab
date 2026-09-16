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

import json
import re
from statistics import fmean
from typing import Literal, Protocol, cast

from context_router.domain import Contract
from context_router.evaluation.scoring import deterministic_coverage
from context_router.providers.openai_compatible import AnswerResult

Verdict = Literal["a", "b", "tie"]

JUDGE_INSTRUCTIONS = (
    "You compare two candidate answers to the same question about a software project. You are "
    "not told how either was produced and that information is deliberately withheld: judge "
    "only what the answers say.\n\n"
    "Procedure, in this order:\n"
    "1. For each requirement, decide whether Answer A satisfies it, and whether Answer B "
    "does. Work through the requirements one at a time.\n"
    "2. An answer satisfies a requirement only if it is actually addressed. Repeating a "
    "requirement's wording while saying the answer cannot be given does NOT satisfy it.\n"
    '3. Winner: "a" if A satisfies strictly more requirements; "b" if B does; "tie" only if '
    "both satisfy the same requirements and neither is more correct.\n\n"
    "The order the answers appear in carries no information, and deciding differently when "
    "they are swapped is an error.\n\n"
    "End your reply with exactly one line and nothing after it:\n"
    "WINNER: <a|b|tie>"
)


class Judge(Protocol):
    model_version: str

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict: ...


class JudgeCall(Contract):
    """One raw model call the judge made, kept so a failure can be diagnosed for free.

    `stop_reason` is the field that would have saved two re-runs: it distinguishes a reply
    that was cut off at the token cap from one the model simply did not produce, and those
    need different fixes.
    """

    text: str
    stop_reason: str | None
    output_tokens: int


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


class VerdictModel(Protocol):
    """Anything that runs a prompt and hands back the raw reply."""

    model_version: str

    def run(self, *, prompt: str, instructions: str) -> AnswerResult: ...


_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.S)
_LABELLED = re.compile(r"\b(?:winner|verdict)\b\s*[:：]\s*[\"']?(a|b|tie)\b", re.I)
_BARE = re.compile(r"^\s*[\"']?(a|b|tie)[\"']?\s*[.!]?\s*$", re.I)


#: Fires on a reply cut off mid-JSON, where the object will not load whole but the winner
#: field is already on the wire. Anchored on the field name so it cannot fire on prose,
#: which matters because a spurious "a" is worse than an honest tie.
_TRUNCATED = re.compile(r"[\"']?winner[\"']?\s*[:：]\s*[\"']?(a|b|tie)\b", re.I)


def parse_verdict(text: str) -> Verdict | None:
    """Recover the winner from a judge reply, tolerating fences and surrounding prose.

    Returns None rather than guessing: a verdict that cannot be read is not a tie, and the
    caller counts it as a parse failure so an unparseable judge shows up as a broken
    instrument instead of as agreement.
    """

    for match in _JSON_OBJECT.finditer(text):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        winner = str(payload.get("winner", "")).strip().lower()
        if winner in ("a", "b", "tie"):
            return cast(Verdict, winner)
    # The last mention wins: the judge is told to end with its verdict, so anything
    # earlier is reasoning about the verdict rather than the verdict.
    truncated = _TRUNCATED.findall(text)
    if truncated:
        return cast(Verdict, truncated[-1].lower())
    labelled = _LABELLED.findall(text)
    if labelled:
        return cast(Verdict, labelled[-1].lower())
    bare = _BARE.match(text)
    if bare:
        return cast(Verdict, bare.group(1).lower())
    return None


class LLMJudge:
    """A blinded pairwise judge backed by a real model.

    The prompt is built by ``build_judge_prompt``, which takes no arm name and no token
    count, so this class cannot leak what it is not given.
    """

    def __init__(self, model: VerdictModel, *, retries: int = 1) -> None:
        self.model = model
        self.retries = retries
        self.model_version = f"llm-judge:{model.model_version}"
        self.parse_failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        #: Every call, kept because the first judge run lost 14 of 40 to unreadable output
        #: and there was no way to find out what they said without paying for them again.
        self.calls: list[JudgeCall] = []

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        prompt = build_judge_prompt(
            query=query, requirements=requirements, answer_a=answer_a, answer_b=answer_b
        )
        # Retried because an empty reply is transient: a judge run lost 7 of 40 to it.
        for _ in range(self.retries + 1):
            result = self.model.run(prompt=prompt, instructions=JUDGE_INSTRUCTIONS)
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens
            self.calls.append(
                JudgeCall(
                    text=result.text,
                    stop_reason=result.stop_reason,
                    output_tokens=result.output_tokens,
                )
            )
            verdict = parse_verdict(result.text)
            if verdict is not None:
                return verdict
        self.parse_failures += 1
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
