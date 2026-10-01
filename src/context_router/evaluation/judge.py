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

import hashlib
import json
import re
from collections import Counter
from statistics import fmean
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import Field

from context_router.domain import Contract
from context_router.evaluation.scoring import deterministic_coverage, is_refusal
from context_router.providers.openai_compatible import AnswerResult

Verdict = Literal["a", "b", "tie"]
DecisionSource = Literal["judge", "refusal_gate"]

JUDGE_INSTRUCTIONS = (
    "You compare two candidate answers to the same question about a software project. You are "
    "not told how either was produced and that information is deliberately withheld: judge "
    "only what the answers say.\n\n"
    "Judge quickly. This is a comparison, not an essay, and the deliberation is not wanted:\n"
    "1. Go through the requirements once. For each, note in one short line whether A "
    "satisfies it and whether B does. Do not restate either answer.\n"
    "2. An answer satisfies a requirement only if it is actually addressed. Repeating a "
    "requirement's wording while saying the answer cannot be given does NOT satisfy it.\n"
    '3. Winner: "a" if A satisfies strictly more requirements; "b" if B does; "tie" only if '
    "both satisfy the same requirements and neither is more correct.\n\n"
    "The order the answers appear in carries no information, and deciding differently when "
    "they are swapped is an error.\n\n"
    "Keep the whole reply to at most four lines, ending with exactly this and nothing after "
    "it:\n"
    "WINNER: <a|b|tie>"
)

#: The same rubric with the analysis anchored in the text. Identical blinding, identical three
#: rules, identical final line; the only change is that a requirement may not be marked
#: satisfied without a quote from the answer that decides it.
#:
#: This is aimed at a measured failure, not a guessed one. On the published conversation, all
#: thirteen pairs the judge contradicted itself on contradicted itself *within a single order*
#: -- no position bias at all -- so the instability is in how it reads the answers, not in where
#: they sit. `JUDGE_INSTRUCTIONS` forbids the one thing that would ground that reading ("Do not
#: restate either answer"), which is the cost optimisation the README records as having taken
#: order agreement down with the output tokens. Splitting that diagnosis is
#: `docs/judge-anchoring-2026-10-01.md`.
JUDGE_INSTRUCTIONS_ANCHORED = (
    "You compare two candidate answers to the same question about a software project. You are "
    "not told how either was produced and that information is deliberately withheld: judge "
    "only what the answers say.\n\n"
    "Work through the requirements one at a time, and decide each on the text rather than on an "
    "impression:\n"
    '1. For each requirement, quote the few words from A that decide it, or write "none" if '
    "nothing in A bears on it, and do the same for B. A requirement marked satisfied without a "
    "quote is a guess.\n"
    "2. An answer satisfies a requirement only if it is actually addressed. Repeating a "
    "requirement's wording while saying the answer cannot be given does NOT satisfy it.\n"
    '3. Winner: "a" if A satisfies strictly more requirements; "b" if B does; "tie" only if '
    "both satisfy the same requirements and neither is more correct.\n\n"
    "The order the answers appear in carries no information, and deciding differently when "
    "they are swapped is an error.\n\n"
    "End with exactly this and nothing after it:\n"
    "WINNER: <a|b|tie>"
)

#: Named prompts, because the prompt is a parameter of every number a judge produces and was
#: recorded nowhere until now -- the same defect `state-of-play` records for the lexical
#: signal's rendering and threshold. A style name travels with the run; the hash below makes a
#: silent edit to the text detectable after the fact.
JUDGE_STYLES: dict[str, str] = {
    "terse": JUDGE_INSTRUCTIONS,
    "anchored": JUDGE_INSTRUCTIONS_ANCHORED,
}


def judge_instructions(style: str) -> str:
    """The prompt text for a named style."""

    try:
        return JUDGE_STYLES[style]
    except KeyError:
        known = ", ".join(sorted(JUDGE_STYLES))
        raise ValueError(f"unknown judge style {style!r}; known: {known}") from None


class Judge(Protocol):
    model_version: str

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict: ...


class JudgePredecision(Contract):
    verdict: Verdict
    source: DecisionSource


@runtime_checkable
class PrejudgingJudge(Protocol):
    def prejudge(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> JudgePredecision | None: ...


class RefusalGatedJudge:
    """Skip model judging when both candidates explicitly decline to answer.

    The gate is deliberately narrow.  A single refusal is still sent to the delegate,
    because ``is_refusal`` is a lexical heuristic with measured false positives and must
    not decide which substantive answer wins.  When both sides decline, however, neither
    answer satisfies a non-refusal requirement; treating the pair as a tie is deterministic
    and avoids paying twice for a position-swapped comparison.
    """

    def __init__(self, delegate: Judge) -> None:
        self.delegate = delegate
        self.model_version = f"refusal-gated:{delegate.model_version}"
        self.gate_hits = 0

    def compare(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> Verdict:
        predecision = self.prejudge(
            query=query,
            requirements=requirements,
            answer_a=answer_a,
            answer_b=answer_b,
        )
        if predecision is not None:
            return predecision.verdict
        return self.delegate.compare(
            query=query,
            requirements=requirements,
            answer_a=answer_a,
            answer_b=answer_b,
        )

    def prejudge(
        self, *, query: str, requirements: list[str], answer_a: str, answer_b: str
    ) -> JudgePredecision | None:
        del query, requirements
        if not (is_refusal(answer_a) and is_refusal(answer_b)):
            return None
        self.gate_hits += 1
        return JudgePredecision(verdict="tie", source="refusal_gate")


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
    #: A refusal is the correct behaviour on this checkpoint. Named for the expectation rather
    #: than for `must_abstain`, which the annotation protocol sets whenever the reference reply
    #: drew on no labelled context -- including the `new_context` case where it answered anyway.
    expects_refusal: bool = False


class JudgeOutcome(Contract):
    sample_id: str
    arm_a: str
    arm_b: str
    #: Resolved back onto arm_a / arm_b, never the position the judge happened to see.
    winner: Verdict
    swapped: bool
    #: None when only one order was judged; False means the calls disagreed.
    agreement: bool | None
    #: Every call's verdict, normalised onto arm_a / arm_b. Kept so a majority can be re-derived
    #: and so a pair's instability is visible rather than summarised away.
    verdicts: list[Verdict] = Field(default_factory=list)
    #: Share of calls that landed on the winner. With one repeat this is 1.0 or 0.5, and the
    #: binary `agreement` above is the same information at a coarser grain; with more repeats it
    #: is the only place the degree of agreement survives.
    consistency: float | None = None
    decision_source: DecisionSource = "judge"


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

    def __init__(
        self, model: VerdictModel, *, retries: int = 1, instructions: str = JUDGE_INSTRUCTIONS
    ) -> None:
        self.model = model
        self.retries = retries
        self.instructions = instructions
        #: A digest of the prompt, so an artefact can say which instrument produced it even
        #: after the constant is edited. Every judge number in this project before today was
        #: produced by a prompt nobody recorded.
        self.instructions_sha256 = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
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
            result = self.model.run(prompt=prompt, instructions=self.instructions)
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
    judge: Judge, pairs: list[JudgePair], *, swap: bool = True, repeats: int = 1
) -> list[JudgeOutcome]:
    """Judge every pair, optionally in both orders and more than once, and resolve onto the arms.

    ``repeats`` is the instrument's stability knob. Measured across this project's judge runs, 42%
    of the ties are not the judge finding two answers equal -- they are the two orders
    contradicting each other and the protocol collapsing that into a tie. Those are noise, and
    asking the same question again is what removes noise. The 52% that are genuine ties stay
    ties however many times they are asked, which is the right behaviour: repetition should not
    manufacture a decision the judge does not have.

    Each repeat contributes both orders, so the calls per pair are ``2 * repeats`` and a strict
    majority is required. No majority means a tie, and the verdicts are kept on the outcome so
    the split is visible rather than flattened.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    outcomes: list[JudgeOutcome] = []
    for pair in pairs:
        predecision = (
            judge.prejudge(
                query=pair.query,
                requirements=pair.requirements,
                answer_a=pair.answer_a,
                answer_b=pair.answer_b,
            )
            if isinstance(judge, PrejudgingJudge)
            else None
        )
        if predecision is not None:
            outcomes.append(
                JudgeOutcome(
                    sample_id=pair.sample_id,
                    arm_a=pair.arm_a,
                    arm_b=pair.arm_b,
                    winner=predecision.verdict,
                    swapped=False,
                    agreement=None,
                    decision_source=predecision.source,
                )
            )
            continue
        verdicts: list[Verdict] = []
        for _ in range(repeats):
            verdicts.append(
                judge.compare(
                    query=pair.query,
                    requirements=pair.requirements,
                    answer_a=pair.answer_a,
                    answer_b=pair.answer_b,
                )
            )
            if not swap:
                continue
            # Judging the same pair with the answers exchanged: a verdict that flips with the
            # position is a verdict about the position, not about the answers. Normalised back
            # onto arm_a before counting, so every entry means the same thing.
            verdicts.append(
                _flip(
                    judge.compare(
                        query=pair.query,
                        requirements=pair.requirements,
                        answer_a=pair.answer_b,
                        answer_b=pair.answer_a,
                    )
                )
            )

        counts = Counter(verdicts)
        winner, most = counts.most_common(1)[0]
        # A strict majority, not a plurality: with an even number of calls a 3-3 split is a tie
        # and must not be decided by which verdict `most_common` happened to return first.
        decided = most * 2 > len(verdicts)
        outcomes.append(
            JudgeOutcome(
                sample_id=pair.sample_id,
                arm_a=pair.arm_a,
                arm_b=pair.arm_b,
                winner=winner if decided else "tie",
                swapped=swap,
                # About the *orders*, so it is unknown when only one order was judged -- even
                # though the repeats would still give a consistency figure.
                agreement=(len(counts) == 1) if swap else None,
                verdicts=verdicts,
                consistency=most / len(verdicts),
            )
        )
    return outcomes


def summarise_wins(outcomes: list[JudgeOutcome]) -> dict[str, dict[str, float | None]]:
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
    order_agreement = fmean(agreements) if agreements else None
    for row in tally.values():
        row["win_rate"] = row["wins"] / row["comparisons"] if row["comparisons"] else 0.0
    return {arm: {**row, "order_agreement": order_agreement} for arm, row in tally.items()}


def _summarise_outcome_group(outcomes: list[JudgeOutcome]) -> dict[str, int | float | None]:
    known_agreement = [outcome for outcome in outcomes if outcome.agreement is not None]
    agreed = sum(outcome.agreement is True for outcome in known_agreement)
    return {
        "pairs": len(outcomes),
        "judged_pairs": len(known_agreement),
        "agreed": agreed,
        "disagreed": sum(outcome.agreement is False for outcome in known_agreement),
        "order_agreement": agreed / len(known_agreement) if known_agreement else None,
        "arm_a_wins": sum(outcome.winner == "a" for outcome in outcomes),
        "arm_b_wins": sum(outcome.winner == "b" for outcome in outcomes),
        "ties": sum(outcome.winner == "tie" for outcome in outcomes),
    }


def summarise_judge_strata(
    pairs: list[JudgePair], outcomes: list[JudgeOutcome]
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    """Separate benchmark intent from the answers' observed refusal behaviour.

    ``expects_refusal`` identifies checkpoints where declining is the correct behaviour, which
    is the annotation protocol's rule 1 and nothing else. It used to read ``must_abstain``, which
    the protocol sets for rule 2 as well -- the `new_context` case where the reference reply drew
    on no labelled context but answered anyway. Under that reading two checkpoints whose
    requirements ask for a substantive answer were filed as failed refusals. The response strata
    independently show whether both, one or neither candidate actually declined; reporting both
    prevents a refusal-heavy tie bucket from hiding the judge's behaviour on answerable
    checkpoints.
    """

    if len(pairs) != len(outcomes):
        raise ValueError("pairs and outcomes must have the same length")
    checkpoint: dict[str, list[JudgeOutcome]] = {"answerable": [], "must_refuse": []}
    response: dict[str, list[JudgeOutcome]] = {
        "neither_refuses": [],
        "one_refuses": [],
        "both_refuse": [],
    }
    for pair, outcome in zip(pairs, outcomes, strict=True):
        if pair.sample_id != outcome.sample_id:
            raise ValueError("pairs and outcomes must have matching sample ids")
        checkpoint["must_refuse" if pair.expects_refusal else "answerable"].append(outcome)
        refusal_count = int(is_refusal(pair.answer_a)) + int(is_refusal(pair.answer_b))
        response_key = ("neither_refuses", "one_refuses", "both_refuse")[refusal_count]
        response[response_key].append(outcome)
    return {
        "checkpoint": {name: _summarise_outcome_group(group) for name, group in checkpoint.items()},
        "response": {name: _summarise_outcome_group(group) for name, group in response.items()},
    }
