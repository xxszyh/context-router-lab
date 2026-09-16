"""Deterministic answer scoring from ``answer_requirements``.

This is a *shallow* check by construction: it verifies that an answer names the identifier
and the sub-topic the labelled episode turns on, which is precisely what distinguishes that
episode from its siblings. It cannot tell a correct explanation from an answer that merely
lists the right nouns, and it must never be reported as answer quality on its own. Its job
is to be cheap, deterministic and label-independent of any model, so the pipeline can be
tested and so a model's output can be sanity-checked before a judge is paid for.
"""

from __future__ import annotations

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")
_QUOTED = re.compile(r"「([^」]+)」")

#: Requirement prefix marking a checkpoint whose correct behaviour is a refusal.
REFUSAL_PREFIX = "说明上下文不足"

#: Cues that count as declining to answer rather than fabricating one, in both languages
#: the benchmark uses. These are the phrasings the first real run actually produced, which
#: is why they include stance verbs like 没有说明 and 未给出 rather than only the obvious
#: 无法回答: the model tended to state that the context *lacks* something and then reason
#: from that, rather than announcing that it refuses.
#:
#: This is a heuristic and it errs in both directions -- it will flag a real answer that
#: says "无法确定 X，但 Y 是..." as a decline, and miss a decline phrased some way not
#: listed here. That is the measured reason a blinded judge is still required, and why
#: callers should report coverage and declining side by side rather than collapsing them.
REFUSAL_CUES = (
    "无法回答",
    "无法确定",
    "无法判断",
    "无法据此",
    "不能确定",
    "不足以",
    "没有说明",
    "没有提到",
    "没有相关",
    "没有关于",
    "未提及",
    "未讨论",
    "未给出",
    "不包含",
    "上下文不足",
    "cannot answer",
    "cannot determine",
    "not enough information",
    "no information",
    "does not contain",
    "does not mention",
    "unable to",
    "insufficient",
)


def is_refusal_requirement(requirement: str) -> bool:
    return requirement.startswith(REFUSAL_PREFIX)


def is_refusal(answer: str) -> bool:
    """Whether the answer declines to answer.

    Needed because the lexical check alone is fooled by an answer that paraphrases the
    requirement's own terms while explaining that it cannot meet them. That pattern scored
    full marks in the first real run, on one answer in five, and it flattered whichever arm
    quoted the requirement back most fluently. Callers can therefore report coverage and
    declining separately and let a reader see the difference rather than trusting one
    number.
    """

    lowered = answer.lower()
    return any(cue.lower() in lowered for cue in REFUSAL_CUES)


def requirement_terms(requirement: str) -> list[str]:
    """The checkable terms of a requirement: ASCII identifiers plus quoted sub-topics."""

    return [*_IDENTIFIER.findall(requirement), *_QUOTED.findall(requirement)]


def requirement_satisfied(requirement: str, answer: str) -> bool:
    if is_refusal_requirement(requirement):
        lowered = answer.lower()
        return any(cue.lower() in lowered for cue in REFUSAL_CUES)
    terms = requirement_terms(requirement)
    if not terms:
        return False
    lowered = answer.lower()
    return all(term.lower() in lowered for term in terms)


def deterministic_coverage(requirements: list[str], answer: str) -> float:
    """Fraction of requirements the answer satisfies. 1.0 when there are none."""

    if not requirements:
        return 1.0
    satisfied = sum(1 for requirement in requirements if requirement_satisfied(requirement, answer))
    return satisfied / len(requirements)
