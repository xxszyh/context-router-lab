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
#: the benchmark uses.
REFUSAL_CUES = (
    "无法回答",
    "不足以",
    "没有提到",
    "未提及",
    "未讨论",
    "没有相关",
    "无法确定",
    "上下文不足",
    "cannot answer",
    "not enough information",
    "no information",
    "does not contain",
    "insufficient",
)


def is_refusal_requirement(requirement: str) -> bool:
    return requirement.startswith(REFUSAL_PREFIX)


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
