"""Necessity by ablation: does the query survive removing one context?

`required_context_ids` means "without this context the query cannot be answered", which is a
counterfactual and cannot be read off any similarity score. Three lexical signals were tried
and all three failed, in opposite directions, because "the reply mentions X" is neither
necessary nor sufficient for "X is required".

So it is asked directly instead. For each candidate context the model is shown the recent
window plus the material of every other context, and asked whether the query can still be
answered. A "no" makes that context required. Contexts that are plainly irrelevant act as a
control: their ablations should come back "yes", and a "no" there is a signal that the judge
is not reading the material.

Two choices decide whether the answer means anything.

**The recent window is always present.** It is always available in the real system, so
"required" has to mean "needed in addition to the window". That is also the semantics the
synthetic benchmark uses, where evidence is placed deliberately beyond the window.

**Material is retrieved evidence, not the context description.** Descriptions are generic
prose; ablating one tests whether a summary is enough, which is not the question. Each
context contributes the events its own retrieval would surface for this query, which is what
the router would actually have shown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from context_router.assembly.builder import TokenCounter
from context_router.domain import Contract, RawEvent
from context_router.providers.openai_compatible import AnswerResult
from context_router.retrieval import BM25Index, LexicalAnalyzer

NecessityModel = Literal["ablation"]

DEFAULT_INSTRUCTIONS = (
    "You decide whether a question can be answered from a given body of material. "
    "Be strict: the question counts as answerable only if the material contains what is "
    "needed to answer it. Material that is merely related, or that names the same files "
    "without stating the answer, does not make the question answerable. "
    "Reply with one word: YES or NO."
)

#: Events each context contributes to an ablation.
EVIDENCE_PER_CONTEXT = 4
#: Characters kept per event. Real assistant turns run to thousands of characters, and a
#: dry run of the first batch measured prompts from 3 181 to 36 032 tokens, which is both
#: expensive and likely to degrade the judgement the ablation depends on.
EVENT_EXCERPT = 400


class AblationVerdict(Contract):
    """One ablated context and what the model said about the material left behind."""

    sample_id: str
    ablated_context_id: str
    answerable_without: bool
    raw_reply: str
    #: True when the ablation was expected to be trivially answerable, so a "no" is a warning.
    is_control: bool


def context_material(
    events: list[RawEvent],
    query: str,
    *,
    analyzer: LexicalAnalyzer | None = None,
    limit: int = EVIDENCE_PER_CONTEXT,
    excerpt: int = EVENT_EXCERPT,
) -> str:
    """The events this context's own retrieval would surface for the query.

    Each event is truncated to `excerpt` characters so one long assistant reply cannot
    dominate the material and so the ablation stays affordable.
    """

    if not events:
        return ""
    active = analyzer or LexicalAnalyzer()
    documents = {event.event_id: event.content for event in events}
    ranked = BM25Index(documents, analyzer=active).rank(query, limit)
    if not ranked:
        # No lexical overlap at all: still show the most recent turns of the context, because
        # a context can be required for continuity rather than for a term match.
        chosen = sorted(events, key=lambda event: event.sequence)[-limit:]
    else:
        wanted = {event_id for event_id, _ in ranked}
        chosen = sorted(
            (event for event in events if event.event_id in wanted),
            key=lambda event: event.sequence,
        )
    return "\n".join(f"[{event.event_id}] {event.content[:excerpt]}" for event in chosen)


def build_ablation_prompt(
    *,
    query: str,
    recent_window: str,
    materials: dict[str, str],
    context_names: dict[str, str],
) -> str:
    """The model's whole view: the window, every context's material, then the question."""

    blocks = [f"[Recent conversation]\n{recent_window or '(none)'}"]
    for context_id, material in materials.items():
        label = context_names.get(context_id, context_id)
        blocks.append(f"[Context: {label}]\n{material or '(no material retrieved)'}")
    blocks.append(f"[Question]\n{query}")
    return "\n\n".join(blocks)


def parse_verdict(text: str) -> bool | None:
    """Yes means answerable without the ablated context, so the context is not required.

    Returns None when the reply is unreadable, which the caller counts rather than guessing:
    an unreadable judge is a broken instrument, not a "no".
    """

    upper = text.strip().upper()
    if not upper:
        return None
    if upper.startswith("NO") or "\nNO" in upper[:40]:
        return False
    if upper.startswith("YES") or "\nYES" in upper[:40]:
        return True
    return None


class AnswerableJudge(Protocol):
    def answer(self, *, query: str, working_context: str, instructions: str) -> AnswerResult: ...


@dataclass
class AblationRun:
    """Verdicts plus the accounting needed to tell a result from a broken judge."""

    verdicts: list[AblationVerdict]
    unreadable: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: Text of replies that could not be read. Without it a run of 117 unreadable replies
    #: reports a number and no way to find out why, which is what happened.
    unreadable_replies: list[str] = field(default_factory=list)


def run_ablations(
    *,
    sample_id: str,
    query: str,
    recent_window: str,
    materials: dict[str, str],
    context_names: dict[str, str],
    control_context_ids: set[str],
    judge: AnswerableJudge,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> AblationRun:
    """Ablate each context in turn and record whether the query survived."""

    run = AblationRun(verdicts=[])
    for context_id in materials:
        remaining = {k: v for k, v in materials.items() if k != context_id}
        prompt = build_ablation_prompt(
            query=query,
            recent_window=recent_window,
            materials=remaining,
            context_names=context_names,
        )
        result = judge.answer(query=query, working_context=prompt, instructions=instructions)
        run.input_tokens += result.input_tokens
        run.output_tokens += result.output_tokens
        answerable = parse_verdict(result.text)
        if answerable is None:
            run.unreadable += 1
            run.unreadable_replies.append(result.text[:200])
            continue
        run.verdicts.append(
            AblationVerdict(
                sample_id=sample_id,
                ablated_context_id=context_id,
                answerable_without=answerable,
                raw_reply=result.text.strip()[:80],
                is_control=context_id in control_context_ids,
            )
        )
    return run


def required_from_ablations(verdicts: list[AblationVerdict]) -> list[str]:
    """A context is required when removing it made the query unanswerable."""

    return sorted(v.ablated_context_id for v in verdicts if not v.answerable_without)


def control_failures(verdicts: list[AblationVerdict]) -> list[str]:
    """Controls that came back "not answerable". A judge that fails these is not reading."""

    return sorted(
        v.ablated_context_id for v in verdicts if v.is_control and not v.answerable_without
    )


def evidence_tokens(materials: dict[str, str], *, counter: TokenCounter | None = None) -> int:
    """Size of the material shown, so a run's cost can be compared with the router's budget."""

    tokens = counter or TokenCounter()
    return sum(tokens.count(text) for text in materials.values())
