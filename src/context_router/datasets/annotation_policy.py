"""NOT IN USE -- kept because its failure is the finding. See docs/annotation-protocol.md.

The real-replay query-type decision procedure, as code rather than as guidance.

`docs/annotation-protocol.md` states the rules; this module applies them so a label is
reproducible instead of a matter of reading. The first pass assigned types by judgement and
used `cross_context` for 7 of 26 checkpoints, most of which a re-derivation could not confirm.
Applying it changed 23 of 26 checkpoints and produced plainly wrong labels, because every
rule that matters reads `reply_contexts` and that quantity is a counterfactual, not a
similarity: "the reply mentions X" is neither necessary nor sufficient for "the query
cannot be answered without X". Three different lexical signals were tried and all three
failed, in opposite directions. Do not reuse this module's `reply_contexts`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from context_router.domain import QueryType, RawEvent
from context_router.evaluation.scoring import is_refusal
from context_router.retrieval import BM25Index, LexicalAnalyzer

#: Words that name a task. A query containing one carries its own referent; a query made only
#: of a demonstrative does not, and that is the whole content of `short_coreference`.
TASK_NOUN = re.compile(
    r"\.py|\.vsdx|\.csv|\.xlsx|\.pdf|题|问|图|表|附录|代码|模型|算法|文件|材料|网格|方案"
)


@dataclass(frozen=True)
class ContextSet:
    """Which context owns each sequence, and each context's searchable description."""

    member_of: dict[int, str]
    descriptor_of: dict[str, str]

    def previous_active(self, seq: int) -> str | None:
        earlier = [s for s in self.member_of if s < seq]
        return self.member_of[max(earlier)] if earlier else None

    def active_before(self, seq: int) -> set[str]:
        return {self.member_of[s] for s in self.member_of if s < seq}


def reply_contexts(
    seq: int,
    events: list[RawEvent],
    contexts: ContextSet,
    *,
    analyzer: LexicalAnalyzer | None = None,
) -> list[str]:
    """Contexts the assistant's next reply materially draws on.

    Scores each context description against the reply and keeps those within half of the best.
    An earlier version counted raw entity overlap and filtered entities to three characters or
    more, which stripped `dr`, `cn`, `50` and `0.05` out of their own contexts and left the
    signal empty for several of them.
    """

    reply = ""
    for event in events:
        if event.sequence <= seq:
            continue
        if event.actor == "user" and event.kind == "message":
            break
        if event.actor == "assistant":
            reply += event.content
    if not reply.strip() or not contexts.descriptor_of:
        return []

    index = BM25Index(contexts.descriptor_of, analyzer=analyzer or LexicalAnalyzer())
    scores = index.scores(reply)
    best = max(scores.values(), default=0.0)
    if best <= 0.0:
        return []
    threshold = best * 0.5
    return sorted(context_id for context_id, score in scores.items() if score >= threshold)


def decide_type(
    seq: int,
    query: str,
    reply_context_ids: list[str],
    reply_text: str,
    contexts: ContextSet,
) -> tuple[QueryType, list[str], bool]:
    """Return the label, the required contexts, and whether routing must abstain."""

    if not reply_context_ids:
        # Rule 1 versus 2: a reply that declines means the material exists but does not contain
        # the answer; one that does not decline is opening a topic instead.
        if is_refusal(reply_text):
            return "unanswerable", [], True
        return "new_context", [], True

    if len(reply_context_ids) >= 2:
        return "cross_context", list(reply_context_ids), False

    target = reply_context_ids[0]
    previous = contexts.previous_active(seq)
    if target == previous:
        if not TASK_NOUN.search(query):
            return "short_coreference", [target], False
        return "continue", [target], False
    if target in contexts.active_before(seq):
        return "return", [target], False
    return "switch", [target], False
