from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from context_router.domain import (
    AssemblyRequest,
    EventContextAssignment,
    FlatContext,
    RawEvent,
    RouteDecision,
    SourceSpan,
    WorkingContext,
)
from context_router.providers.embedding import HashEmbeddingProvider, cosine
from context_router.retrieval import BM25Index, LexicalAnalyzer

_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[A-Za-z_][A-Za-z0-9_./\\:-]*|\d+(?:\.\d+)*|[^\s]"
)

#: How evidence events are ranked inside a selected context, named so the profile can be
#: swept rather than guessed at. Moving weight off `dense` and onto `lexical` was measured
#: and made no difference on the current benchmark, so these are the untuned defaults, not
#: a settled choice: the dense channel is only as semantic as its embedding provider, and
#: the default provider is a hashing placeholder.
EVIDENCE_WEIGHTS = {"lexical": 0.45, "dense": 0.40, "relevance": 0.15}

#: Minimal-sufficiency controls. Admitting evidence groups until the token budget runs out
#: spends most of the context on low-relevance turns, which is the pollution selective
#: assembly exists to avoid: `depth` caps how many groups each selected context may
#: contribute, and `floor` drops groups scoring below that fraction of their context's best
#: group. Only the system's own scores are used, so this is a relevance floor rather than a
#: label-fitted cutoff.
#:
#: `floor=0.5` was chosen by sweep on sessions 0-29 and holds evidence recall at 1.000
#: while cutting router memory tokens from 821 to 626 per checkpoint (-24%). `depth` is
#: left uncapped because the sweep showed it trades recall away for less than the floor
#: buys (depth=4 reached 624 tokens but dropped recall to 0.970, where floor=0.5 kept
#: 1.000); it stays available as a knob for the budget sweep.
EVIDENCE_MAX_GROUPS_PER_CONTEXT: int | None = None
EVIDENCE_SCORE_FLOOR = 0.5

#: How many characters of an event's body survive into the indexed form. The index keeps every
#: event's identity -- section, context, sequence, actor and kind -- and truncates only the
#: prose, which is the trade the form makes: the model can see *that* a turn exists and what it
#: was about, and pays for the head rather than the whole.
INDEX_HEAD_CHARS = 160

#: `prose` is the original rendering, one block per event with the body intact. `index` renders
#: the same selection as one compact row per block. Both are selections of the same events in
#: the same order -- the difference is the form, which is the variable the `indexed_router` arm
#: exists to isolate.
RenderMode = Literal["prose", "index"]


class TokenCounter:
    """Deterministic model-neutral token estimate used for comparable experiments."""

    model_version = "regex-token-counter-v1"

    def count(self, text: str) -> int:
        return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True)
class _Block:
    text: str
    section: str
    context_id: str | None = None
    event: RawEvent | None = None


@dataclass(frozen=True)
class _EventGroup:
    events: tuple[RawEvent, ...]
    context_id: str
    score: float


def _interleave_by_context(groups: list[_EventGroup]) -> list[_EventGroup]:
    """Round-robin evidence groups across contexts, each context's best group first.

    Admitting groups in global score order lets one context's long tail exhaust the
    budget before another selected context contributes anything, so a *correct*
    multi-context routing decision can still lose the answer it routed for. Covering
    every selected context before deepening any one of them is what "minimal sufficient"
    has to mean once more than one context is in play.
    """

    buckets: dict[str, list[_EventGroup]] = {}
    for group in groups:
        buckets.setdefault(group.context_id, []).append(group)
    ordered: list[_EventGroup] = []
    depth = max((len(bucket) for bucket in buckets.values()), default=0)
    for rank in range(depth):
        for bucket in buckets.values():
            if rank < len(bucket):
                ordered.append(bucket[rank])
    return ordered


class ContextBuilder:
    def __init__(
        self,
        *,
        analyzer: LexicalAnalyzer | None = None,
        token_counter: TokenCounter | None = None,
        render_mode: RenderMode = "prose",
        index_head_chars: int = INDEX_HEAD_CHARS,
    ) -> None:
        self.analyzer = analyzer or LexicalAnalyzer()
        self.token_counter = token_counter or TokenCounter()
        self.embedding = HashEmbeddingProvider(analyzer=self.analyzer)
        self.render_mode: RenderMode = render_mode
        #: Sweepable, because how much of a body survives is the parameter that decides what the
        #: indexed form can still support -- including whether the model can tell that its
        #: grounds are insufficient.
        self.index_head_chars = index_head_chars

    def assemble(self, request: AssemblyRequest, decision: RouteDecision) -> WorkingContext:
        selected = list(dict.fromkeys(decision.selected_context_ids))
        contexts = {
            context.context_id: context
            for context in request.context_catalog
            if context.context_id in selected
        }
        causal_events = {
            event.event_id: event
            for event in request.event_pool
            if event.sequence <= request.as_of_sequence
        }
        future_events = [
            event for event in request.event_pool if event.sequence > request.as_of_sequence
        ]
        latest_assignments = self._latest_assignments(request.assignments)
        context_by_event: dict[str, set[str]] = defaultdict(set)
        relevance_by_pair: dict[tuple[str, str], float] = {}
        for assignment in latest_assignments:
            if assignment.event_id not in causal_events or assignment.context_id not in contexts:
                continue
            context_by_event[assignment.event_id].add(assignment.context_id)
            relevance_by_pair[(assignment.event_id, assignment.context_id)] = assignment.relevance

        descriptor_blocks = self._descriptor_blocks(selected, contexts)
        recent_events = [
            event for event in request.recent_events if event.sequence <= request.as_of_sequence
        ][-3:]
        recent_blocks = [self._event_block(event, "recent", None) for event in recent_events]
        recent_ids = {event.event_id for event in recent_events}
        evidence_groups = self._evidence_groups(
            query=request.query,
            selected=selected,
            causal_events=causal_events,
            context_by_event=context_by_event,
            relevance_by_pair=relevance_by_pair,
            excluded_event_ids=recent_ids,
        )

        selected_blocks: list[_Block] = []
        dropped: list[dict[str, object]] = [
            {"event_id": event.event_id, "reason": "future_event"} for event in future_events
        ]
        remaining = request.token_budget

        for block in descriptor_blocks:
            cost = self.token_counter.count(block.text)
            if cost <= remaining:
                selected_blocks.append(block)
                remaining -= cost
            else:
                dropped.append({"context_id": block.context_id, "reason": "token_budget"})

        recent_target = max(math.floor(request.token_budget * 0.15), 1)
        recent_used = 0
        chosen_recent: list[_Block] = []
        for block in reversed(recent_blocks):
            cost = self.token_counter.count(block.text)
            if cost + recent_used <= recent_target and cost <= remaining:
                chosen_recent.append(block)
                recent_used += cost
                remaining -= cost
            else:
                dropped.append(
                    {
                        "event_id": block.event.event_id if block.event else None,
                        "reason": "recent_budget",
                    }
                )
        selected_blocks.extend(reversed(chosen_recent))

        already_selected = {
            block.event.event_id for block in selected_blocks if block.event is not None
        }
        best_by_context: dict[str, float] = {}
        for group in evidence_groups:
            best_by_context[group.context_id] = max(
                best_by_context.get(group.context_id, 0.0), group.score
            )
        admitted_by_context: Counter[str] = Counter()
        for group in _interleave_by_context(evidence_groups):
            group_blocks = [
                self._event_block(event, "evidence", group.context_id)
                for event in group.events
                if event.event_id not in already_selected
            ]
            cost = sum(self.token_counter.count(block.text) for block in group_blocks)
            if (
                EVIDENCE_MAX_GROUPS_PER_CONTEXT is not None
                and admitted_by_context[group.context_id] >= EVIDENCE_MAX_GROUPS_PER_CONTEXT
            ):
                reason = "minimal_sufficiency_depth"
            elif group.score < EVIDENCE_SCORE_FLOOR * best_by_context[group.context_id]:
                reason = "minimal_sufficiency_floor"
            else:
                reason = None
            if reason is not None:
                for event in group.events:
                    if event.event_id not in already_selected:
                        dropped.append({"event_id": event.event_id, "reason": reason})
                continue
            if group_blocks and cost <= remaining:
                selected_blocks.extend(group_blocks)
                remaining -= cost
                admitted_by_context[group.context_id] += 1
                already_selected.update(
                    block.event.event_id for block in group_blocks if block.event is not None
                )
            else:
                for event in group.events:
                    if event.event_id not in already_selected:
                        dropped.append({"event_id": event.event_id, "reason": "token_budget"})

        # Separators may add tokens not charged while selecting. Remove whole trailing blocks
        # until the rendered text is safe.
        #
        # The fit is measured on the *prose* cost even when the output is indexed, and that is
        # deliberate rather than an oversight: this loop decides which events are admitted, so
        # fitting it on the rendered form would make the selection depend on the rendering. The
        # indexed form is cheaper per event, so it would admit more of them, and the two arms
        # would then differ in content as well as in form -- which is exactly the confound the
        # `indexed_router` arm exists to avoid. Fitting on a fixed cost holds the selection
        # identical and leaves the rendering as the only variable. A router that spent the freed
        # budget on more events is a different, also interesting arm; it is not this one.
        prose_cost = self.token_counter.count(self._prose_text(selected_blocks))
        while prose_cost > request.token_budget and selected_blocks:
            removed = selected_blocks.pop()
            if removed.event:
                dropped.append({"event_id": removed.event.event_id, "reason": "separator_budget"})
            prose_cost = self.token_counter.count(self._prose_text(selected_blocks))

        rendered, spans = self._render(selected_blocks)
        memory_tokens = self.token_counter.count(rendered)

        included_event_ids = [
            block.event.event_id for block in selected_blocks if block.event is not None
        ]
        total_input_tokens = memory_tokens + self.token_counter.count(request.query)
        total_input_tokens += self.token_counter.count(request.system_rules)
        return WorkingContext(
            rendered_text=rendered,
            selected_context_ids=[key for key in selected if key in contexts],
            included_event_ids=included_event_ids,
            source_spans=spans,
            memory_tokens=memory_tokens,
            total_input_tokens=total_input_tokens,
            dropped_candidates=dropped,
            causal_cutoff=request.as_of_sequence,
            routing_trace_id=decision.trace_id,
        )

    @staticmethod
    def _latest_assignments(
        assignments: list[EventContextAssignment],
    ) -> list[EventContextAssignment]:
        latest: dict[tuple[str, str, str], EventContextAssignment] = {}
        for assignment in assignments:
            key = (assignment.event_id, assignment.context_id, assignment.source)
            if key not in latest or assignment.annotation_version > latest[key].annotation_version:
                latest[key] = assignment
        return list(latest.values())

    @staticmethod
    def _descriptor_blocks(selected: list[str], contexts: dict[str, FlatContext]) -> list[_Block]:
        blocks: list[_Block] = []
        for context_id in selected:
            context = contexts.get(context_id)
            if context is None:
                continue
            text = (
                f"[Context: {context.context_id} | {context.name}]\n"
                f"Goal: {context.goal}\nSummary cue: {context.summary}"
            )
            blocks.append(_Block(text=text, section="context", context_id=context_id))
        return blocks

    @staticmethod
    def _event_block(event: RawEvent, section: str, context_id: str | None) -> _Block:
        prefix = "Recent" if section == "recent" else f"Evidence:{context_id}"
        text = f"[{prefix}][{event.event_id}][{event.actor}/{event.kind}] {event.content}"
        return _Block(text=text, section=section, context_id=context_id, event=event)

    def _evidence_groups(
        self,
        *,
        query: str,
        selected: list[str],
        causal_events: dict[str, RawEvent],
        context_by_event: dict[str, set[str]],
        relevance_by_pair: dict[tuple[str, str], float],
        excluded_event_ids: set[str],
    ) -> list[_EventGroup]:
        scores: dict[tuple[str, str], float] = {}
        for context_id in selected:
            event_ids = [
                event_id
                for event_id, context_ids in context_by_event.items()
                if context_id in context_ids and event_id not in excluded_event_ids
            ]
            if not event_ids:
                continue
            documents = {event_id: causal_events[event_id].content for event_id in event_ids}
            bm25 = BM25Index(documents, analyzer=self.analyzer).scores(query)
            query_vector = self.embedding.embed([query])[0]
            event_vectors = self.embedding.embed([documents[event_id] for event_id in event_ids])
            dense = {
                event_id: max(0.0, cosine(query_vector, vector))
                for event_id, vector in zip(event_ids, event_vectors, strict=True)
            }
            max_bm25 = max(bm25.values(), default=1.0) or 1.0
            max_dense = max(dense.values(), default=1.0) or 1.0
            weights = EVIDENCE_WEIGHTS
            for event_id in event_ids:
                scores[(event_id, context_id)] = (
                    weights["lexical"] * bm25[event_id] / max_bm25
                    + weights["dense"] * dense[event_id] / max_dense
                    + weights["relevance"] * relevance_by_pair.get((event_id, context_id), 0.0)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        event_to_children: dict[str, list[RawEvent]] = defaultdict(list)
        for event in causal_events.values():
            if event.parent_event_id:
                event_to_children[event.parent_event_id].append(event)

        groups: list[_EventGroup] = []
        seen: set[tuple[str, str]] = set()
        for (event_id, context_id), score in ranked:
            if (event_id, context_id) in seen:
                continue
            anchor = causal_events[event_id]
            members: dict[str, RawEvent] = {anchor.event_id: anchor}
            if anchor.kind == "tool_call":
                members.update(
                    {child.event_id: child for child in event_to_children[anchor.event_id]}
                )
            elif anchor.kind == "tool_result" and anchor.parent_event_id in causal_events:
                parent = causal_events[anchor.parent_event_id]
                members[parent.event_id] = parent
                members.update(
                    {child.event_id: child for child in event_to_children[parent.event_id]}
                )

            for neighbor in causal_events.values():
                if abs(
                    neighbor.sequence - anchor.sequence
                ) == 1 and context_id in context_by_event.get(neighbor.event_id, set()):
                    members[neighbor.event_id] = neighbor
            ordered = tuple(sorted(members.values(), key=lambda event: event.sequence))
            for member in ordered:
                seen.add((member.event_id, context_id))
            groups.append(_EventGroup(events=ordered, context_id=context_id, score=score))
        return groups

    def _render(self, blocks: list[_Block]) -> tuple[str, list[SourceSpan]]:
        """Render the selected blocks in the builder's configured form.

        Both forms take the same list in the same order and emit one row per block, so the
        source spans line up and the two are interchangeable downstream. Only the text differs.
        """

        rows = [self._row(index, block) for index, block in enumerate(blocks)]
        separator = "\n\n" if self.render_mode == "prose" else "\n"

        parts: list[str] = []
        spans: list[SourceSpan] = []
        cursor = 0
        for index, (block, row) in enumerate(zip(blocks, rows, strict=True)):
            if index:
                parts.append(separator)
                cursor += len(separator)
            start = cursor
            parts.append(row)
            cursor += len(row)
            spans.append(
                SourceSpan(
                    event_id=block.event.event_id if block.event else None,
                    context_id=block.context_id,
                    start_char=start,
                    end_char=cursor,
                    section=block.section,  # type: ignore[arg-type]
                )
            )
        return "".join(parts), spans

    @staticmethod
    def _prose_text(blocks: list[_Block]) -> str:
        """What these blocks would cost in the original form, whatever mode is configured.

        Only the budget fit uses this, so that the admitted set is a property of the selection
        and not of the rendering.
        """

        return "\n\n".join(block.text for block in blocks)

    def _row(self, index: int, block: _Block) -> str:
        if self.render_mode == "prose":
            return block.text

        candidate = self._index_row(index, block)
        # The indexed form is a compression, so it must never expand. A short turn costs less to
        # keep than to re-render with its metadata, and a row that grew would break two things at
        # once: the arm would exceed the budget its prose twin respected, and -- because the
        # budget fit measures the prose cost -- the admitted set would stop matching. Falling
        # back keeps both invariants true by construction rather than by hoping the data is long.
        if self.token_counter.count(candidate) > self.token_counter.count(block.text):
            return block.text
        return candidate

    def _index_row(self, index: int, block: _Block) -> str:
        """One block as a single compact row: identity kept, body truncated to a head."""

        if block.event is None:
            # A context descriptor is already short, and its goal and summary are the whole
            # reason it is in the selection -- so the row drops the three field labels and the
            # newlines rather than any of the content.
            lines = block.text.splitlines()
            head = lines[0].removeprefix("[Context: ").removesuffix("]") if lines else ""
            goal = lines[1].removeprefix("Goal: ") if len(lines) > 1 else ""
            cue = lines[2].removeprefix("Summary cue: ") if len(lines) > 2 else ""
            return f"[{index}] context {head} | goal: {goal} | cue: {cue}"

        event = block.event
        head = " ".join(event.content.split())
        if len(head) > self.index_head_chars:
            head = head[: self.index_head_chars] + "…"
        return (
            f"[{index}] {block.section} ctx={block.context_id} seq={event.sequence} "
            f"{event.actor}/{event.kind}: {head}"
        )


_DEFAULT_BUILDER = ContextBuilder()
_INDEXED_BUILDER = ContextBuilder(render_mode="index")
#: Builders are cached by head length so a sweep reuses them instead of rebuilding per call.
_INDEXED_BUILDERS: dict[int, ContextBuilder] = {INDEX_HEAD_CHARS: _INDEXED_BUILDER}


def assemble_context(request: AssemblyRequest, decision: RouteDecision) -> WorkingContext:
    """Assemble a causal working set through the default deterministic profile."""

    return _DEFAULT_BUILDER.assemble(request, decision)


def assemble_indexed_context(
    request: AssemblyRequest,
    decision: RouteDecision,
    *,
    head_chars: int | None = None,
) -> WorkingContext:
    """Assemble the same working set, rendered as an index rather than as prose.

    Identical selection and identical events in identical order; only the form changes. It takes
    the same `RouteDecision` as `assemble_context` on purpose, so a caller can hold the routing
    fixed and vary the rendering -- which is the only way to tell a routing win from a formatting
    one.

    `head_chars` overrides how much of each body survives. It is the knob a sweep turns, and the
    selection does not depend on it: the budget fit measures the prose cost, so varying the head
    varies the rendering and nothing else.
    """

    if head_chars is None:
        return _INDEXED_BUILDER.assemble(request, decision)
    if head_chars not in _INDEXED_BUILDERS:
        _INDEXED_BUILDERS[head_chars] = ContextBuilder(
            render_mode="index", index_head_chars=head_chars
        )
    return _INDEXED_BUILDERS[head_chars].assemble(request, decision)
