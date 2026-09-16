from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

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


class ContextBuilder:
    def __init__(
        self,
        *,
        analyzer: LexicalAnalyzer | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.analyzer = analyzer or LexicalAnalyzer()
        self.token_counter = token_counter or TokenCounter()
        self.embedding = HashEmbeddingProvider(analyzer=self.analyzer)

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
        for group in evidence_groups:
            group_blocks = [
                self._event_block(event, "evidence", group.context_id)
                for event in group.events
                if event.event_id not in already_selected
            ]
            cost = sum(self.token_counter.count(block.text) for block in group_blocks)
            if group_blocks and cost <= remaining:
                selected_blocks.extend(group_blocks)
                remaining -= cost
                already_selected.update(
                    block.event.event_id for block in group_blocks if block.event is not None
                )
            else:
                for event in group.events:
                    if event.event_id not in already_selected:
                        dropped.append({"event_id": event.event_id, "reason": "token_budget"})

        rendered, spans = self._render(selected_blocks)
        memory_tokens = self.token_counter.count(rendered)
        # Separators may add tokens not charged while selecting.
        # Remove whole trailing blocks until the rendered text is safe.
        while memory_tokens > request.token_budget and selected_blocks:
            removed = selected_blocks.pop()
            if removed.event:
                dropped.append({"event_id": removed.event.event_id, "reason": "separator_budget"})
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
            for event_id in event_ids:
                scores[(event_id, context_id)] = (
                    0.45 * bm25[event_id] / max_bm25
                    + 0.40 * dense[event_id] / max_dense
                    + 0.15 * relevance_by_pair.get((event_id, context_id), 0.0)
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

    @staticmethod
    def _render(blocks: list[_Block]) -> tuple[str, list[SourceSpan]]:
        parts: list[str] = []
        spans: list[SourceSpan] = []
        cursor = 0
        for index, block in enumerate(blocks):
            if index:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(block.text)
            cursor += len(block.text)
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


_DEFAULT_BUILDER = ContextBuilder()


def assemble_context(request: AssemblyRequest, decision: RouteDecision) -> WorkingContext:
    """Assemble a causal working set through the default deterministic profile."""

    return _DEFAULT_BUILDER.assemble(request, decision)
