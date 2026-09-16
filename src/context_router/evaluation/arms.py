"""Offline comparison arms behind the Phase 0 selective-context gate.

Every arm answers the same question — "what memory would this strategy show the main
model for this query?" — and is scored on the same token counter, the same causal
cutoff and the same labelled evidence sets. Only the memory strategy differs, so token
counts and evidence recall are directly comparable across arms.

Arms that do not route (`full_history`, `sliding_window`, the global retrievers) report
``decision="route"`` with ``confidence=1.0``: they always emit their memory and never
abstain, which is exactly the property that makes them the baselines this project has to
beat.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Literal, Protocol

from pydantic import Field

from context_router.assembly import assemble_context
from context_router.assembly.builder import TokenCounter
from context_router.domain import (
    AssemblyRequest,
    BenchmarkQuery,
    Contract,
    Decision,
    EventContextAssignment,
    FlatContext,
    RawEvent,
    Relation,
    RouteDecision,
    RouteRequest,
    SourceSpan,
    WorkingContext,
)
from context_router.evaluation.metrics import evidence_set_recall
from context_router.providers.embedding import HashEmbeddingProvider, cosine
from context_router.retrieval import BM25Index, LexicalAnalyzer
from context_router.routing import ContextRouter

ArmName = Literal[
    "query_recent_only",
    "full_history",
    "sliding_window",
    "summary_recent",
    "global_bm25",
    "global_dense",
    "global_hybrid",
    "hybrid_router",
    "oracle_router",
]

ARM_NAMES: tuple[ArmName, ...] = (
    "query_recent_only",
    "full_history",
    "sliding_window",
    "summary_recent",
    "global_bm25",
    "global_dense",
    "global_hybrid",
    "hybrid_router",
    "oracle_router",
)

Section = Literal["recent", "context", "evidence"]

#: Turns of recent history every arm may see, matching ContextBuilder's own window.
RECENT_TURNS = 3
#: Events a sliding-window baseline keeps.
SLIDING_WINDOW = 8
#: Events a global retriever may pull before grouping and budgeting.
EVIDENCE_LIMIT = 8
#: Oracle must save at least this share of full-history memory tokens to pass gate one.
TOKEN_REDUCTION_TARGET = 0.30

_RRF_K = 60


class ArmCaseSource(Protocol):
    """The slice of the event store the arm harness needs."""

    def list_events(self, session_id: str, as_of_sequence: int | None = None) -> list[RawEvent]: ...

    def list_contexts(self) -> list[FlatContext]: ...

    def list_assignments(
        self, *, session_id: str | None = None, latest_only: bool = False
    ) -> list[EventContextAssignment]: ...

    def get_event(self, event_id: str) -> RawEvent | None: ...


class ArmCase(Contract):
    """One benchmark checkpoint projected into everything every arm is allowed to see."""

    sample_id: str
    session_id: str
    query_event_id: str
    query: str
    as_of_sequence: int = Field(ge=0)
    events: list[RawEvent]
    future_events: list[RawEvent] = Field(default_factory=list)
    contexts: list[FlatContext]
    assignments: list[EventContextAssignment]
    recent_events: list[RawEvent] = Field(default_factory=list)
    required_context_ids: list[str] = Field(default_factory=list)
    acceptable_evidence_sets: list[list[str]] = Field(default_factory=list)
    relation_label: Relation = "unknown"
    primary_context_id: str | None = None
    recent_context_ids: list[str] = Field(default_factory=list)
    token_budget: int = Field(default=2048, ge=32)
    system_rules: str = ""


class ArmCaseResult(Contract):
    """Comparable outcome of running one arm on one checkpoint."""

    arm: ArmName
    sample_id: str
    session_id: str
    required_context_ids: list[str]
    selected_context_ids: list[str]
    included_event_ids: list[str]
    sections: list[str]
    memory_tokens: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    token_budget: int = Field(ge=1)
    evidence_set_recall: float = Field(ge=0.0, le=1.0)
    future_leakage: int = Field(ge=0)
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    routing_trace_id: str
    #: Describes the configuration that produced the result, not just its arm name.
    router_profile: str = "deterministic"


@dataclass(frozen=True)
class _Block:
    text: str
    section: Section
    context_id: str | None = None
    event: RawEvent | None = None


@dataclass(frozen=True)
class _Assembly:
    rendered_text: str
    spans: list[SourceSpan]
    included_event_ids: list[str]
    dropped: list[dict[str, object]]
    memory_tokens: int
    total_input_tokens: int
    sections: list[str]


def _event_block(event: RawEvent, section: Section, context_id: str | None) -> _Block:
    if section == "recent":
        label = "Recent"
    else:
        label = f"Evidence:{context_id}" if context_id else "Evidence"
    text = f"[{label}][{event.event_id}][{event.actor}/{event.kind}] {event.content}"
    return _Block(text=text, section=section, context_id=context_id, event=event)


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
                section=block.section,
            )
        )
    return "".join(parts), spans


def _assemble_blocks(
    blocks: list[_Block], *, case: ArmCase, counter: TokenCounter, budget: int | None
) -> _Assembly:
    selected: list[_Block] = []
    dropped: list[dict[str, object]] = []
    remaining = budget
    for block in blocks:
        cost = counter.count(block.text)
        if remaining is None or cost <= remaining:
            selected.append(block)
            if remaining is not None:
                remaining -= cost
        else:
            dropped.append(
                {
                    "event_id": block.event.event_id if block.event else None,
                    "reason": "token_budget",
                }
            )
    rendered, spans = _render(selected)
    memory_tokens = counter.count(rendered)
    # Separators are not charged while selecting, so drop whole trailing blocks until safe.
    while budget is not None and memory_tokens > budget and selected:
        removed = selected.pop()
        dropped.append(
            {
                "event_id": removed.event.event_id if removed.event else None,
                "reason": "separator_budget",
            }
        )
        rendered, spans = _render(selected)
        memory_tokens = counter.count(rendered)
    return _Assembly(
        rendered_text=rendered,
        spans=spans,
        included_event_ids=[b.event.event_id for b in selected if b.event is not None],
        dropped=dropped,
        memory_tokens=memory_tokens,
        total_input_tokens=memory_tokens
        + counter.count(case.query)
        + counter.count(case.system_rules),
        sections=list(dict.fromkeys(block.section for block in selected)),
    )


def _from_working_context(working: WorkingContext) -> _Assembly:
    return _Assembly(
        rendered_text=working.rendered_text,
        spans=working.source_spans,
        included_event_ids=working.included_event_ids,
        dropped=working.dropped_candidates,
        memory_tokens=working.memory_tokens,
        total_input_tokens=working.total_input_tokens,
        sections=list(dict.fromkeys(span.section for span in working.source_spans)),
    )


def oracle_decision(case: ArmCase) -> RouteDecision:
    """Build the ground-truth routing decision the labelled contexts imply."""

    return RouteDecision(
        decision="route",
        relation=case.relation_label,
        relation_probabilities={case.relation_label: 1.0},
        candidates=[],
        selected_context_ids=list(case.required_context_ids),
        confidence=1.0,
        fallback_level=0,
        trace_id=f"oracle:{case.sample_id}",
        index_version="ground-truth-labels",
        model_versions={"oracle": "ground-truth-labels"},
    )


def build_arm_cases(
    source: ArmCaseSource,
    queries: list[BenchmarkQuery],
    *,
    token_budget: int = 2048,
    system_rules: str = "",
    recent_window: int = 6,
) -> list[ArmCase]:
    """Project benchmark checkpoints into causal arm cases, never reading the future."""

    all_contexts = source.list_contexts()
    session_sequences: dict[str, dict[str, int]] = {}
    cases: list[ArmCase] = []
    for sample in queries:
        session_events = source.list_events(sample.session_id)
        sequence_of = session_sequences.get(sample.session_id)
        if sequence_of is None:
            sequence_of = {event.event_id: event.sequence for event in session_events}
            session_sequences[sample.session_id] = sequence_of
        # A context only exists once its creation event is causal. Filtering on the
        # whole session instead would show checkpoints contexts from their future.
        causal_contexts = [
            context
            for context in all_contexts
            if sequence_of.get(context.created_at_event, sample.as_of_sequence + 1)
            <= sample.as_of_sequence
        ]
        query_event = source.get_event(sample.query_event_id)
        if query_event is None:
            raise KeyError(f"missing query event: {sample.query_event_id}")
        causal = [event for event in session_events if event.sequence <= sample.as_of_sequence]
        future = [event for event in session_events if event.sequence > sample.as_of_sequence]
        recent = [event for event in causal if event.sequence < sample.as_of_sequence][
            -recent_window:
        ]
        cases.append(
            ArmCase(
                sample_id=sample.sample_id,
                session_id=sample.session_id,
                query_event_id=sample.query_event_id,
                query=query_event.content,
                as_of_sequence=sample.as_of_sequence,
                events=causal,
                future_events=future,
                contexts=causal_contexts,
                assignments=source.list_assignments(session_id=sample.session_id, latest_only=True),
                recent_events=recent,
                required_context_ids=list(sample.required_context_ids),
                acceptable_evidence_sets=[list(item) for item in sample.acceptable_evidence_sets],
                relation_label=sample.relation_label,
                primary_context_id=sample.primary_context_id,
                recent_context_ids=list(sample.recent_context_ids),
                token_budget=token_budget,
                system_rules=system_rules,
            )
        )
    return cases


def _recent_blocks(case: ArmCase) -> list[_Block]:
    return [_event_block(event, "recent", None) for event in case.recent_events[-RECENT_TURNS:]]


def _global_documents(case: ArmCase) -> dict[str, str]:
    return {event.event_id: event.content for event in case.events}


def _bm25_order(
    query: str, documents: dict[str, str], analyzer: LexicalAnalyzer, limit: int
) -> list[str]:
    return [key for key, _ in BM25Index(documents, analyzer=analyzer).rank(query, limit)]


def _dense_order(
    query: str, documents: dict[str, str], embedder: HashEmbeddingProvider
) -> list[str]:
    if not documents:
        return []
    keys = list(documents)
    query_vector = embedder.embed([query])[0]
    vectors = embedder.embed([documents[key] for key in keys])
    pairs: list[tuple[float, str]] = [
        (cosine(query_vector, vector), key) for key, vector in zip(keys, vectors, strict=True)
    ]
    pairs.sort(key=lambda item: (-item[0], item[1]))
    return [key for score, key in pairs if score > 0.0]


def _reciprocal_rank_fusion(
    rankings: list[list[str]], weights: list[float] | None = None
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for position, key in enumerate(ranking, start=1):
            scores[key] += weight / (_RRF_K + position)
    return sorted(scores, key=lambda key: (-scores[key], key))


def _global_ranking(
    case: ArmCase,
    mode: Literal["global_bm25", "global_dense", "global_hybrid"],
    analyzer: LexicalAnalyzer,
    embedder: HashEmbeddingProvider,
) -> list[str]:
    documents = _global_documents(case)
    if mode == "global_bm25":
        return _bm25_order(case.query, documents, analyzer, EVIDENCE_LIMIT)
    if mode == "global_dense":
        return _dense_order(case.query, documents, embedder)[:EVIDENCE_LIMIT]
    fused = _reciprocal_rank_fusion(
        [
            _bm25_order(case.query, documents, analyzer, len(documents)),
            _dense_order(case.query, documents, embedder),
        ]
    )
    return fused[:EVIDENCE_LIMIT]


def _expand_groups(case: ArmCase, ranking: list[str]) -> list[list[RawEvent]]:
    """Attach tool call/result pairs and one adjacent turn, as indivisible evidence."""

    by_id = {event.event_id: event for event in case.events}
    by_sequence = {event.sequence: event for event in case.events}
    children: dict[str, list[RawEvent]] = defaultdict(list)
    for event in case.events:
        if event.parent_event_id:
            children[event.parent_event_id].append(event)
    groups: list[list[RawEvent]] = []
    seen: set[str] = set()
    for event_id in ranking:
        anchor = by_id.get(event_id)
        if anchor is None:
            continue
        members: dict[str, RawEvent] = {anchor.event_id: anchor}
        if anchor.kind == "tool_call":
            members.update({child.event_id: child for child in children.get(anchor.event_id, [])})
        elif anchor.kind == "tool_result" and anchor.parent_event_id in by_id:
            parent = by_id[anchor.parent_event_id]
            members[parent.event_id] = parent
            members.update({child.event_id: child for child in children.get(parent.event_id, [])})
        for neighbour in (anchor.sequence - 1, anchor.sequence + 1):
            adjacent = by_sequence.get(neighbour)
            if adjacent is not None:
                members[adjacent.event_id] = adjacent
        fresh = [
            event
            for event in sorted(members.values(), key=lambda e: e.sequence)
            if event.event_id not in seen
        ]
        if not fresh:
            continue
        seen.update(event.event_id for event in fresh)
        groups.append(fresh)
    return groups


def _evidence_blocks(case: ArmCase, ranking: list[str]) -> list[_Block]:
    return [
        _event_block(event, "evidence", None)
        for group in _expand_groups(case, ranking)
        for event in group
    ]


def _context_map(case: ArmCase) -> dict[str, FlatContext]:
    return {context.context_id: context for context in case.contexts}


def _active_context_ids(case: ArmCase) -> list[str]:
    candidates = [case.primary_context_id, *case.recent_context_ids]
    return list(dict.fromkeys(cid for cid in candidates if cid))


def _result(
    arm: ArmName,
    case: ArmCase,
    assembly: _Assembly,
    *,
    selected_context_ids: list[str],
    decision: Decision = "route",
    confidence: float = 1.0,
    trace_id: str = "",
    router_profile: str = "deterministic",
) -> ArmCaseResult:
    causal = {event.event_id for event in case.events}
    included = set(assembly.included_event_ids)
    return ArmCaseResult(
        arm=arm,
        sample_id=case.sample_id,
        session_id=case.session_id,
        required_context_ids=list(case.required_context_ids),
        selected_context_ids=list(selected_context_ids),
        included_event_ids=list(assembly.included_event_ids),
        sections=list(assembly.sections),
        memory_tokens=assembly.memory_tokens,
        total_input_tokens=assembly.total_input_tokens,
        token_budget=case.token_budget,
        evidence_set_recall=evidence_set_recall(
            case.acceptable_evidence_sets, assembly.included_event_ids
        ),
        future_leakage=len(included - causal),
        decision=decision,
        confidence=confidence,
        routing_trace_id=trace_id,
        router_profile=router_profile,
    )


def describe_router(router: ContextRouter) -> str:
    """A stable label for the router configuration a result was produced with."""

    policy = router.policy
    thresholds = ",".join(
        f"{name}={getattr(policy, name)}"
        for name in ("t_low", "t_high", "margin", "confidence_threshold", "max_candidates")
    )
    return "|".join(
        (
            f"ranker={router.ranker.model_version}",
            f"relation={router.relation_classifier.model_version}",
            f"embedding={router.embedding_provider.model_version}",
            f"policy({thresholds})",
        )
    )


def run_arm(
    name: ArmName,
    case: ArmCase,
    *,
    counter: TokenCounter | None = None,
    router: ContextRouter | None = None,
) -> ArmCaseResult:
    """Run a single comparison arm on a single checkpoint.

    ``router`` lets the hybrid arm report its *configured* performance, so a trained
    ranker or a tuned policy can be measured instead of only the defaults.
    """

    tokens = counter or TokenCounter()
    analyzer = LexicalAnalyzer()
    embedder = HashEmbeddingProvider(analyzer=analyzer)

    if name == "query_recent_only":
        assembly = _assemble_blocks(
            _recent_blocks(case), case=case, counter=tokens, budget=case.token_budget
        )
        return _result(name, case, assembly, selected_context_ids=[])

    if name == "full_history":
        blocks = [_event_block(event, "evidence", None) for event in case.events]
        assembly = _assemble_blocks(blocks, case=case, counter=tokens, budget=None)
        return _result(name, case, assembly, selected_context_ids=[])

    if name == "sliding_window":
        blocks = [_event_block(event, "recent", None) for event in case.events[-SLIDING_WINDOW:]]
        assembly = _assemble_blocks(blocks, case=case, counter=tokens, budget=case.token_budget)
        return _result(name, case, assembly, selected_context_ids=[])

    if name == "summary_recent":
        active = _active_context_ids(case)
        blocks = _descriptor_blocks(active, _context_map(case)) + _recent_blocks(case)
        assembly = _assemble_blocks(blocks, case=case, counter=tokens, budget=case.token_budget)
        present = [cid for cid in active if cid in _context_map(case)]
        return _result(name, case, assembly, selected_context_ids=present)

    if name in ("global_bm25", "global_dense", "global_hybrid"):
        ranking = _global_ranking(case, name, analyzer, embedder)
        blocks = _evidence_blocks(case, ranking)
        assembly = _assemble_blocks(blocks, case=case, counter=tokens, budget=case.token_budget)
        return _result(name, case, assembly, selected_context_ids=[])

    request = AssemblyRequest(
        query=case.query,
        recent_events=case.recent_events,
        event_pool=case.events + case.future_events,
        context_catalog=case.contexts,
        assignments=case.assignments,
        as_of_sequence=case.as_of_sequence,
        token_budget=case.token_budget,
        system_rules=case.system_rules,
    )
    active_router = router or ContextRouter()
    profile = "deterministic"
    if name == "hybrid_router":
        profile = describe_router(active_router)
    elif name == "oracle_router":
        profile = "ground-truth"
    if name == "hybrid_router":
        decision = active_router.route(
            RouteRequest(
                query_event_id=case.query_event_id,
                query=case.query,
                recent_events=case.recent_events,
                primary_context_id=case.primary_context_id,
                recent_context_ids=case.recent_context_ids,
                context_catalog=case.contexts,
                as_of_sequence=case.as_of_sequence,
            )
        )
    else:
        decision = oracle_decision(case)
    assembly = _from_working_context(assemble_context(request, decision))
    return _result(
        name,
        case,
        assembly,
        selected_context_ids=decision.selected_context_ids,
        decision=decision.decision,
        confidence=decision.confidence,
        trace_id=decision.trace_id,
        router_profile=profile,
    )


def evaluate_arms(results: list[ArmCaseResult]) -> dict[str, dict[str, float]]:
    """Aggregate arm results into one comparable row per arm."""

    if not results:
        raise ValueError("at least one arm result is required")
    grouped: dict[str, list[ArmCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.arm].append(result)
    summary: dict[str, dict[str, float]] = {}
    for arm, rows in grouped.items():
        tokens = [float(row.memory_tokens) for row in rows]
        summary[arm] = {
            "count": float(len(rows)),
            "mean_memory_tokens": mean(tokens),
            "median_memory_tokens": float(median(tokens)),
            "max_memory_tokens": max(tokens),
            "mean_total_input_tokens": mean(float(row.total_input_tokens) for row in rows),
            "evidence_set_recall": mean(row.evidence_set_recall for row in rows),
            "future_leakage_total": float(sum(row.future_leakage for row in rows)),
            "abstention_rate": mean(float(row.decision == "abstain") for row in rows),
        }
    reference = summary.get("full_history", {}).get("mean_memory_tokens")
    if reference:
        for row in summary.values():
            row["token_reduction_vs_full_history"] = 1.0 - row["mean_memory_tokens"] / reference
    return summary


def first_gate(summary: dict[str, dict[str, float]]) -> dict[str, object]:
    """Decide whether selective context is worth a real router at all.

    Gate one is deliberately answerable offline: it compares memory tokens and evidence
    recall against full history. The plan's answer-quality criterion needs a pinned main
    model, so it is reported as unverified rather than silently assumed.
    """

    full = summary.get("full_history")
    oracle = summary.get("oracle_router")
    if full is None or oracle is None:
        raise KeyError("first_gate requires the full_history and oracle_router arms")
    reference = full["mean_memory_tokens"]
    reduction = 1.0 - oracle["mean_memory_tokens"] / reference if reference else 0.0
    recall_ok = oracle["evidence_set_recall"] >= full["evidence_set_recall"] - 1e-9
    token_ok = reduction >= TOKEN_REDUCTION_TARGET
    return {
        "oracle_token_reduction": reduction,
        "token_reduction_target": TOKEN_REDUCTION_TARGET,
        "token_gate_passed": token_ok,
        "full_history_mean_memory_tokens": full["mean_memory_tokens"],
        "oracle_mean_memory_tokens": oracle["mean_memory_tokens"],
        "full_history_recall": full["evidence_set_recall"],
        "oracle_recall": oracle["evidence_set_recall"],
        "oracle_recall_not_worse_than_full_history": recall_ok,
        "answer_quality_verified": False,
        "verdict": "continue" if token_ok and recall_ok else "stop",
    }
