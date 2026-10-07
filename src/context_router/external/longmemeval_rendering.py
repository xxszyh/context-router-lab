"""Label-free memory rendering with provenance for each fully visible source word.

Annotations are never accepted by this interface. The legacy policy preserves
existing prompts; the optional union policy removes overlapping source spans and
keeps each session's original order and whitespace.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Protocol

from context_router.external.longmemeval import session_passages
from context_router.retrieval import BM25Index

LEGACY_RENDERING = "equal-share-query-passages-v1"
UNION_RENDERING = "chronological-union-v2"
TURN_RENDERING = "whole-turn-bm25-v3"
RENDERINGS = (LEGACY_RENDERING, UNION_RENDERING, TURN_RENDERING)
PASSAGE_WORDS = 180
PASSAGE_OVERLAP = 40
PASSAGE_STRIDE = PASSAGE_WORDS - PASSAGE_OVERLAP


class Budget(Protocol):
    maximum: int

    def count(self, text: str) -> int: ...

    def clip(self, text: str, maximum: int) -> str: ...


@dataclass(frozen=True)
class VisibleSession:
    date: str
    turns: tuple[tuple[str, str], ...]


@dataclass
class SessionText:
    body: str
    words: list[re.Match[str]]
    content_ranges: list[tuple[int, int]]
    roles: list[str]


def index_session(turns: tuple[tuple[str, str], ...]) -> SessionText:
    pieces = []
    content_ranges = []
    offset = 0
    for role, text in turns:
        prefix = f"{role}: "
        pieces.append(prefix + text)
        content_ranges.append((offset + len(prefix), offset + len(prefix) + len(text)))
        offset += len(prefix) + len(text) + 1
    body = "\n".join(pieces)
    return SessionText(body, list(re.finditer(r"\S+", body)), content_ranges, [r for r, _ in turns])


def word_ranges(indices: set[int]) -> list[list[int]]:
    """Compact source indices into half-open ranges."""
    ranges: list[list[int]] = []
    for index in sorted(indices):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] += 1
        else:
            ranges.append([index, index + 1])
    return ranges


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _union_text(
    source: SessionText, ranges: list[tuple[int, int]], *, include_trace: bool = True
) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    trace: list[tuple[int, int]] = []
    offset = 0
    for lo, hi in _merge(ranges):
        start, end = source.words[lo].start(), source.words[hi - 1].end()
        prefix = ""
        # A window starting mid-turn needs its original speaker, not a guessed role.
        for role, (content_start, content_end) in zip(
            source.roles, source.content_ranges, strict=True
        ):
            if content_start <= start < content_end:
                prefix = f"[{role} excerpt]\n"
                break
        separator = "\n[...]\n" if pieces else ""
        offset += len(separator) + len(prefix)
        pieces.append(separator + prefix + source.body[start:end])
        if include_trace:
            trace.extend((j, offset + source.words[j].end() - start) for j in range(lo, hi))
        offset += end - start
    return "".join(pieces), trace


def _legacy_chunk(
    source: SessionText,
    header: str,
    order: list[int],
    passages: list[str],
    share: int,
    budget: Budget,
) -> tuple[str, list[tuple[int, int]], list[int]]:
    remaining = max(0, share - budget.count(header) - budget.count("\n\n"))
    parts: list[str] = []
    chosen: list[int] = []
    trace: list[tuple[int, int]] = []
    offset = len(header)
    for j in order:
        if remaining <= 0:
            break
        piece = budget.clip(passages[j], remaining)
        # Map complete words from the full passage, not a possibly partial final word.
        trace.extend(
            (j * PASSAGE_STRIDE + k, offset + match.end())
            for k, match in enumerate(re.finditer(r"\S+", passages[j]))
            if match.end() <= len(piece)
        )
        parts.append(piece)
        chosen.append(j)
        remaining -= budget.count(piece) + budget.count("\n")
        offset += len(piece) + 1
    chunk = budget.clip(header + "\n".join(parts), max(0, share - budget.count("\n\n")))
    return chunk, [(word, end) for word, end in trace if end <= len(chunk)], chosen


def _union_chunk(
    source: SessionText, header: str, order: list[int], share: int, budget: Budget
) -> tuple[str, list[tuple[int, int]], list[int]]:
    maximum = max(0, share - budget.count("\n\n"))
    selected: list[tuple[int, int]] = []
    chosen: list[int] = []
    body = ""
    trace: list[tuple[int, int]] = []
    for j in order:
        lo = j * PASSAGE_STRIDE
        hi = min(lo + PASSAGE_WORDS, len(source.words))
        if lo >= hi:
            continue
        trial = _merge([*selected, (lo, hi)])
        if trial == selected:
            continue
        partial = False
        candidate, _ = _union_text(source, trial, include_trace=False)
        # At most 180 word-boundary checks for a final, partial window. There is
        # no character clipping that could invent a complete evidence word.
        while hi > lo and budget.count(header + candidate) > maximum:
            partial = True
            hi -= 1
            trial = _merge([*selected, (lo, hi)]) if hi > lo else selected
            candidate, _ = _union_text(source, trial, include_trace=False)
        if hi > lo and trial != selected:
            selected = trial
            body, trace = _union_text(source, selected)
            chosen.append(j)
        if partial:
            break
    chunk = budget.clip(header + body, maximum)
    return (
        chunk,
        [(word, len(header) + end) for word, end in trace if len(header) + end <= len(chunk)],
        chosen,
    )


def _turn_chunk(
    source: SessionText, header: str, question: str, share: int, budget: Budget
) -> tuple[str, list[tuple[int, int]], list[int]]:
    maximum = max(0, share - budget.count("\n\n"))
    spans = []
    documents = {}
    starts = [word.start() for word in source.words]
    ends = [word.end() for word in source.words]
    for j, ((start, end), role) in enumerate(zip(source.content_ranges, source.roles, strict=True)):
        beginning = start - len(f"{role}: ")
        spans.append((bisect_left(starts, beginning), bisect_right(ends, end)))
        documents[str(j)] = source.body[beginning:end]
    scores = BM25Index(documents).scores(question) if documents else {}
    order = sorted(range(len(spans)), key=lambda j: (-scores[str(j)], j))
    selected: list[tuple[int, int]] = []
    chosen = []
    body = ""
    for j in order:
        lo, hi = spans[j]
        if lo >= hi:
            continue
        trial = _merge([*selected, (lo, hi)])
        candidate, _ = _union_text(source, trial, include_trace=False)
        if budget.count(header + candidate) <= maximum:
            selected, body = trial, candidate
            chosen.append(j)
    if not selected:
        # A session containing only oversized turns still gets bounded excerpts.
        passages = session_passages(source.body, words=PASSAGE_WORDS, overlap=PASSAGE_OVERLAP)
        window_scores = BM25Index({str(j): p for j, p in enumerate(passages)}).scores(question)
        windows = sorted(range(len(passages)), key=lambda j: (-window_scores[str(j)], j))
        chunk, trace, _ = _union_chunk(source, header, windows, share, budget)
        return chunk, trace, []
    _, trace = _union_text(source, selected)
    chunk = budget.clip(header + body, maximum)
    return (
        chunk,
        [(word, len(header) + end) for word, end in trace if len(header) + end <= len(chunk)],
        chosen,
    )


def render_memory(
    question: str, sessions: list[VisibleSession], budget: Budget, *, policy: str = LEGACY_RENDERING
) -> tuple[str, list[dict[str, Any]]]:
    if policy not in RENDERINGS:
        raise ValueError("unsupported memory rendering policy")
    if not sessions:
        return "", []
    share = budget.maximum // len(sessions)
    chunks = []
    pending = []
    offset = 0
    for i, session in enumerate(sessions, 1):
        source = index_session(session.turns)
        header = f"[Memory {i}; date {session.date}; excerpts]\n"
        if policy == TURN_RENDERING:
            chunk, trace, chosen = _turn_chunk(source, header, question, share, budget)
            count = len(source.content_ranges)
        else:
            passages = session_passages(source.body, words=PASSAGE_WORDS, overlap=PASSAGE_OVERLAP)
            scores = BM25Index({str(j): text for j, text in enumerate(passages)}).scores(question)
            order = sorted(range(len(passages)), key=lambda j: (-scores[str(j)], j))
            if policy == LEGACY_RENDERING:
                chunk, trace, chosen = _legacy_chunk(source, header, order, passages, share, budget)
            else:
                chunk, trace, chosen = _union_chunk(source, header, order, share, budget)
            count = len(passages)
        chunks.append(chunk)
        pending.append((i, source, chosen, count, [(w, offset + e) for w, e in trace]))
        offset += len(chunk) + 2
    memory = budget.clip("\n\n".join(chunks), budget.maximum)
    audit = []
    for slot, source, chosen, count, trace in pending:
        retained = [word for word, end in trace if end <= len(memory)]
        unique = set(retained)
        audit.append(
            {
                "slot": slot,
                "selection_unit": "turn" if policy == TURN_RENDERING else "word_window",
                "used_window_fallback": policy == TURN_RENDERING and not chosen,
                "passages": chosen,
                "source_passages": count,
                "source_words": len(source.words),
                "retained_word_ranges": word_ranges(unique),
                "retained_unique_words": len(unique),
                "retained_word_occurrences": len(retained),
                "duplicated_word_occurrences": len(retained) - len(unique),
            }
        )
    return memory, audit
