"""Adapter for the SCALE-QA external benchmark.

SCALE-QA (arXiv:2608.25655) is the closest published setting to this project's question: 3,000
four-way multiple-choice questions across ten domains, each with an item-level source dialogue,
and a runtime builder that **flattens those dialogues into one mixed-topic turn stream with the
episode metadata removed**. Grading is deterministic multiple choice, so no judge is involved.

Three properties make it worth adapting to rather than rebuilding:

* every question carries ``expected_doc`` -- the exact evidence copied out of its own dialogue --
  which is the per-query evidence ground truth this project had to delete as circular;
* the answer is a letter, so evidence recall is computable offline and free;
* it is not this project's benchmark, so it cannot have been built to confirm this project.

**What it does not give is a discriminating arrangement.** The builder produces a truth corpus and
a noise corpus and leaves their order to the evaluator. Concatenating truth then noise -- the
arrangement a plain reading of "combine ``GROUND_TRUTH_HISTORY`` and ``NOISE``" produces -- puts
every question's evidence a median of 125,350 tokens from the end of a 128k stream: a recency
baseline scores zero on it, which is the mirror image of this project's own generator, where
recency is right 100% of the time on some query types. Both are degenerate, and neither is the
benchmark's fault.

So the arrangement is a **named, seeded, reported parameter** here, exactly as
``RECENT_BUDGET_FRACTION`` now is in the builder, and ``arrangement_reachability`` reports what it
did. A result from this module that does not name its arrangement is not comparable to another one,
which is the same rule `describe_builder` enforces inside the repo.

The dense channel is the hashing placeholder unless a provider is supplied. That is the project's
own standing limitation (limitation 2 in the README), and it is why the default baseline here is
lexical: ``fixed_window_lexical`` is a real, reproducible baseline, and it is **not** numerically
comparable to SCALE-QA's own dense fixed-window numbers. Adding a real embedder is a one-argument
change, and the module says so rather than silently reporting a placeholder as dense retrieval.
"""

from __future__ import annotations

import importlib.util
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from context_router.assembly.builder import TokenCounter
from context_router.domain import ContextCandidate, FlatContext, RouteDecision
from context_router.providers.embedding import EmbeddingProvider, HashEmbeddingProvider, cosine
from context_router.retrieval import BM25Index, LexicalAnalyzer
from context_router.routing import ContextRouter

#: How the evaluator orders the truth and noise blocks before anyone reads them.
#:
#: ``blocks`` is the naive concatenation and is degenerate by construction; it is kept because it
#: is what the obvious reading produces, and because a benchmark report should show that it is
#: degenerate rather than assert it. ``interleaved`` distributes noise between truth dialogues;
#: ``shuffled`` permutes every block. Both are seeded, so both are reproducible.
Arrangement = Literal["blocks", "interleaved", "shuffled"]

#: Window sizes to sweep, matching the fixed-window controls in SCALE-QA's Table 21.
WINDOW_SIZES: tuple[int, ...] = (128, 256, 320)

#: Retrieved units per query, matching SCALE-QA's recall@5.
RETRIEVAL_DEPTH = 5

#: The arm whose depth-1 result decides, in `difficulty_strata`, which questions a cheap lookup
#: has already answered. Lexical because it is the cheapest real retriever here and the one every
#: other arm is measured against; naming it as a constant keeps a report from calling something
#: else a baseline without saying so.
STRATA_BASELINE = "lexical"

#: Ordering fields the attribution compares: each channel's own rank, the fusion score, and the
#: calibrated probability that produces the order the router actually uses.
ORDERING_FIELDS: tuple[str, ...] = (
    "lexical_rank",
    "dense_rank",
    "entity_rank",
    "rrf_score",
    "calibrated_probability",
)

#: The subset of those where lower is better and a missing value means "this channel never
#: retrieved it", which sorts last rather than first.
RANK_ORDERING_FIELDS: tuple[str, ...] = ("lexical_rank", "dense_rank", "entity_rank")


@dataclass(frozen=True)
class ScaleQaQuestion:
    global_id: str
    query: str
    expected_doc: tuple[str, ...]
    answer_key: str
    topic: str
    creator: str


@dataclass(frozen=True)
class ScaleQaBlock:
    """One source dialogue or noise document, as a run of turns."""

    index: int
    kind: Literal["truth", "noise"]
    text: str
    turns: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ScaleQaWindow:
    """A fixed-size retrieved unit, which is what SCALE-QA's fixed-window control returns."""

    window_id: str
    text: str
    start_block: int
    tokens: int


@dataclass(frozen=True)
class ScaleQaPackage:
    truth: tuple[ScaleQaBlock, ...]
    noise: tuple[ScaleQaBlock, ...]
    questions: tuple[ScaleQaQuestion, ...]
    source: str


@dataclass(frozen=True)
class ArrangementReachability:
    """What an arrangement did to the recency question, per this repo's own diagnostic."""

    arrangement: str
    seed: int
    stream_tokens: int
    truth_tokens: int
    questions: int
    located: int
    median_distance_from_end: float
    reachable_within_window: dict[int, float]


@dataclass(frozen=True)
class DifficultyStratum:
    """One arm's exact-evidence recall, split by whether one cheap lookup already answers it."""

    mode: str
    questions: int
    solved_by_baseline: int
    solved_share: float
    recall_pooled: float
    recall_on_solved: float | None
    recall_on_unsolved: float | None


@dataclass(frozen=True)
class DifficultyReport:
    """How much of a benchmark one cheap baseline already answers.

    A pooled recall number averages over questions of unknown difficulty. On a benchmark where a
    single lookup settles most of them, that average is mostly a statement about the benchmark and
    only weakly one about any arm: every arm containing a lexical channel scores near 1.000 on the
    solved part and the whole comparison is decided by the remainder.

    A question is *solved* when one unit returned by the baseline at ``baseline_depth`` already
    contains all of the question's evidence. Only ``expected_doc`` is needed, so this runs on any
    benchmark that carries per-query evidence labels -- it is `recency_reachability` generalised
    from "can a recency window reach the evidence" to "has a cheap lookup already answered this".

    An empty stratum is ``None`` rather than ``0.0``, the same rule as `requirement_reachability`:
    a subset with no questions in it has no recall, and reporting one as zero would be a number
    the measurement did not produce.
    """

    baseline: str
    baseline_depth: int
    depth: int
    questions: int
    solved_by_baseline: int
    solved_share: float
    strata: tuple[DifficultyStratum, ...]


def _mean(values: Sequence[float]) -> float | None:
    """The mean, or ``None`` when there is nothing to average -- never ``0.0``."""

    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class MissDecomposition:
    """A routing miss split into "never offered" and "offered and then dropped".

    Counting misses says an arm is behind; this says *where*, in the vocabulary this project
    already uses on its own benchmark. A **candidate miss** means the evidence was never in the
    candidate list, which points at retrieval. A **selection loss** means it was in the list and
    was not returned, which points at the ranker and the selection policy -- and is the one of the
    two that a change to ranking can fix.

    ``no_evidence_window`` counts questions no unit at this window size holds the evidence for, so
    they are unanswerable rather than mis-retrieved. They are also counted in ``candidate_miss``;
    the field exists so a reader can subtract them.
    """

    questions: int
    recovered: int
    selection_loss: int
    candidate_miss: int
    no_evidence_window: int
    selected_counts: dict[int, int]
    median_candidate_rank: float | None


@dataclass(frozen=True)
class OrderingAttribution:
    """What one ordering of the same candidate list would have recalled."""

    field: str
    questions: int
    recall_at_depth: float


@dataclass(frozen=True)
class OrderingReport:
    """Which stage of a ranker's pipeline loses the ordering.

    The candidate list is held fixed and only the order changes, so a difference between two rows
    belongs to the field and not to retrieval. A ranker whose fused order scores below its own best
    single channel is paying for the fusion; one whose calibrated order scores below the fusion is
    paying for the calibration.

    ``oracle_at_depth`` is the ceiling the list itself sets -- the recall of any ``depth`` of its
    members, which no ordering can exceed. The distance from the best ordering to it is what a
    better ranker could still win.
    """

    questions: int
    depth: int
    candidate_size_median: float | None
    orderings: tuple[OrderingAttribution, ...]
    oracle_at_depth: float


def baseline_solved_mask(
    questions: tuple[ScaleQaQuestion, ...],
    retrievers: Mapping[str, WindowRetriever],
    *,
    baseline: str = STRATA_BASELINE,
    baseline_depth: int = 1,
) -> list[bool]:
    """Which questions a cheap baseline already answers -- the primitive the split is built on.

    Exposed rather than inlined into `difficulty_strata` so a caller can select the hard subset for
    a second measurement, and so the subset it selects is the same one the split reports instead of
    a re-derivation that agrees by luck.
    """

    if baseline not in retrievers:
        raise ValueError(f"baseline {baseline!r} is not among {sorted(retrievers)}")
    return [
        recall == 1.0
        for recall in retrievers[baseline].recall_per_question(questions, depth=baseline_depth)
    ]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_package(root: Path) -> ScaleQaPackage:
    """Load a runtime package produced by SCALE-QA's ``build_eval_dataset.py``.

    Accepts either the package root or the ``runtime/<mode>`` directory itself, because both are
    what a reader of SCALE-QA's README will have in hand.
    """

    runtime = root if (root / "test_cases.py").exists() else None
    if runtime is None:
        candidates = sorted(root.glob("runtime/*/test_cases.py"))
        if not candidates:
            raise FileNotFoundError(f"no test_cases.py under {root}")
        runtime = candidates[0].parent
    cases = _load_module(runtime / "test_cases.py", "scale_qa_test_cases")
    noise_module = _load_module(runtime / "processed_noise.py", "scale_qa_noise")

    def to_block(index: int, kind: Literal["truth", "noise"], dialogue: Any) -> ScaleQaBlock:
        turns = tuple((str(text), str(role)) for text, role in dialogue)
        return ScaleQaBlock(
            index=index,
            kind=kind,
            text="\n".join(text for text, _ in turns),
            turns=turns,
        )

    truth = tuple(
        to_block(index, "truth", dialogue)
        for index, dialogue in enumerate(cases.GROUND_TRUTH_HISTORY)
    )
    # Noise blocks continue the truth blocks' index space rather than restarting it. They share
    # one stream and one offset table, so a second 0 would silently overwrite the first block's
    # position and corrupt every distance `arrangement_reachability` reports.
    noise = tuple(
        to_block(len(truth) + index, "noise", dialogue)
        for index, dialogue in enumerate(noise_module.NOISE)
    )
    questions = tuple(
        ScaleQaQuestion(
            global_id=str(row["global_id"]),
            query=str(row["query"]),
            expected_doc=tuple(str(item) for item in row["expected_doc"]),
            answer_key=str(row["expected_answer_keywords"]),
            topic=str(row.get("topic", "")),
            creator=str(row.get("creator", "")),
        )
        for row in cases.EVALUATION_QUERIES
    )
    return ScaleQaPackage(truth=truth, noise=noise, questions=questions, source=str(runtime))


def arrange(package: ScaleQaPackage, *, arrangement: Arrangement, seed: int) -> list[ScaleQaBlock]:
    """Order the blocks. This is the parameter SCALE-QA leaves to the evaluator."""

    if arrangement == "blocks":
        return [*package.truth, *package.noise]
    rng = random.Random(seed)
    if arrangement == "shuffled":
        blocks = [*package.truth, *package.noise]
        rng.shuffle(blocks)
        return blocks
    # `interleaved`: every truth dialogue, with noise spread evenly between them, so the stream
    # reads as a real interleaved session rather than two concatenated corpora.
    if not package.truth:
        return [*package.noise]
    per_gap = max(len(package.noise) // (len(package.truth) + 1), 0)
    pool = list(package.noise)
    rng.shuffle(pool)
    ordered: list[ScaleQaBlock] = []
    cursor = 0
    for index, block in enumerate(package.truth):
        take = per_gap if index else per_gap
        ordered.extend(pool[cursor : cursor + take])
        cursor += take
        ordered.append(block)
    ordered.extend(pool[cursor:])
    return ordered


def window_blocks(
    blocks: list[ScaleQaBlock],
    *,
    size: int,
    counter: TokenCounter,
) -> list[ScaleQaWindow]:
    """Cut the stream into fixed-size windows -- SCALE-QA's fixed-window retrieved unit.

    Windows never span a block boundary, so a window is always inside one source dialogue. That
    matches the control the paper describes (a window is a retrieved unit, not a slice of the
    concatenation) and it keeps the unit meaningful in a stream built from whole dialogues.
    """

    windows: list[ScaleQaWindow] = []
    for block in blocks:
        sentences = [line for line in block.text.splitlines() if line.strip()]
        current: list[str] = []
        current_tokens = 0
        ordinal = 0
        for sentence in sentences:
            cost = counter.count(sentence)
            if current and current_tokens + cost > size:
                windows.append(
                    ScaleQaWindow(
                        window_id=f"w{len(windows):05d}",
                        text="\n".join(current),
                        start_block=block.index,
                        tokens=current_tokens,
                    )
                )
                ordinal += 1
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += cost
        if current:
            windows.append(
                ScaleQaWindow(
                    window_id=f"w{len(windows):05d}",
                    text="\n".join(current),
                    start_block=block.index,
                    tokens=current_tokens,
                )
            )
    return windows


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def evidence_recall(question: ScaleQaQuestion, assembled: str) -> float:
    """Share of the question's exact evidence present in the assembled memory.

    This is SCALE-QA's ``all-evidence recall`` restricted to one unit budget, and it is exact:
    ``expected_doc`` is copied out of the source dialogue, so containment is checkable by string
    search with no model and no judge. Whitespace is collapsed on both sides because the runtime
    package renders turns on separate lines.
    """

    if not question.expected_doc:
        return 0.0
    haystack = _normalize(assembled)
    hits = sum(1 for item in question.expected_doc if _normalize(item) in haystack)
    return hits / len(question.expected_doc)


def arrangement_reachability(
    blocks: list[ScaleQaBlock],
    questions: tuple[ScaleQaQuestion, ...],
    *,
    counter: TokenCounter,
    arrangement: str,
    seed: int,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
) -> ArrangementReachability:
    """Report whether the arrangement can separate a recency strategy from a relevance one.

    Same instrument as `evaluation.arms.recency_reachability`, applied to a stream instead of a
    case list: distance from each question's evidence to the end of the stream, and the share of
    questions a trailing window of each size would reach. A share of 0.0 or 1.0 means the
    arrangement decides the recency question in advance and cannot be evidence about it.
    """

    offsets: dict[int, int] = {}
    cursor = 0
    for block in blocks:
        offsets[block.index] = cursor
        cursor += counter.count(block.text)
    total = cursor

    truth_tokens = sum(counter.count(block.text) for block in blocks if block.kind == "truth")
    distances: list[float] = []
    for question in questions:
        evidence = [_normalize(item) for item in question.expected_doc]
        position: int | None = None
        for block in blocks:
            if block.kind != "truth":
                continue
            haystack = _normalize(block.text)
            if all(item in haystack for item in evidence):
                # Distance to the end of the block, not its start: the window has to contain the
                # evidence, and within a dialogue the evidence is usually in the last turns.
                position = offsets[block.index] + counter.count(block.text)
                break
        if position is not None:
            distances.append(float(total - position))

    reachable = {
        size: (
            sum(1 for distance in distances if distance <= size) / len(distances)
            if distances
            else 0.0
        )
        for size in window_sizes
    }
    ordered = sorted(distances)
    median = ordered[len(ordered) // 2] if ordered else 0.0
    return ArrangementReachability(
        arrangement=arrangement,
        seed=seed,
        stream_tokens=total,
        truth_tokens=truth_tokens,
        questions=len(questions),
        located=len(distances),
        median_distance_from_end=median,
        reachable_within_window=reachable,
    )


def _window_context(
    window: ScaleQaWindow, analyzer: LexicalAnalyzer, *, derived_fields: bool = False
) -> FlatContext:
    """A window as a routable context.

    SCALE-QA removes episode metadata on purpose, so a window has no name, goal or summary that
    the benchmark supplies. By default every one of those fields is left empty and ``summary`` is
    the window's own text, so ``FlatContext.searchable_text()`` is exactly the window and nothing
    else -- the router then sees the same document the lexical baseline sees.

    ``derived_fields`` fills name, entities and lexical terms from the window text, which adds no
    information the benchmark withheld. It is off by default because it was **measured to hurt**:
    on the 50-question interleaved 256-token arrangement, all-evidence recall at depth 5 was
    **0.900 with the derived fields and 0.980 without**, against a plain-BM25 baseline of 1.000.
    `searchable_text` concatenates a context's name, goal, summary, entities and terms into one
    indexed document, so padding a window with identifiers and top terms it already contains
    lengthens the document, flattens the BM25 term weights and costs eight points of recall. The
    knob is kept so the effect stays re-measurable rather than becoming folklore.
    """

    if not derived_fields:
        return FlatContext(
            context_id=window.window_id,
            name="",
            goal="",
            summary=window.text,
            entities=[],
            lexical_terms=[],
            status="active",
            created_at_event=window.window_id,
            last_active_sequence=window.start_block,
            version=1,
        )
    terms = analyzer.tokens(window.text)
    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_.\-]{2,}", window.text)
    return FlatContext(
        context_id=window.window_id,
        name=f"window {window.window_id}",
        goal="",
        summary=window.text,
        entities=list(dict.fromkeys(identifiers))[:20],
        lexical_terms=[term for term, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:20]],
        status="active",
        created_at_event=window.window_id,
        last_active_sequence=window.start_block,
        version=1,
    )


RetrievalMode = Literal["lexical", "dense", "hybrid", "router"]


def retrieve(
    windows: list[ScaleQaWindow],
    query: str,
    *,
    mode: RetrievalMode,
    analyzer: LexicalAnalyzer,
    embedder: EmbeddingProvider,
    depth: int = RETRIEVAL_DEPTH,
    derived_fields: bool = False,
) -> list[ScaleQaWindow]:
    """Return the units one arm would show the model.

    ``lexical`` is the default baseline and the only one that is meaningful with the hashing
    embedder. ``dense`` and ``hybrid`` exist so a real provider can be dropped in; with the
    placeholder they measure the placeholder. ``router`` runs this project's calibrated ranker and
    relation classifier over the same units, which is the comparison the adapter exists for.
    """

    if not windows:
        return []
    by_id = {window.window_id: window for window in windows}
    documents = {window.window_id: window.text for window in windows}
    index = BM25Index(documents, analyzer=analyzer)
    lexical = [key for key, _ in index.rank(query, depth)]

    if mode == "lexical":
        return [by_id[key] for key in lexical]

    query_vector = embedder.embed([query])[0]
    vectors = embedder.embed([documents[key] for key in documents])
    dense_scores = {
        key: max(0.0, cosine(query_vector, vector))
        for key, vector in zip(documents, vectors, strict=True)
    }
    dense = [
        key
        for key, score in sorted(dense_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if score > 0.0
    ][:depth]
    if mode == "dense":
        return [by_id[key] for key in dense]

    if mode == "hybrid":
        fused: dict[str, float] = {}
        for ranking in (lexical, dense):
            for position, key in enumerate(ranking, start=1):
                fused[key] = fused.get(key, 0.0) + 1.0 / (60 + position)
        ordered = sorted(fused, key=lambda key: (-fused[key], key))[:depth]
        return [by_id[key] for key in ordered]

    from context_router.domain import RouteRequest

    catalog = [
        _window_context(window, analyzer, derived_fields=derived_fields) for window in windows
    ]
    decision = ContextRouter(analyzer=analyzer, embedding_provider=embedder).route(
        RouteRequest(
            query_event_id="scale-qa",
            query=query,
            recent_events=[],
            context_catalog=catalog,
            as_of_sequence=max(window.start_block for window in windows),
        )
    )
    chosen = [by_id[key] for key in decision.selected_context_ids if key in by_id]
    if len(chosen) < depth:
        seen = {window.window_id for window in chosen}
        for candidate in decision.candidates:
            if candidate.context_id in seen or candidate.context_id not in by_id:
                continue
            chosen.append(by_id[candidate.context_id])
            seen.add(candidate.context_id)
            if len(chosen) >= depth:
                break
    return chosen[:depth]


def evaluate(
    questions: tuple[ScaleQaQuestion, ...],
    windows: list[ScaleQaWindow],
    *,
    mode: RetrievalMode,
    analyzer: LexicalAnalyzer,
    embedder: EmbeddingProvider,
    depth: int = RETRIEVAL_DEPTH,
) -> dict[str, float]:
    """Mean exact-evidence recall over the questions. Offline, deterministic, no model call."""

    if not questions:
        return {"questions": 0.0, "all_evidence_recall": 0.0, "mean_evidence_recall": 0.0}
    all_hits = 0
    partial = 0.0
    for question in questions:
        chosen = retrieve(
            windows,
            question.query,
            mode=mode,
            analyzer=analyzer,
            embedder=embedder,
            depth=depth,
        )
        assembled = "\n".join(window.text for window in chosen)
        recall = evidence_recall(question, assembled)
        partial += recall
        all_hits += int(recall == 1.0)
    return {
        "questions": float(len(questions)),
        "all_evidence_recall": all_hits / len(questions),
        "mean_evidence_recall": partial / len(questions),
    }


def difficulty_strata(
    questions: tuple[ScaleQaQuestion, ...],
    retrievers: Mapping[str, WindowRetriever],
    *,
    depth: int = RETRIEVAL_DEPTH,
    baseline: str = "lexical",
    baseline_depth: int = 1,
) -> DifficultyReport:
    """Split a benchmark by whether one cheap lookup already answers each question.

    Call this before quoting a pooled recall number, and call it on the same retrievers that
    produced that number so the split and the pooled score are the same measurement. On
    SCALE-QA's 1,000-question package the lexical baseline's first hit already contains all the
    evidence for 837 questions, and every arm scores 1.000 there; the pooled comparison is then
    decided entirely by the remaining 163.
    """

    if baseline not in retrievers:
        raise ValueError(f"baseline {baseline!r} is not among {sorted(retrievers)}")
    solved = baseline_solved_mask(
        questions, retrievers, baseline=baseline, baseline_depth=baseline_depth
    )
    solved_count = sum(solved)
    share = solved_count / len(questions) if questions else 0.0
    strata: list[DifficultyStratum] = []
    for mode, retriever in retrievers.items():
        recalls = retriever.recall_per_question(questions, depth=depth)
        pooled = _mean(recalls)
        strata.append(
            DifficultyStratum(
                mode=mode,
                questions=len(questions),
                solved_by_baseline=solved_count,
                solved_share=share,
                recall_pooled=pooled if pooled is not None else 0.0,
                recall_on_solved=_mean(
                    [r for r, is_solved in zip(recalls, solved, strict=True) if is_solved]
                ),
                recall_on_unsolved=_mean(
                    [r for r, is_solved in zip(recalls, solved, strict=True) if not is_solved]
                ),
            )
        )
    return DifficultyReport(
        baseline=baseline,
        baseline_depth=baseline_depth,
        depth=depth,
        questions=len(questions),
        solved_by_baseline=solved_count,
        solved_share=share,
        strata=tuple(strata),
    )


def routing_miss_decomposition(
    questions: tuple[ScaleQaQuestion, ...],
    retriever: WindowRetriever,
    *,
    depth: int = RETRIEVAL_DEPTH,
) -> MissDecomposition:
    """Split the router arm's misses into candidate misses and selection losses.

    Run it on the hard subset, because on a benchmark where a cheap lookup already answers most
    questions the pooled miss count is mostly a statement about the benchmark. On SCALE-QA's
    1,000-question package, over the 164 questions the lexical baseline does not answer: 103
    recovered, 37 selection losses, 24 candidate misses, and the gold sits at a median candidate
    rank of 3.
    """

    if retriever.mode != "router":
        raise ValueError(f"the decomposition needs the router arm, not {retriever.mode!r}")
    selected_counts: dict[int, int] = {}
    ranks: list[int] = []
    recovered = selection_loss = candidate_miss = unanswerable = 0
    normalized = {window.window_id: _normalize(window.text) for window in retriever.windows}
    for question in questions:
        chosen, decision = retriever.retrieve_with_decision(question.query, depth=depth)
        evidence = [_normalize(item) for item in question.expected_doc]
        gold = {key for key, text in normalized.items() if all(item in text for item in evidence)}
        if not gold:
            unanswerable += 1
        if decision is None:
            candidate_miss += 1
            continue
        selected = [key for key in decision.selected_context_ids if key in retriever.by_id]
        candidates = [c.context_id for c in decision.candidates if c.context_id in retriever.by_id]
        selected_counts[len(selected)] = selected_counts.get(len(selected), 0) + 1
        rank = next((pos for pos, key in enumerate(candidates, start=1) if key in gold), None)
        if rank is not None:
            ranks.append(rank)
        recall = evidence_recall(question, "\n".join(window.text for window in chosen))
        if recall == 1.0:
            recovered += 1
        elif set(candidates) & gold:
            selection_loss += 1
        else:
            candidate_miss += 1
    ordered = sorted(ranks)
    return MissDecomposition(
        questions=len(questions),
        recovered=recovered,
        selection_loss=selection_loss,
        candidate_miss=candidate_miss,
        no_evidence_window=unanswerable,
        selected_counts=selected_counts,
        median_candidate_rank=float(ordered[len(ordered) // 2]) if ordered else None,
    )


def _ordering_value(candidate: ContextCandidate, field: str) -> int | float | None:
    if field == "lexical_rank":
        return candidate.lexical_rank
    if field == "dense_rank":
        return candidate.dense_rank
    if field == "entity_rank":
        return candidate.entity_rank
    if field == "rrf_score":
        return candidate.rrf_score
    if field == "calibrated_probability":
        return candidate.calibrated_probability
    raise ValueError(f"unknown ordering field {field!r}")


def _ordered_candidates(
    candidates: Sequence[ContextCandidate], field: str
) -> list[ContextCandidate]:
    if field in RANK_ORDERING_FIELDS:
        return sorted(
            candidates,
            key=lambda c: (
                _ordering_value(c, field) is None,
                _ordering_value(c, field) or 0,
                c.context_id,
            ),
        )
    return sorted(candidates, key=lambda c: (-(_ordering_value(c, field) or 0.0), c.context_id))


def candidate_order_attribution(
    questions: tuple[ScaleQaQuestion, ...],
    retriever: WindowRetriever,
    *,
    depth: int = RETRIEVAL_DEPTH,
    fields: Sequence[str] = ORDERING_FIELDS,
) -> OrderingReport:
    """Recall of the router's own candidate list under each ordering field, in one router pass.

    Run it on the hard subset. On SCALE-QA's 164 unsolved questions the lexical channel scores
    0.640 over the router's own candidates, the equal-weight fusion 0.567 and the calibrated order
    0.628, against 0.848 for a perfect pick of five from the same list.
    """

    if retriever.mode != "router":
        raise ValueError(f"the attribution needs the router arm, not {retriever.mode!r}")
    normalized = {window.window_id: _normalize(window.text) for window in retriever.windows}
    totals = {field: 0.0 for field in fields}
    sizes: list[int] = []
    oracle = 0.0
    for question in questions:
        _, decision = retriever.retrieve_with_decision(question.query, depth=depth)
        if decision is None:
            continue
        candidates = [c for c in decision.candidates if c.context_id in retriever.by_id]
        sizes.append(len(candidates))
        evidence = [_normalize(item) for item in question.expected_doc]
        gold = {key for key, text in normalized.items() if all(item in text for item in evidence)}
        for field in fields:
            ordered = _ordered_candidates(candidates, field)[:depth]
            totals[field] += evidence_recall(
                question, "\n".join(retriever.documents[c.context_id] for c in ordered)
            )
        oracle += float(bool(gold & {c.context_id for c in candidates}))
    count = len(questions)
    return OrderingReport(
        questions=count,
        depth=depth,
        candidate_size_median=float(sorted(sizes)[len(sizes) // 2]) if sizes else None,
        orderings=tuple(
            OrderingAttribution(
                field=field,
                questions=count,
                recall_at_depth=totals[field] / count if count else 0.0,
            )
            for field in fields
        ),
        oracle_at_depth=oracle / count if count else 0.0,
    )


def default_embedder(analyzer: LexicalAnalyzer) -> HashEmbeddingProvider:
    """The project's placeholder, named here so a caller cannot mistake it for a real one."""

    return HashEmbeddingProvider(analyzer=analyzer)


def build_embedder(
    name: str, windows: list[ScaleQaWindow], *, analyzer: LexicalAnalyzer, dimensions: int = 256
) -> EmbeddingProvider:
    """An embedder for one window set.

    `hash` is the placeholder and makes `dense` and `hybrid` meaningless. `lsa` fits a real
    distributional embedding on the windows themselves, so the dense arm measures something; it
    is fitted per window set because the corpus is the window set. `neural` is a pinned static
    sentence encoder -- the only one of the three that is actually neural, and therefore the only
    one that can answer whether the dense channel helps.

    A pinned neural encoder is still preferable and is a one-argument swap: construct
    `OpenAICompatibleEmbeddingProvider` here and pass it through.
    """

    if name == "hash":
        return default_embedder(analyzer)
    if name == "lsa":
        from context_router.providers.embedding import LsaEmbeddingProvider

        return LsaEmbeddingProvider(
            [window.text for window in windows], dimensions=dimensions, analyzer=analyzer
        )
    if name == "neural":
        from context_router.providers.embedding import StaticNeuralEmbeddingProvider

        return StaticNeuralEmbeddingProvider(analyzer=analyzer)
    raise ValueError(f"unknown embedder {name!r}; known: hash, lsa, neural")


class WindowRetriever:
    """One window set, indexed once, answering many queries.

    `retrieve` rebuilds its BM25 index on every call, which is correct and unusable: a 512k-token
    stream windows into ~2,000 units, and 1,000 questions against it is two million document
    tokenisations. The index depends on the window set, not on the query, so it is built once
    here and the sweep becomes linear in questions.

    The router arm is still per-query expensive -- `ContextRouter.route` builds its own index over
    the context catalog on every call, which is a property of the router and not of this adapter
    -- so a caller sweeping it should cap the question count and say so in the report.
    """

    def __init__(
        self,
        windows: list[ScaleQaWindow],
        *,
        mode: RetrievalMode,
        analyzer: LexicalAnalyzer,
        embedder: EmbeddingProvider,
        derived_fields: bool = False,
    ) -> None:
        self.windows = windows
        self.mode = mode
        self.analyzer = analyzer
        self.embedder = embedder
        self.derived_fields = derived_fields
        self.by_id = {window.window_id: window for window in windows}
        self.documents = {window.window_id: window.text for window in windows}
        self.index = BM25Index(self.documents, analyzer=analyzer)
        self._vectors: list[list[float]] | None = None
        self._router: ContextRouter | None = None
        self._catalog: list[FlatContext] | None = None
        self.cache_key = (
            f"windows:{len(windows)}:{windows[0].window_id}:{windows[-1].window_id}"
            if windows
            else "windows:empty"
        )

    def _router_and_catalog(self) -> tuple[ContextRouter, list[FlatContext]]:
        if self._router is None or self._catalog is None:
            self._router = ContextRouter(
                analyzer=self.analyzer,
                embedding_provider=self.embedder,
                # The catalog is fixed for this retriever's lifetime, so one prepared entry is
                # all the cache ever needs. Without it the router rebuilds a 2,810-document BM25
                # index and re-embeds every window on every query.
                index_cache_size=1,
            )
            self._catalog = [
                _window_context(window, self.analyzer, derived_fields=self.derived_fields)
                for window in self.windows
            ]
        return self._router, self._catalog

    def retrieve(self, query: str, *, depth: int = RETRIEVAL_DEPTH) -> list[ScaleQaWindow]:
        return self.retrieve_with_decision(query, depth=depth)[0]

    def retrieve_with_decision(
        self, query: str, *, depth: int = RETRIEVAL_DEPTH
    ) -> tuple[list[ScaleQaWindow], RouteDecision | None]:
        """The units an arm would show, plus -- for `router` -- the decision that chose them.

        `retrieve` keeps only the units, and the units are all a recall number needs. They are not
        enough to say *why* a miss happened: the decision is what separates "the candidate list
        never held the evidence" from "it held it and something else was returned". Returning it
        costs nothing, because the routing has already happened by then.
        """

        if not self.windows:
            return [], None
        lexical = [key for key, _ in self.index.rank(query, depth)]
        if self.mode == "lexical":
            return [self.by_id[key] for key in lexical], None

        if self.mode in ("dense", "hybrid"):
            query_vector = self.embedder.embed([query])[0]
            # Embedded once and reused. Recomputing this per query was 2.8 million embeddings for
            # a 1,000-question sweep over 2,810 windows, which is what made the dense and hybrid
            # arms appear to hang.
            if self._vectors is None:
                self._vectors = self.embedder.embed([self.documents[key] for key in self.documents])
            vectors = self._vectors
            scores = {
                key: max(0.0, cosine(query_vector, vector))
                for key, vector in zip(self.documents, vectors, strict=True)
            }
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            dense = [key for key, score in ranked if score > 0.0][:depth]
            if self.mode == "dense":
                return [self.by_id[key] for key in dense], None
            fused: dict[str, float] = {}
            for ranking in (lexical, dense):
                for position, key in enumerate(ranking, start=1):
                    fused[key] = fused.get(key, 0.0) + 1.0 / (60 + position)
            ordered = sorted(fused, key=lambda key: (-fused[key], key))[:depth]
            return [self.by_id[key] for key in ordered], None

        from context_router.domain import RouteRequest

        router, catalog = self._router_and_catalog()
        decision = router.route(
            RouteRequest(
                query_event_id="scale-qa",
                query=query,
                recent_events=[],
                context_catalog=catalog,
                as_of_sequence=max(window.start_block for window in self.windows),
            ),
            cache_key=self.cache_key,
        )
        chosen = [self.by_id[key] for key in decision.selected_context_ids if key in self.by_id]
        seen = {window.window_id for window in chosen}
        for candidate in decision.candidates:
            if len(chosen) >= depth:
                break
            if candidate.context_id in seen or candidate.context_id not in self.by_id:
                continue
            chosen.append(self.by_id[candidate.context_id])
            seen.add(candidate.context_id)
        return chosen[:depth], decision

    def recall_per_question(
        self, questions: tuple[ScaleQaQuestion, ...], *, depth: int = RETRIEVAL_DEPTH
    ) -> list[float]:
        """Per-question exact-evidence recall, so a caller can stratify instead of pooling.

        `evaluate` is this list averaged. Reporting the pooled mean alone is what hides a benchmark
        whose difficulty sits somewhere the arm's claimed advantage does not; keeping the question
        axis lets `difficulty_strata` split it without running the retrieval a second time.
        """

        recalls: list[float] = []
        for question in questions:
            chosen = self.retrieve(question.query, depth=depth)
            assembled = "\n".join(window.text for window in chosen)
            recalls.append(evidence_recall(question, assembled))
        return recalls

    def evaluate(
        self, questions: tuple[ScaleQaQuestion, ...], *, depth: int = RETRIEVAL_DEPTH
    ) -> dict[str, float]:
        if not questions:
            return {"questions": 0.0, "all_evidence_recall": 0.0, "mean_evidence_recall": 0.0}
        recalls = self.recall_per_question(questions, depth=depth)
        return {
            "questions": float(len(recalls)),
            "all_evidence_recall": sum(1 for recall in recalls if recall == 1.0) / len(recalls),
            "mean_evidence_recall": sum(recalls) / len(recalls),
        }
