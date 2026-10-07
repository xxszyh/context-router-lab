"""Reproducible session-level LongMemEval evaluation, without answer-label leakage.

Only content enters retrieval. answer, answer_session_ids and has_answer are
evaluation labels. All-gold recall, not any-gold recall, is the primary metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import ijson

from context_router.domain import FlatContext, RouteRequest
from context_router.external.scale_qa import (
    OrderingHits,
    is_established,
    paired_ordering_comparison,
)
from context_router.providers.embedding import EmbeddingProvider, cosine
from context_router.providers.reranking import PairScorer
from context_router.retrieval import BM25Index
from context_router.routing import ContextRouter
from context_router.routing.router import RoutingPolicy

HARD_TYPES = ("multi-session", "temporal-reasoning", "single-session-preference")


def iter_instances(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the large JSON array; never hold the whole benchmark in RAM."""
    with path.open("rb") as stream:
        if not stream.read(4096).lstrip().startswith(b"["):
            raise ValueError("LongMemEval must be a JSON array")
        stream.seek(0)
        try:
            for instance in ijson.items(stream, "item"):
                if not isinstance(instance, dict):
                    raise ValueError("each benchmark instance must be an object")
                yield instance
        except ijson.JSONError as error:
            raise ValueError("invalid or truncated LongMemEval JSON") from error


def session_text(turns: list[dict[str, Any]]) -> str:
    # Preserve the old experiment's text representation. All other keys,
    # particularly has_answer, stay out of the retrieval input.
    return "\n".join(str(turn.get("content", "")) for turn in turns)


def validate_instance(instance: dict[str, Any]) -> None:
    for key in ("question_id", "question_type", "question"):
        if not isinstance(instance.get(key), str) or not instance[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    ids = instance.get("haystack_session_ids")
    sessions = instance.get("haystack_sessions")
    gold = instance.get("answer_session_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(key, str) for key in ids):
        raise ValueError("haystack_session_ids must be a non-empty list of strings")
    if not isinstance(sessions, list) or len(sessions) != len(ids):
        raise ValueError("session ids and sessions must align")
    for session in sessions:
        if not isinstance(session, list) or not all(isinstance(turn, dict) for turn in session):
            raise ValueError("each session must be a list of turn objects")
        if any(not isinstance(turn.get("content"), str) for turn in session):
            raise ValueError("turn content must be a string")
    # The official cleaned S release repeats identical sessions in seven of the
    # 296 hard instances. Collapse copies, but reject conflicting content under
    # one label instead of choosing a copy that flatters retrieval.
    copies: dict[str, str] = {}
    for key, turns in zip(ids, sessions, strict=True):
        visible = json.dumps(
            [(turn.get("role"), turn["content"]) for turn in turns], ensure_ascii=False
        )
        if key in copies and copies[key] != visible:
            raise ValueError("conflicting duplicate haystack_session_ids")
        copies[key] = visible
    if not isinstance(gold, list) or not all(isinstance(key, str) for key in gold):
        raise ValueError("answer_session_ids must be a list of strings")
    if not set(gold) <= set(ids):
        # Never silently intersect away a missing gold session and improve recall.
        raise ValueError("answer_session_ids contains a session missing from the haystack")


def evidence_metrics(gold: set[str], retrieved: list[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("empty gold is an abstention case, not a retrieval success")
    found = len(gold & set(retrieved))
    return {
        "any": float(found > 0),
        "all": float(gold <= set(retrieved)),
        "coverage": found / len(gold),
    }


def session_passages(text: str, *, words: int = 180, overlap: int = 40) -> list[str]:
    if words < 1 or not 0 <= overlap < words:
        raise ValueError("passage words must be positive and overlap smaller than words")
    tokens = text.split()
    passages = []
    for start in range(0, len(tokens), words - overlap):
        passages.append(" ".join(tokens[start : start + words]))
        if start + words >= len(tokens):
            break
    return passages or [""]


def rerank_sessions(
    query: str,
    documents: dict[str, str],
    candidates: list[str],
    scorer: PairScorer,
    *,
    passages_per_session: int = 2,
    passage_words: int = 180,
    passage_overlap: int = 40,
) -> tuple[list[str], dict[str, float]]:
    """Score bounded, query-selected passages, then max-pool by session.

    Selection uses every candidate's passages as one corpus. Neither the gold set
    nor a turn's has_answer annotation is available to this function.
    """
    if passages_per_session < 1:
        raise ValueError("passages_per_session must be positive")
    if len(set(candidates)) != len(candidates):
        raise ValueError("duplicate reranking candidates")
    chunks = {
        key: session_passages(documents[key], words=passage_words, overlap=passage_overlap)
        for key in candidates
    }
    passages = {
        f"{i}:{j}": text for i, key in enumerate(candidates) for j, text in enumerate(chunks[key])
    }
    scores = BM25Index(passages).scores(query)
    texts: list[str] = []
    owners: list[str] = []
    for i, key in enumerate(candidates):
        ordered = sorted(range(len(chunks[key])), key=lambda j: (-scores[f"{i}:{j}"], j))
        for j in ordered[:passages_per_session]:
            texts.append(chunks[key][j])
            owners.append(key)
    values = scorer.score(query, texts)
    if len(values) != len(texts) or not all(math.isfinite(value) for value in values):
        raise ValueError("reranker must return one finite score per passage")
    aggregated: dict[str, float] = {}
    for key, value in zip(owners, values, strict=True):
        aggregated[key] = max(value, aggregated.get(key, -math.inf))
    return sorted(candidates, key=lambda key: (-aggregated[key], key)), aggregated


class _CachingEmbedder:
    def __init__(self, inner: EmbeddingProvider) -> None:
        self.inner = inner
        self.model_version = inner.model_version
        self.cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            self.cache.update(zip(missing, self.inner.embed(missing), strict=True))
        return [self.cache[text] for text in texts]


def evaluate_instance(
    instance: dict[str, Any],
    embedder: EmbeddingProvider,
    *,
    depth: int = 5,
    policy: RoutingPolicy | None = None,
    scorer: PairScorer | None = None,
    passages_per_session: int = 2,
    allow_abstention: bool = False,
) -> dict[str, Any]:
    validate_instance(instance)
    if not 1 <= depth <= 20:
        raise ValueError("depth must be between 1 and 20")
    gold = set(instance["answer_session_ids"])
    abstention = not gold or instance["question_id"].endswith("_abs")
    if abstention and not allow_abstention:
        raise ValueError("abstention cases must be evaluated separately")
    query = instance["question"]
    ids = instance["haystack_session_ids"]
    texts = [session_text(session) for session in instance["haystack_sessions"]]
    documents = dict(zip(ids, texts, strict=True))
    duplicate_sessions = len(ids) - len(documents)
    ids = list(documents)
    texts = list(documents.values())
    cached = _CachingEmbedder(embedder)
    lexical = [key for key, _ in BM25Index(documents).rank(query, len(ids))]
    vectors = cached.embed(texts)
    query_vector = cached.embed([query])[0]
    dense_scores = {
        key: max(0.0, cosine(query_vector, vector))
        for key, vector in zip(ids, vectors, strict=True)
    }
    dense = sorted(
        (key for key in ids if dense_scores[key] > 0), key=lambda key: (-dense_scores[key], key)
    )
    router = ContextRouter(embedding_provider=cached, policy=policy)
    catalog = [
        FlatContext(
            context_id=key,
            name="",
            goal="",
            summary=text,
            status="active",
            created_at_event=key,
            last_active_sequence=i,
            version=1,
        )
        for i, (key, text) in enumerate(zip(ids, texts, strict=True))
    ]
    decision = router.route(
        RouteRequest(
            query_event_id=instance["question_id"],
            query=query,
            recent_events=[],
            context_catalog=catalog,
            as_of_sequence=len(ids),
            max_selected_contexts=depth,
        )
    )
    candidates = [candidate.context_id for candidate in decision.candidates]
    fused = sorted(
        decision.candidates, key=lambda candidate: (-candidate.rrf_score, candidate.context_id)
    )
    hybrid = [candidate.context_id for candidate in fused]
    selected = list(dict.fromkeys([*decision.selected_context_ids, *candidates]))
    orderings = {"lexical": lexical, "dense": dense, "hybrid": hybrid, "router": selected}
    rerank_scores: dict[str, float] = {}
    if scorer is not None:
        joint, rerank_scores = rerank_sessions(
            query, documents, candidates, scorer, passages_per_session=passages_per_session
        )
        orderings["joint"] = joint
    arms = {
        name: {
            "retrieved_ids": ordered[:depth],
            "metrics": None if abstention else evidence_metrics(gold, ordered[:depth]),
        }
        for name, ordered in orderings.items()
    }
    record = {
        "question_id": instance["question_id"],
        "question_type": instance["question_type"],
        "gold_session_ids": sorted(gold),
        "haystack_sessions": len(ids),
        "duplicate_sessions_collapsed": duplicate_sessions,
        "candidate_ids": candidates,
        "candidate_all_gold": None if abstention else gold <= set(candidates),
        "oracle_all_at_depth": None
        if abstention
        else len(gold) <= depth and gold <= set(candidates),
        "rerank_scores": rerank_scores,
        "arms": arms,
    }
    if abstention:
        record["expects_refusal"] = True
    return record


def summarize_records(records: list[dict[str, Any]], *, seed: int = 20260420) -> dict[str, Any]:
    if not records:
        raise ValueError("no answerable questions matched the requested types and limit")
    names = list(records[0]["arms"])

    def arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            name: {
                metric: statistics.mean(row["arms"][name]["metrics"][metric] for row in rows)
                for metric in ("any", "all", "coverage")
            }
            for name in names
        }

    fields = {
        name: [int(record["arms"][name]["metrics"]["all"]) for record in records] for name in names
    }
    pairs = [("router", "lexical"), ("hybrid", "lexical"), ("router", "hybrid")]
    if "joint" in names:
        pairs += [("joint", "hybrid"), ("joint", "router")]
    comparisons = paired_ordering_comparison(
        OrderingHits(len(records), fields, [], None), pairs, seed=seed
    )
    return {
        "questions": len(records),
        "arms": arm_metrics(records),
        "by_type": {
            kind: {"questions": len(rows), "arms": arm_metrics(rows)}
            for kind in sorted({record["question_type"] for record in records})
            if (rows := [record for record in records if record["question_type"] == kind])
        },
        "candidate_all_gold": statistics.mean(int(r["candidate_all_gold"]) for r in records),
        "oracle_all_at_depth": statistics.mean(int(r["oracle_all_at_depth"]) for r in records),
        "decomposition": {
            name: {
                "recovered": sum(int(r["arms"][name]["metrics"]["all"]) for r in records),
                "selection_loss": sum(
                    int(r["candidate_all_gold"] and not r["arms"][name]["metrics"]["all"])
                    for r in records
                ),
                "candidate_miss": sum(int(not r["candidate_all_gold"]) for r in records),
            }
            for name in ("router", "hybrid", "joint")
            if name in names
        },
        "paired_all_gold": [
            {**asdict(pair), "established": is_established(pair)} for pair in comparisons
        ],
    }


def evaluate_longmemeval(
    path: Path,
    embedder: EmbeddingProvider,
    *,
    depth: int = 5,
    types: tuple[str, ...] = HARD_TYPES,
    limit: int | None = None,
    policy: RoutingPolicy | None = None,
    scorer: PairScorer | None = None,
    passages_per_session: int = 2,
    seed: int = 20260420,
    progress: Callable[[int], None] | None = None,
    checkpoint: Path | None = None,
    resume: bool = False,
    include_abstention: bool = False,
    abstention_only: bool = False,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    policy = policy or RoutingPolicy()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    configuration = {
        "dataset_sha256": digest,
        "depth": depth,
        "types": list(types),
        "policy": asdict(policy),
        "embedding": embedder.model_version,
        "reranker": scorer.model_version if scorer else None,
        "passages_per_session": passages_per_session if scorer else None,
        "implementation": "longmemeval-session-v1",
    }
    records: list[dict[str, Any]] = []
    abstention_records: list[dict[str, Any]] = []
    include_abstention = include_abstention or abstention_only
    seen: set[str] = set()
    skipped_abstention = 0
    with ExitStack() as stack:
        saved: dict[str, dict[str, Any]] = {}
        journal = None
        if resume and checkpoint is None:
            raise ValueError("resume requires a checkpoint")
        if checkpoint is not None:
            if checkpoint.resolve() == path.resolve():
                raise ValueError("checkpoint must not overwrite the source dataset")
            exists = checkpoint.exists()
            if exists and not resume:
                raise ValueError("checkpoint exists; use --resume or choose a new output")
            if exists:
                with checkpoint.open("rb") as source:
                    header = json.loads(source.readline())
                    if not isinstance(header, dict) or header.get("configuration") != configuration:
                        raise ValueError("checkpoint configuration or dataset changed")
                    complete_end = source.tell()
                    for line in source:
                        # A killed write can leave an incomplete UTF-8/JSON tail.
                        # Only newline-terminated records are committed.
                        if not line.endswith(b"\n"):
                            break
                        record = json.loads(line)
                        key = record["question_id"]
                        if key in saved:
                            raise ValueError("duplicate question_id in checkpoint")
                        saved[key] = record
                        complete_end += len(line)
                if complete_end != checkpoint.stat().st_size:
                    with checkpoint.open("r+b") as repair:
                        repair.truncate(complete_end)
            elif resume:
                raise ValueError("checkpoint does not exist")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            journal = stack.enter_context(checkpoint.open("a" if exists else "x", encoding="utf-8"))
            if not exists:
                journal.write(json.dumps({"configuration": configuration}) + "\n")
                journal.flush()
        for instance in iter_instances(path):
            if instance.get("question_type") not in types:
                continue
            validate_instance(instance)
            key = instance["question_id"]
            if key in seen:
                raise ValueError("duplicate question_id")
            seen.add(key)
            abstention = not instance["answer_session_ids"] or key.endswith("_abs")
            if abstention_only and not abstention:
                continue
            if abstention and not include_abstention:
                skipped_abstention += 1
                continue
            record = saved.get(key)
            if record is None:
                record = evaluate_instance(
                    instance,
                    embedder,
                    depth=depth,
                    policy=policy,
                    scorer=scorer,
                    passages_per_session=passages_per_session,
                    allow_abstention=abstention,
                )
                if journal is not None:
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()
            (abstention_records if abstention else records).append(record)
            count = len(records) + len(abstention_records)
            if progress is not None:
                progress(count)
            if limit is not None and count >= limit:
                break
    if not records and not abstention_records:
        category = "eligible" if include_abstention else "answerable"
        raise ValueError(f"no {category} questions matched the requested types and limit")
    result = {
        "schema_version": "1.0",
        "benchmark": "LongMemEval",
        "dataset_sha256": digest,
        "unit": "session",
        "primary_metric": "all_gold_session_recall",
        "answer_quality": "not_measured",
        "depth": depth,
        "types": list(types),
        "limit": limit,
        "skipped_abstention": skipped_abstention,
        "seed": seed,
        "implementation": configuration["implementation"],
        "policy": asdict(policy),
        "models": {
            "embedding": embedder.model_version,
            "reranker": scorer.model_version if scorer else None,
        },
        "passages_per_session": passages_per_session if scorer else None,
        "passage_words": 180 if scorer else None,
        "passage_overlap_words": 40 if scorer else None,
        "summary": summarize_records(records, seed=seed)
        if records
        else {"questions": 0, "arms": {}},
        "records": records,
    }
    if include_abstention:
        result.update(
            abstention_records=abstention_records,
            abstention_questions=len(abstention_records),
            question_scope="abstention_only" if abstention_only else "answerable_and_abstention",
        )
    return result


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Replace a complete report atomically; readers never see half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
