from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from context_router.domain import (
    ContextCandidate,
    Decision,
    FlatContext,
    Relation,
    RouteDecision,
    RouteRequest,
    new_uuid7,
)
from context_router.providers.embedding import EmbeddingProvider, HashEmbeddingProvider, cosine
from context_router.retrieval import BM25Index, LexicalAnalyzer
from context_router.routing.calibration import (
    CandidateRanker,
    HeuristicContextRanker,
)
from context_router.routing.relation import RuleRelationClassifier


@dataclass(frozen=True)
class RoutingPolicy:
    top_per_retriever: int = 10
    max_candidates: int = 20
    rrf_k: int = 60
    t_low: float = 0.35
    t_high: float = 0.72
    margin: float = 0.16
    confidence_threshold: float = 0.52


def _rank_map(values: list[tuple[str, float]]) -> dict[str, int]:
    return {context_id: index for index, (context_id, _) in enumerate(values, start=1)}


def _normalize(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in values}
    return {key: max(0.0, value) / maximum for key, value in values.items()}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class ContextRouter:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        relation_classifier: RuleRelationClassifier | None = None,
        analyzer: LexicalAnalyzer | None = None,
        policy: RoutingPolicy | None = None,
        ranker: CandidateRanker | None = None,
    ) -> None:
        self.analyzer = analyzer or LexicalAnalyzer()
        self.embedding_provider = embedding_provider or HashEmbeddingProvider(
            analyzer=self.analyzer
        )
        self.relation_classifier = relation_classifier or RuleRelationClassifier()
        self.policy = policy or RoutingPolicy()
        self.ranker = ranker or HeuristicContextRanker()

    def route(self, request: RouteRequest) -> RouteDecision:
        relation, relation_probabilities = self.relation_classifier.classify(request)
        contexts = {context.context_id: context for context in request.context_catalog}
        if not contexts:
            return self._empty_decision(relation, relation_probabilities)

        # The retrieval query is the user's query and nothing else. The recent window is a
        # coreference signal that belongs in the reranking features, never in retrieval:
        # concatenating it here lets the context being left dominate BM25 and the dense
        # channel, which is precisely the context stickiness the plan warns about.
        retrieval_query = request.query
        lexical_index = BM25Index(
            {key: value.searchable_text() for key, value in contexts.items()},
            analyzer=self.analyzer,
        )
        lexical_values = lexical_index.rank(retrieval_query, self.policy.top_per_retriever)
        lexical_scores = dict(lexical_values)

        query_vector = self.embedding_provider.embed([retrieval_query])[0]
        context_ids = list(contexts)
        context_vectors = self.embedding_provider.embed(
            [contexts[context_id].searchable_text() for context_id in context_ids]
        )
        dense_scores = {
            context_id: max(0.0, cosine(query_vector, vector))
            for context_id, vector in zip(context_ids, context_vectors, strict=True)
        }
        dense_values = sorted(dense_scores.items(), key=lambda item: (-item[1], item[0]))
        dense_values = [item for item in dense_values if item[1] > 0.0][
            : self.policy.top_per_retriever
        ]

        entity_scores = self._entity_scores(request.query, contexts)
        entity_values = sorted(entity_scores.items(), key=lambda item: (-item[1], item[0]))
        entity_values = [item for item in entity_values if item[1] > 0.0][
            : self.policy.top_per_retriever
        ]

        dense_ranks = _rank_map(dense_values)
        lexical_ranks = _rank_map(lexical_values)
        entity_ranks = _rank_map(entity_values)
        candidate_ids = set(dense_ranks) | set(lexical_ranks) | set(entity_ranks)
        if request.primary_context_id in contexts:
            candidate_ids.add(request.primary_context_id)
        candidate_ids.update(key for key in request.recent_context_ids if key in contexts)

        rrf_scores = {
            context_id: sum(
                1.0 / (self.policy.rrf_k + ranks[context_id])
                for ranks in (dense_ranks, lexical_ranks, entity_ranks)
                if context_id in ranks
            )
            for context_id in candidate_ids
        }
        ordered_ids = sorted(candidate_ids, key=lambda key: (-rrf_scores[key], key))[
            : self.policy.max_candidates
        ]
        rrf_normalized = _normalize(rrf_scores)
        dense_normalized = _normalize(dense_scores)
        lexical_normalized = _normalize(lexical_scores)
        entity_normalized = _normalize(entity_scores)

        candidates: list[ContextCandidate] = []
        for context_id in ordered_ids:
            context = contexts[context_id]
            features = self._features(
                context=context,
                context_id=context_id,
                request=request,
                relation=relation,
                rrf=rrf_normalized.get(context_id, 0.0),
                dense=dense_normalized.get(context_id, 0.0),
                lexical=lexical_normalized.get(context_id, 0.0),
                entity=entity_normalized.get(context_id, 0.0),
                agreement=sum(
                    context_id in ranks for ranks in (dense_ranks, lexical_ranks, entity_ranks)
                )
                / 3.0,
            )
            probability = self.ranker.probability(features)
            reasons = [
                name
                for name, ranks in (
                    ("dense", dense_ranks),
                    ("lexical", lexical_ranks),
                    ("entity", entity_ranks),
                )
                if context_id in ranks
            ]
            if features["primary"]:
                reasons.append("primary")
            if features["recent"]:
                reasons.append("recent")
            candidates.append(
                ContextCandidate(
                    context_id=context_id,
                    dense_rank=dense_ranks.get(context_id),
                    lexical_rank=lexical_ranks.get(context_id),
                    entity_rank=entity_ranks.get(context_id),
                    rrf_score=rrf_scores[context_id],
                    calibrated_probability=probability,
                    feature_values=features,
                    reasons=reasons,
                )
            )
        candidates.sort(key=lambda item: (-item.calibrated_probability, item.context_id))
        selected, confidence, decision = self._select(request, candidates, relation)
        return RouteDecision(
            decision=decision,
            relation=relation,
            relation_probabilities=relation_probabilities,
            candidates=candidates,
            selected_context_ids=selected,
            confidence=confidence,
            fallback_level=0 if decision == "route" else 1,
            trace_id=new_uuid7(),
            index_version=self._index_version(request.context_catalog),
            model_versions={
                "embedding": self.embedding_provider.model_version,
                "relation": self.relation_classifier.model_version,
                "ranker": self.ranker.model_version,
            },
        )

    def _entity_scores(self, query: str, contexts: dict[str, FlatContext]) -> dict[str, float]:
        lowered = query.casefold()
        query_tokens = set(self.analyzer.tokens(query))
        scores: dict[str, float] = {}
        for context_id, context in contexts.items():
            matches = 0.0
            for entity in context.entities:
                entity_lowered = entity.casefold()
                entity_tokens = set(self.analyzer.tokens(entity))
                if entity_lowered and entity_lowered in lowered:
                    matches += 1.0
                elif entity_tokens and entity_tokens <= query_tokens:
                    matches += 0.7
            scores[context_id] = matches / max(len(context.entities), 1)
        return scores

    @staticmethod
    def _features(
        *,
        context: FlatContext,
        context_id: str,
        request: RouteRequest,
        relation: str,
        rrf: float,
        dense: float,
        lexical: float,
        entity: float,
        agreement: float,
    ) -> dict[str, float]:
        primary = float(context_id == request.primary_context_id)
        recent = float(context_id in request.recent_context_ids)
        delta = max(request.as_of_sequence - context.last_active_sequence, 0)
        recency = math.exp(-delta / 20.0)
        # A return reaches a context that is by construction not among the recent ones, so
        # rewarding `recent` for `switch_or_return` rewards the opposite of the answer.
        # Only cues that genuinely point at the live context are kept.
        relation_match = 0.0
        if relation == "continue" and primary:
            relation_match = 1.0
        elif relation == "cross_context" and (primary or recent):
            relation_match = 0.6
        return {
            "rrf": rrf,
            "dense": dense,
            "lexical": lexical,
            "entity": entity,
            "agreement": agreement,
            "primary": primary,
            "recent": recent,
            "recency": recency,
            "relation_match": relation_match,
        }

    def _select(
        self,
        request: RouteRequest,
        candidates: list[ContextCandidate],
        relation: Relation,
    ) -> tuple[list[str], float, Decision]:
        if not candidates:
            return [], 0.0, "abstain"
        first = candidates[0].calibrated_probability
        second = candidates[1].calibrated_probability if len(candidates) > 1 else 0.0
        margin = first - second
        relation_confidence = 1.0 if relation != "unknown" else 0.35
        agreement = candidates[0].feature_values["agreement"]
        confidence = max(
            0.0,
            min(
                1.0,
                0.50 * first
                + 0.20 * min(margin / 0.35, 1.0)
                + 0.15 * agreement
                + 0.15 * relation_confidence,
            ),
        )

        maximum = min(request.max_selected_contexts, self.policy.max_candidates)
        # Only a relation that asserts a single continuation context may collapse to the
        # top-1. `cross_context` needs several contexts by definition, and `unknown` is an
        # admission of ignorance: collapsing it to one confident-looking context turns a
        # relation-classifier miss into silently discarded evidence.
        single_context = relation in ("continue", "switch_or_return")
        if first >= self.policy.t_high and margin >= self.policy.margin and single_context:
            selected = [candidates[0].context_id]
        else:
            selected = [
                candidate.context_id
                for candidate in candidates
                if candidate.calibrated_probability >= self.policy.t_low
            ][:maximum]

        if relation == "cross_context" and len(candidates) >= 2 and maximum >= 2:
            for candidate in candidates[:2]:
                if candidate.context_id not in selected:
                    selected.append(candidate.context_id)
            selected = selected[:maximum]

        if relation == "new_context" and first < 0.80:
            return [], min(confidence, 0.79), "new_context_candidate"
        if not selected or confidence < self.policy.confidence_threshold:
            safe = [candidate.context_id for candidate in candidates[:maximum]]
            return safe, confidence, "abstain"
        return selected, confidence, "route"

    def _empty_decision(self, relation: Relation, probabilities: dict[str, float]) -> RouteDecision:
        decision: Decision = "new_context_candidate" if relation == "new_context" else "abstain"
        return RouteDecision(
            decision=decision,
            relation=relation,
            relation_probabilities=probabilities,
            candidates=[],
            selected_context_ids=[],
            confidence=0.0,
            fallback_level=1,
            trace_id=new_uuid7(),
            index_version="empty",
            model_versions={
                "embedding": self.embedding_provider.model_version,
                "relation": self.relation_classifier.model_version,
                "ranker": self.ranker.model_version,
            },
        )

    @staticmethod
    def _index_version(contexts: list[FlatContext]) -> str:
        payload = [context.model_dump(mode="json") for context in contexts]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


_DEFAULT_ROUTER = ContextRouter()


def route(request: RouteRequest) -> RouteDecision:
    """Route through the default deterministic profile."""

    return _DEFAULT_ROUTER.route(request)
