from __future__ import annotations

import math
import re
from collections import Counter

import jieba

jieba.setLogLevel(30)

_RAW_PART = re.compile(r"[A-Za-z_][A-Za-z0-9_./\\:-]*|[\u3400-\u4dbf\u4e00-\u9fff]+|\d+(?:\.\d+)*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CJK = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


class LexicalAnalyzer:
    """Tokenize Chinese, English and code without discarding exact identifiers."""

    def tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in _RAW_PART.findall(text):
            lowered = raw.lower()
            tokens.append(lowered)
            if _CJK.match(raw):
                words = [word.strip() for word in jieba.cut(raw) if word.strip()]
                tokens.extend(word for word in words if word != raw)
                tokens.extend(raw[index : index + 2] for index in range(len(raw) - 1))
                continue
            normalized = raw.replace("\\", "/")
            segments = re.split(r"[\s_./:\-]+", normalized)
            for segment in segments:
                for part in _CAMEL.split(segment):
                    part = part.lower()
                    if part and part != lowered:
                        tokens.append(part)
        return list(dict.fromkeys(tokens))


class BM25Index:
    def __init__(
        self,
        documents: dict[str, str],
        *,
        analyzer: LexicalAnalyzer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.analyzer = analyzer or LexicalAnalyzer()
        self.k1 = k1
        self.b = b
        self.documents = {key: self.analyzer.tokens(value) for key, value in documents.items()}
        self.term_frequencies = {key: Counter(tokens) for key, tokens in self.documents.items()}
        self.average_length = sum(len(tokens) for tokens in self.documents.values()) / max(
            len(self.documents), 1
        )
        document_frequency: Counter[str] = Counter()
        for tokens in self.documents.values():
            document_frequency.update(set(tokens))
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        query_terms = self.analyzer.tokens(query)
        results: dict[str, float] = {}
        for document_id, tokens in self.documents.items():
            frequencies = self.term_frequencies[document_id]
            length_norm = 1.0 - self.b + self.b * len(tokens) / max(self.average_length, 1.0)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
                )
            results[document_id] = score
        return results

    def rank(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda item: (-item[1], item[0]))
        return [item for item in ranked if item[1] > 0.0][:limit]
