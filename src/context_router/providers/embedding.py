from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from context_router.retrieval import LexicalAnalyzer

DEFAULT_STATIC_MODEL = "minishlab/potion-base-8M"
DEFAULT_STATIC_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"


class EmbeddingProvider(Protocol):
    model_version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic offline embedding adapter for tests and reproducible smoke runs."""

    model_version = "hash-embedding-v1"

    def __init__(self, dimensions: int = 256, analyzer: LexicalAnalyzer | None = None) -> None:
        self.dimensions = dimensions
        self.analyzer = analyzer or LexicalAnalyzer()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self.analyzer.tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


class LsaEmbeddingProvider:
    """A fitted distributional embedding: real semantics, offline, deterministic.

    `HashEmbeddingProvider` is a placeholder with no semantics at all, which is why this project
    has never measured its dense path (limitation 2 in the README). A neural encoder would be the
    obvious replacement, but the obvious ones need either a model download or an API endpoint, and
    neither is available in an offline run.

    This fits TF-IDF followed by a truncated SVD on a corpus and normalises the result. That is a
    genuine distributional embedding -- terms that occur in similar contexts land near each other,
    which is exactly what hashing cannot do -- and it is deterministic given the corpus.

    Two things it is not, and both matter for reading a result produced with it. It is not a
    neural sentence encoder, so it does not carry the paraphrase behaviour a transformer does. And
    it is **fitted on the corpus it is then used to retrieve over**, which is the same
    characteristic TF-IDF itself has and means the dense arm is not measuring transfer to unseen
    text. A real pinned encoder is still the right answer; `OpenAICompatibleEmbeddingProvider`
    takes one, and swapping to it is a one-argument change in `external/scale_qa.py`.
    """

    def __init__(
        self,
        corpus: Sequence[str],
        *,
        dimensions: int = 256,
        analyzer: LexicalAnalyzer | None = None,
    ) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer

        self.analyzer = analyzer or LexicalAnalyzer()
        documents = [text for text in corpus if text.strip()]
        if not documents:
            raise ValueError("a non-empty corpus is required to fit the embedding")
        vectorizer = TfidfVectorizer(
            tokenizer=self.analyzer.tokens,
            token_pattern=None,
            lowercase=False,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(documents)
        # A truncated SVD cannot keep more components than the matrix has columns. Clamping here
        # rather than raising keeps a small corpus usable -- which a test fixture and a small
        # window set both are -- and the effective width is what `model_version` reports, so a
        # result never claims a dimension it did not use.
        effective = max(1, min(dimensions, matrix.shape[1] - 1))
        self.dimensions = effective
        self.model_version = f"lsa-tfidf-svd{effective}-v1"
        self._model = make_pipeline(
            TfidfVectorizer(
                tokenizer=self.analyzer.tokens,
                token_pattern=None,
                lowercase=False,
                min_df=1,
            ),
            TruncatedSVD(n_components=effective, random_state=0),
            Normalizer(copy=False),
        )
        self._model.fit(documents)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(value) for value in row] for row in self._model.transform(texts)]


class StaticNeuralEmbeddingProvider:
    """A pinned neural encoder that needs no GPU and no API.

    Wraps a `model2vec` static embedding model: a sentence encoder distilled into a lookup table,
    so it runs on CPU at tokenisation speed while keeping the encoder's semantics. It is a real
    neural embedding, which is what `HashEmbeddingProvider` is not and what `LsaEmbeddingProvider`
    approximates without being one -- this is the provider that answers whether the dense channel
    helps, rather than measuring a placeholder or a bag-of-words factorisation.

    The model is fetched once and cached locally. Where the model hub is unreachable, set
    ``HF_ENDPOINT`` to a mirror before constructing this; the default hub times out from some
    networks and a mirror is reachable from the same machine.

    Two limits worth stating with any result it produces. The default model is English-centric,
    so it is a fair encoder for an English benchmark and not for this project's mixed-language
    replay data. And it is *static*: no attention, so it does not carry word-order or long-range
    behaviour a transformer does. It is a much better dense retriever than hashing and a weaker
    one than the encoder it was distilled from.
    """

    def __init__(
        self,
        model: str = DEFAULT_STATIC_MODEL,
        *,
        analyzer: LexicalAnalyzer | None = None,
        revision: str | None = None,
    ) -> None:
        try:
            from model2vec import StaticModel
        except ImportError as error:  # pragma: no cover - depends on the optional extra
            raise ImportError(
                "the neural embedder needs the optional extra: pip install -e '.[neural]'"
            ) from error
        self.analyzer = analyzer or LexicalAnalyzer()
        model_path = model
        if revision is not None:
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError("embedding revision must be an immutable 40-character commit")
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                model,
                revision=revision,
                allow_patterns=["config.json", "model.safetensors", "tokenizer.json"],
            )
        self._model = StaticModel.from_pretrained(model_path, force_download=False)
        self.model_version = f"static-neural:{model}" + (f"@{revision}" if revision else "")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(value) for value in row] for row in self._model.encode(texts)]
