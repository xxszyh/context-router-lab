from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_longmemeval import instance

from context_router.external.longmemeval import evaluate_longmemeval
from context_router.providers.embedding import HashEmbeddingProvider


def test_checkpoint_survives_interruption_and_resumes_without_recomputing(tmp_path: Path) -> None:
    source = tmp_path / "dataset.json"
    first = instance()
    second = instance()
    second["question_id"] = "q2"
    source.write_text(json.dumps([first, second]), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.jsonl"

    def interrupt(count: int) -> None:
        raise RuntimeError("interrupted after durable result")

    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_longmemeval(
            source, HashEmbeddingProvider(), checkpoint=checkpoint, progress=interrupt
        )
    assert len(checkpoint.read_text().splitlines()) == 2
    # Hard termination can cut a record before its newline, even during UTF-8.
    with checkpoint.open("ab") as partial:
        partial.write(b'{"question_id":"q2","partial":"\xe4')

    class CountingEmbedder(HashEmbeddingProvider):
        calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return super().embed(texts)

    embedder = CountingEmbedder()
    resumed = evaluate_longmemeval(source, embedder, checkpoint=checkpoint, resume=True)
    assert resumed["summary"]["questions"] == 2
    assert len(checkpoint.read_text().splitlines()) == 3
    calls = embedder.calls
    again = evaluate_longmemeval(source, embedder, checkpoint=checkpoint, resume=True)
    assert embedder.calls == calls
    assert resumed == again
    with pytest.raises(ValueError, match="checkpoint exists"):
        evaluate_longmemeval(source, embedder, checkpoint=checkpoint)
    with pytest.raises(ValueError, match="configuration or dataset changed"):
        evaluate_longmemeval(source, embedder, checkpoint=checkpoint, resume=True, depth=1)


def test_checkpoint_does_not_overwrite_dataset(tmp_path: Path) -> None:
    source = tmp_path / "dataset.json"
    original = json.dumps([instance()])
    source.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite"):
        evaluate_longmemeval(source, HashEmbeddingProvider(), checkpoint=source)
    assert source.read_text() == original
