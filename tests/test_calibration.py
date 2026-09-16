from __future__ import annotations

from pathlib import Path

from context_router.routing.calibration import PlattContextRanker

FEATURES = [
    "rrf",
    "dense",
    "lexical",
    "entity",
    "agreement",
    "primary",
    "recent",
    "recency",
    "relation_match",
]


def row(value: float) -> dict[str, float]:
    return {feature: value for feature in FEATURES}


def test_platt_ranker_learns_probabilities_and_round_trips(tmp_path: Path) -> None:
    negatives = [row(value) for value in (0.00, 0.05, 0.10, 0.12, 0.18, 0.20)]
    positives = [row(value) for value in (0.70, 0.75, 0.80, 0.85, 0.90, 1.00)]
    ranker = PlattContextRanker.fit(
        train_rows=negatives[:4] + positives[:4],
        train_labels=[0] * 4 + [1] * 4,
        calibration_rows=negatives[4:] + positives[4:],
        calibration_labels=[0, 0, 1, 1],
    )

    low = ranker.probability(row(0.05))
    high = ranker.probability(row(0.95))
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0

    path = tmp_path / "ranker.json"
    ranker.save(path)
    restored = PlattContextRanker.load(path)
    assert restored.probability(row(0.95)) == ranker.probability(row(0.95))
