from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Protocol, Self

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "rrf",
    "dense",
    "lexical",
    "entity",
    "agreement",
    "primary",
    "recent",
    "recency",
    "relation_match",
)

#: Bump whenever a feature's *meaning* changes, not just its name. A ranker fitted under
#: older semantics produces confident nonsense, so it must be refused rather than loaded.
FEATURE_SCHEMA_VERSION = "2"


class CandidateRanker(Protocol):
    model_version: str

    def probability(self, features: dict[str, float]) -> float: ...


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class HeuristicContextRanker:
    model_version = "heuristic-v1"

    def probability(self, features: dict[str, float]) -> float:
        linear = (
            -2.7
            + 1.45 * features["rrf"]
            + 1.10 * features["dense"]
            + 1.05 * features["lexical"]
            + 1.70 * features["entity"]
            + 0.55 * features["agreement"]
            + 0.35 * features["primary"]
            + 0.20 * features["recent"]
            + 0.15 * features["recency"]
            + 1.00 * features["relation_match"]
        )
        return _sigmoid(linear)


class PlattContextRanker:
    """Standardized logistic reranker followed by a held-out Platt calibrator."""

    def __init__(
        self,
        *,
        means: list[float],
        scales: list[float],
        base_coefficients: list[float],
        base_intercept: float,
        platt_coefficient: float,
        platt_intercept: float,
    ) -> None:
        self.means = means
        self.scales = scales
        self.base_coefficients = base_coefficients
        self.base_intercept = base_intercept
        self.platt_coefficient = platt_coefficient
        self.platt_intercept = platt_intercept
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        self.model_version = f"platt-{hashlib.sha256(payload).hexdigest()[:12]}"

    @classmethod
    def fit(
        cls,
        *,
        train_rows: list[dict[str, float]],
        train_labels: list[int],
        calibration_rows: list[dict[str, float]],
        calibration_labels: list[int],
    ) -> Self:
        if set(train_labels) != {0, 1} or set(calibration_labels) != {0, 1}:
            raise ValueError("training and calibration labels must each contain 0 and 1")
        train = cls._matrix(train_rows)
        calibration = cls._matrix(calibration_rows)
        scaler = StandardScaler().fit(train)
        base = LogisticRegression(class_weight="balanced", random_state=0).fit(
            scaler.transform(train), train_labels
        )
        calibration_logits = base.decision_function(scaler.transform(calibration)).reshape(-1, 1)
        platt = LogisticRegression(class_weight="balanced", random_state=0).fit(
            calibration_logits, calibration_labels
        )
        return cls(
            means=scaler.mean_.astype(float).tolist(),
            scales=scaler.scale_.astype(float).tolist(),
            base_coefficients=base.coef_[0].astype(float).tolist(),
            base_intercept=float(base.intercept_[0]),
            platt_coefficient=float(platt.coef_[0][0]),
            platt_intercept=float(platt.intercept_[0]),
        )

    @staticmethod
    def _matrix(rows: list[dict[str, float]]) -> np.ndarray:
        return np.asarray(
            [[row.get(feature, 0.0) for feature in FEATURE_NAMES] for row in rows], dtype=float
        )

    def probability(self, features: dict[str, float]) -> float:
        values = [features.get(feature, 0.0) for feature in FEATURE_NAMES]
        standardized = [
            (value - mean) / (scale or 1.0)
            for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        base_logit = self.base_intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.base_coefficients, standardized, strict=True)
        )
        return _sigmoid(self.platt_intercept + self.platt_coefficient * base_logit)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "means": self.means,
            "scales": self.scales,
            "base_coefficients": self.base_coefficients,
            "base_intercept": self.base_intercept,
            "platt_coefficient": self.platt_coefficient,
            "platt_intercept": self.platt_intercept,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["feature_names"]) != FEATURE_NAMES:
            raise ValueError("ranker feature schema does not match this version")
        if data.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "ranker was fitted under different feature semantics "
                f"(file={data.get('feature_schema_version')!r}, "
                f"expected={FEATURE_SCHEMA_VERSION!r}); retrain it"
            )
        return cls(
            means=data["means"],
            scales=data["scales"],
            base_coefficients=data["base_coefficients"],
            base_intercept=data["base_intercept"],
            platt_coefficient=data["platt_coefficient"],
            platt_intercept=data["platt_intercept"],
        )
