from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from config import CONFIG, KrakenMaxConfig
from ml_scorer import MLScorer, load_ml_weights


FEATURE_COLS = (
    "mom_7d",
    "mom_21d",
    "mom_accel",
    "breakout_strength",
    "volume_surge",
    "rsi_pullback",
    "trend_quality",
    "rv_21d_inv",
    "breadth",
    "btc_beta",
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class MLTrainer:
    config: KrakenMaxConfig = field(default_factory=lambda: CONFIG)
    samples: deque = field(default_factory=lambda: deque(maxlen=5000))
    last_retrain: Any = None

    def add_closed_trade(
        self,
        features: dict[str, float],
        context: dict[str, float],
        realized_return: float,
    ) -> None:
        merged = {**features, **context}
        label = 1.0 if float(realized_return) > 0 else 0.0
        row = {k: float(merged.get(k, 0.0)) for k in FEATURE_COLS}
        self.samples.append((row, label))

    def should_retrain(self, now) -> bool:
        if self.last_retrain is None:
            return len(self.samples) >= int(self.config.ml_min_samples)
        days = int(self.config.ml_retrain_days)
        return (now - self.last_retrain) >= timedelta(days=days) and len(self.samples) >= int(
            self.config.ml_min_samples
        )

    def fit_logistic(self) -> dict[str, Any]:
        rows = list(self.samples)
        if len(rows) < int(self.config.ml_min_samples):
            return load_ml_weights()
        x = np.array([[r[c] for c in FEATURE_COLS] for r, _ in rows], dtype=float)
        y = np.array([lab for _, lab in rows], dtype=float)
        w = np.zeros(x.shape[1])
        b = 0.0
        lr = float(self.config.ml_learning_rate)
        steps = int(self.config.ml_train_steps)
        for _ in range(steps):
            z = x @ w + b
            p = _sigmoid(z)
            grad = (p - y) / max(len(y), 1)
            w -= lr * (x.T @ grad)
            b -= lr * float(grad.sum())
        return {
            "bias": float(b),
            "weights": {c: float(wi) for c, wi in zip(FEATURE_COLS, w)},
            "trained_samples": len(rows),
        }

    def retrain(self, scorer: MLScorer, now, persist_path: Path | None = None) -> dict[str, Any]:
        blob = self.fit_logistic()
        scorer.bias = float(blob.get("bias", scorer.bias))
        scorer.weights = {str(k): float(v) for k, v in (blob.get("weights") or {}).items()}
        self.last_retrain = now
        path = persist_path or (Path(__file__).resolve().parent / "ml_weights.json")
        path.write_text(json.dumps({"bias": scorer.bias, "weights": scorer.weights}, indent=2), encoding="utf-8")
        return blob

    def try_persist_object_store(self, algo, blob: dict[str, Any]) -> None:
        key = str(self.config.ml_object_store_key)
        try:
            if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
                pass
            payload = json.dumps(blob)
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, payload)
        except Exception:
            return

    def load_from_object_store(self, algo) -> dict[str, Any] | None:
        key = str(self.config.ml_object_store_key)
        try:
            if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
                raw = algo.ObjectStore.Read(key)
                return json.loads(raw)
        except Exception:
            return None
        return None
