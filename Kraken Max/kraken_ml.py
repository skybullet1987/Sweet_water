"""Kraken Max — ML scorer & trainer (`kraken_ml.py`)."""
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

# --- from ml_scorer.py ---


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class MLScorer:
    """Lightweight logistic ensemble — no sklearn required on QC cloud."""

    def __init__(self, weights_blob: dict[str, Any] | None = None) -> None:
        blob = weights_blob or load_ml_weights()
        self.bias = float(blob.get("bias", 0.0))
        self.weights: dict[str, float] = {
            str(k): float(v) for k, v in (blob.get("weights") or {}).items()
        }
        self._online_bias = 0.0
        self._n_updates = 0

    def score(self, features: dict[str, float], context: dict[str, float] | None = None) -> float:
        ctx = context or {}
        merged = {**features, **ctx}
        z = self.bias + self._online_bias
        for name, weight in self.weights.items():
            z += weight * float(merged.get(name, 0.0))
        raw = _sigmoid(z)
        return max(0.0, min(1.0, 2.0 * (raw - 0.5)))

    def online_update(self, realized_return: float, predicted_score: float) -> None:
        """Exponential bias nudge from trade outcomes (risky — fast adaptation)."""
        err = float(realized_return) - float(predicted_score) * 0.05
        self._n_updates += 1
        decay = 0.98
        self._online_bias = decay * self._online_bias + (1.0 - decay) * err * 0.15
        clip = float(CONFIG.score_clip)
        self._online_bias = max(-clip, min(clip, self._online_bias))

# --- from ml_trainer.py ---


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


def default_ml_weights() -> dict[str, Any]:
    return {"bias": 0.0, "weights": {name: 0.0 for name in FEATURE_COLS}}


def load_ml_weights(path: Path | None = None) -> dict[str, Any]:
    """Load ML weights from disk; return neutral defaults if file missing (QC cloud)."""
    root = Path(__file__).resolve().parent
    target = path or (root / "ml_weights.json")
    if not target.is_file():
        return default_ml_weights()
    try:
        with open(target, encoding="utf-8") as fh:
            blob = json.load(fh)
        if not isinstance(blob, dict):
            return default_ml_weights()
        return blob
    except (OSError, json.JSONDecodeError):
        return default_ml_weights()


def _sigmoid_batch(z: np.ndarray) -> np.ndarray:
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
            p = _sigmoid_batch(z)
            grad = (p - y) / max(len(y), 1)
            w -= lr * (x.T @ grad)
            b -= lr * float(grad.sum())
        preds = _sigmoid_batch(x @ w + b) >= 0.5
        acc = float(np.mean(preds == y)) if len(y) else 0.5
        return {
            "bias": float(b),
            "weights": {c: float(wi) for c, wi in zip(FEATURE_COLS, w)},
            "trained_samples": len(rows),
            "train_accuracy": acc,
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