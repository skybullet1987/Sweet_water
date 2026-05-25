from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from config import CONFIG


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def load_ml_weights(path: Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    target = path or (root / "ml_weights.json")
    with open(target, encoding="utf-8") as fh:
        return json.load(fh)


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
