from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import CONFIG, KrakenMaxConfig


@dataclass
class DriftSnapshot:
    live_sharpe: float
    baseline_sharpe: float
    ratio: float
    n_hours: int


class DriftMonitor:
    """Compare rolling live Sharpe to walk-forward baseline (v5)."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.baseline_sharpe = float(config.baseline_sharpe)
        self._equity: deque[tuple[datetime, float]] = deque(maxlen=config.drift_window_hours * 4)

    def load_baseline_from_object_store(self, algo) -> None:
        key = str(self.config.drift_object_store_key)
        try:
            if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
                blob = json.loads(algo.ObjectStore.Read(key))
                self.baseline_sharpe = float(blob.get("oos_sharpe", self.baseline_sharpe))
        except Exception:
            return

    def save_baseline_to_object_store(self, algo, sharpe: float) -> None:
        key = str(self.config.drift_object_store_key)
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps({"oos_sharpe": float(sharpe)}))
                self.baseline_sharpe = float(sharpe)
        except Exception:
            return

    def record_equity(self, when: datetime, equity: float) -> None:
        self._equity.append((when, float(equity)))

    def _hourly_returns(self) -> list[float]:
        if len(self._equity) < 3:
            return []
        points = list(self._equity)
        rets: list[float] = []
        for i in range(1, len(points)):
            prev = points[i - 1][1]
            cur = points[i][1]
            if prev > 0:
                rets.append(cur / prev - 1.0)
        return rets

    def live_sharpe(self, periods_per_year: float | None = None) -> float:
        rets = self._hourly_returns()
        if len(rets) < 8:
            return 0.0
        ppy = periods_per_year
        if ppy is None:
            ppy = 24 * 365 / max(1, self.config.resolution_minutes) * (60 / max(1, self.config.resolution_minutes))
            ppy = 24 * 365 * self.config.bph()
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
        sd = math.sqrt(max(var, 1e-12))
        if sd <= 0:
            return 0.0
        return (mu / sd) * math.sqrt(ppy)

    def evaluate(self) -> DriftSnapshot:
        live = self.live_sharpe()
        base = max(self.baseline_sharpe, 1e-6)
        ratio = live / base if base > 0 else 0.0
        return DriftSnapshot(
            live_sharpe=live,
            baseline_sharpe=base,
            ratio=ratio,
            n_hours=len(self._equity),
        )

    def should_alert(self) -> tuple[bool, str]:
        snap = self.evaluate()
        if snap.n_hours < 24:
            return False, ""
        if snap.baseline_sharpe <= 0:
            return False, ""
        if snap.ratio < float(self.config.drift_sharpe_ratio_threshold):
            return (
                True,
                f"drift live_sharpe={snap.live_sharpe:.2f} baseline={snap.baseline_sharpe:.2f} "
                f"ratio={snap.ratio:.2f}",
            )
        return False, ""
