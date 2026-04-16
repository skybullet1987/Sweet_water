from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Sequence

from nextgen.core.types import FeatureOutput, RegimeState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class RegimeEngineConfig:
    smoothing: float = 0.25
    switch_threshold: float = 0.15
    min_persistence_steps: int = 2


class ProbabilisticRegimeEngine:
    def __init__(self, config: RegimeEngineConfig | None = None) -> None:
        self.config = config or RegimeEngineConfig()
        self._prev: RegimeState | None = None

    def _label(self, trend: float, mean_rev: float) -> str:
        return "trend" if trend >= mean_rev else "mean_reversion"

    def update(self, features: Sequence[FeatureOutput]) -> RegimeState:
        if not features:
            now = datetime.now(UTC)
            return RegimeState(now, 0.5, 0.5, 0.5, 0.5, 0.5, "neutral", 1)

        trend_strength = sum(f.values.get("trend_strength", 0.0) for f in features) / len(features)
        mean_rev_strength = sum(f.values.get("mean_reversion_score", 0.0) for f in features) / len(features)
        realized_vol = sum(f.values.get("realized_vol", 0.0) for f in features) / len(features)
        liquidity = sum(f.values.get("liquidity", 0.0) for f in features) / len(features)
        breadth = sum(f.values.get("breadth", 0.0) for f in features) / len(features)

        raw_trend = _clamp01(0.5 + 0.7 * trend_strength - 0.3 * mean_rev_strength)
        raw_mean_rev = _clamp01(0.5 + 0.7 * mean_rev_strength - 0.3 * abs(trend_strength))
        raw_vol_stress = _clamp01(realized_vol)
        raw_liq_quality = _clamp01(liquidity)
        raw_breadth = _clamp01(0.5 + 0.5 * breadth)

        timestamp = max(f.timestamp for f in features)
        if self._prev is None:
            active = self._label(raw_trend, raw_mean_rev)
            state = RegimeState(timestamp, raw_trend, raw_mean_rev, raw_vol_stress, raw_liq_quality, raw_breadth, active, 1)
            self._prev = state
            return state

        alpha = self.config.smoothing
        trend = _clamp01((1 - alpha) * self._prev.trend_confidence + alpha * raw_trend)
        mean_rev = _clamp01((1 - alpha) * self._prev.mean_reversion_confidence + alpha * raw_mean_rev)
        vol_stress = _clamp01((1 - alpha) * self._prev.volatility_stress + alpha * raw_vol_stress)
        liq_quality = _clamp01((1 - alpha) * self._prev.liquidity_quality + alpha * raw_liq_quality)
        breadth_strength = _clamp01((1 - alpha) * self._prev.breadth_strength + alpha * raw_breadth)

        candidate = self._label(trend, mean_rev)
        margin = abs(trend - mean_rev)
        active = self._prev.active_regime
        persistence = self._prev.regime_persistence_steps + 1

        can_switch = self._prev.regime_persistence_steps >= self.config.min_persistence_steps
        if candidate != active and margin >= self.config.switch_threshold and can_switch:
            active = candidate
            persistence = 1

        state = RegimeState(timestamp, trend, mean_rev, vol_stress, liq_quality, breadth_strength, active, persistence)
        self._prev = state
        return state
