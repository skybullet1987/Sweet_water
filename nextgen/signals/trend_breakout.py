from __future__ import annotations

from nextgen.core.models import FeatureOutput, RegimeState, SignalOutput
from .utils import clamp_signal


class TrendBreakoutSleeve:
    name = "trend_breakout"

    def generate(self, feature: FeatureOutput, regime: RegimeState) -> SignalOutput:
        breakout_strength = feature.values.get("breakout_strength", feature.values.get("trend_strength", 0.0))
        score = clamp_signal(breakout_strength * regime.trend_confidence * (1.0 - regime.volatility_stress))
        confidence = max(0.0, min(1.0, abs(score)))
        return SignalOutput(self.name, feature.symbol, feature.timestamp, score, confidence, {"breakout_strength": float(breakout_strength)})
