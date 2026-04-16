from __future__ import annotations

from nextgen.core.types import FeatureOutput, RegimeState, SignalOutput
from .utils import clamp_signal


class PullbackInTrendSleeve:
    name = "pullback_in_trend"

    def generate(self, feature: FeatureOutput, regime: RegimeState) -> SignalOutput:
        pullback_depth = feature.values.get("pullback_depth", 0.0)
        trend_strength = feature.values.get("trend_strength", 0.0)
        score = clamp_signal((trend_strength - pullback_depth) * regime.trend_confidence)
        confidence = max(0.0, min(1.0, abs(score)))
        return SignalOutput(self.name, feature.symbol, feature.timestamp, score, confidence, {"pullback_depth": float(pullback_depth)})
