from __future__ import annotations

from nextgen.core.types import FeatureOutput, RegimeState, SignalOutput
from .utils import clamp_signal


class CrossSectionalMomentumSleeve:
    name = "cross_sectional_momentum"

    def generate(self, feature: FeatureOutput, regime: RegimeState) -> SignalOutput:
        rank_score = feature.values.get("momentum_rank", feature.values.get("momentum", 0.0))
        score = clamp_signal(rank_score * regime.trend_confidence)
        confidence = max(0.0, min(1.0, abs(score)))
        return SignalOutput(self.name, feature.symbol, feature.timestamp, score, confidence, {"raw_rank": float(rank_score)})
