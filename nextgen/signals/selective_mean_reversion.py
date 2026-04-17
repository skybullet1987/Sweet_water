from __future__ import annotations

from nextgen.core.models import FeatureOutput, RegimeState, SignalOutput
from .utils import clamp_signal


class SelectiveMeanReversionSleeve:
    name = "selective_mean_reversion"

    def __init__(self, max_volatility_stress: float = 0.8) -> None:
        self.max_volatility_stress = max_volatility_stress

    def generate(self, feature: FeatureOutput, regime: RegimeState) -> SignalOutput:
        reversion_edge = feature.values.get("mean_reversion_score", 0.0)
        enabled = (
            regime.mean_reversion_confidence >= regime.trend_confidence
            and regime.volatility_stress <= self.max_volatility_stress
        )
        score = clamp_signal(reversion_edge * regime.mean_reversion_confidence) if enabled else 0.0
        confidence = max(0.0, min(1.0, abs(score)))
        return SignalOutput(self.name, feature.symbol, feature.timestamp, score, confidence, {"enabled": 1.0 if enabled else 0.0})
