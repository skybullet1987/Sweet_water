from __future__ import annotations

from dataclasses import dataclass

from config import KrakenMaxConfig, CONFIG


@dataclass(frozen=True)
class RegimeState:
    name: str
    deployment_cap: float
    allow_new_entries: bool
    prefer_symbols: tuple[str, ...]


class RegimeEngine:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def classify(
        self,
        *,
        btc_features: dict[str, float],
        breadth: float,
        median_rv: float,
    ) -> RegimeState:
        btc_trend = float(btc_features.get("trend_quality", 0.0))
        btc_mom = float(btc_features.get("mom_21d", 0.0))
        vol_stress = median_rv >= float(self.config.vol_stress_threshold)

        if vol_stress and breadth < 0.35:
            return RegimeState("chaos", 0.0, False, ())
        if btc_trend > 0 and btc_mom > 0 and breadth >= float(self.config.breadth_bull_threshold):
            return RegimeState("bull", float(self.config.total_deployment_cap), True, ())
        if btc_trend < 0 or btc_mom < -0.05:
            return RegimeState(
                "bear",
                float(self.config.bear_deployment_cap),
                True,
                tuple(self.config.bear_prefer),
            )
        return RegimeState("neutral", 0.75, True, ())
