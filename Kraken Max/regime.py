from __future__ import annotations

from dataclasses import dataclass

from config import KrakenMaxConfig, CONFIG
from sentiment import SentimentSnapshot, adjust_deployment_cap


@dataclass(frozen=True)
class RegimeState:
    name: str
    deployment_cap: float
    allow_new_entries: bool
    prefer_symbols: tuple[str, ...]
    allow_scalper: bool = False
    micro_regime: str = "unknown"


class RegimeEngine:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def classify(
        self,
        *,
        btc_features: dict[str, float],
        breadth: float,
        median_rv: float,
        sentiment: SentimentSnapshot | None = None,
    ) -> RegimeState:
        btc_trend = float(btc_features.get("trend_quality", 0.0))
        btc_mom = float(btc_features.get("mom_21d", 0.0))
        vol_stress = median_rv >= float(self.config.vol_stress_threshold)
        fg = float(sentiment.fear_greed) if sentiment else 0.5

        if vol_stress and breadth < 0.35:
            return RegimeState("chaos", 0.0, False, (), allow_scalper=False, micro_regime="chaos")
        if btc_trend > 0 and btc_mom > 0 and breadth >= float(self.config.breadth_bull_threshold):
            cap = float(self.config.total_deployment_cap)
            if sentiment:
                cap = adjust_deployment_cap(cap, sentiment, "bull", self.config)
            return RegimeState("bull", cap, True, (), allow_scalper=False, micro_regime="trend")
        if btc_trend < 0 or btc_mom < -0.05:
            cap = float(self.config.bear_deployment_cap)
            if sentiment and fg < float(self.config.fg_extreme_fear):
                cap *= 0.85
            return RegimeState(
                "bear",
                cap,
                True,
                tuple(self.config.bear_prefer),
                allow_scalper=False,
                micro_regime="bear",
            )
        cap = 0.75
        if sentiment:
            cap = adjust_deployment_cap(cap, sentiment, "neutral", self.config)
        ranging = abs(btc_mom) < 0.03 and breadth > 0.4 and breadth < 0.65
        return RegimeState(
            "neutral",
            cap,
            True,
            (),
            allow_scalper=ranging or fg < 0.45,
            micro_regime="ranging" if ranging else "neutral",
        )
