from __future__ import annotations

from .btc_dominance_rotation import BtcDominanceRotationSignal
from .cvd_divergence import CvdDivergenceSignal
from .hurst_regime import HurstRegimeSignal
from .order_flow_imbalance import OrderFlowImbalanceSignal
from .stablecoin_liquidity import StablecoinLiquidityOverlay
from .vol_cone_breakout import VolConeBreakoutSignal


class MicrostructureSignalEnsemble:
    W_CVD = 0.30
    W_OFI = 0.30
    W_VOLC = 0.20
    W_ROT = 0.20

    def __init__(self, algo=None, tracked_symbols: list[str] | None = None) -> None:
        self.cvd_divergence = CvdDivergenceSignal()
        self.order_flow_imbalance = OrderFlowImbalanceSignal()
        self.vol_cone_breakout = VolConeBreakoutSignal()
        self.btc_dominance_rotation = BtcDominanceRotationSignal(tracked_symbols or [])
        self.stablecoin_liquidity = StablecoinLiquidityOverlay(algo)
        self.hurst_regime = HurstRegimeSignal(window=500)

    @staticmethod
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        self.btc_dominance_rotation.set_tracked_symbols(symbols)

    def update(self, symbol, bar_or_tick) -> None:
        self.cvd_divergence.update(symbol, bar_or_tick)
        self.order_flow_imbalance.update(symbol, bar_or_tick)
        self.vol_cone_breakout.update(symbol, bar_or_tick)
        self.btc_dominance_rotation.update(symbol, bar_or_tick)
        self.hurst_regime.update(symbol, bar_or_tick)
        self.stablecoin_liquidity.update(getattr(bar_or_tick, "EndTime", getattr(bar_or_tick, "Time", None)))

    def component_scores(self, symbol) -> dict[str, float]:
        return {
            "cvd": self.cvd_divergence.score(symbol),
            "ofi": self.order_flow_imbalance.score(symbol),
            "volc": self.vol_cone_breakout.score(symbol),
            "rot": self.btc_dominance_rotation.score(symbol),
        }

    def composite_score(self, symbol) -> float:
        return float(self.snapshot(symbol)["final"])

    def snapshot(self, symbol) -> dict[str, float | str]:
        parts = self.component_scores(symbol)
        mult = self.stablecoin_liquidity.multiplier()
        raw = self.W_CVD * parts["cvd"] + self.W_OFI * parts["ofi"] + self.W_VOLC * parts["volc"] + self.W_ROT * parts["rot"]
        h = self.hurst_regime.hurst(symbol)
        regime = self.hurst_regime.regime(symbol)
        final = 0.0 if regime == "random" else self._clamp(raw * mult)
        return {"cvd": parts["cvd"], "ofi": parts["ofi"], "volc": parts["volc"], "rot": parts["rot"], "mult": mult, "raw": raw, "hurst": h, "hurst_regime": regime, "final": final}

    def init_status(self) -> dict[str, str]:
        return {
            "cvd": "trade-tick if available, bar fallback enabled",
            "ofi": "quote L1" if not self.order_flow_imbalance.using_fallback() else "quote-bar fallback",
            "vol_cone": "garman-klass",
            "btc_rotation": "active",
            "stablecoin_overlay": "coingecko+cache",
            "hurst": "R/S(500)",
        }
