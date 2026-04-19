from __future__ import annotations

try:
    from config import CONFIG, StrategyConfig
except ModuleNotFoundError:  # pragma: no cover
    from .config import CONFIG, StrategyConfig  # type: ignore


class Scorer:
    ADX_TREND_THRESHOLD = 25.0
    W_CVD = 0.30
    W_OFI = 0.30
    W_VOLC = 0.20
    W_ROT = 0.20

    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.weights = {
            "trend": 1.0 / 4.0,
            "momentum": 1.0 / 4.0,
            "trend_strength": 1.0 / 4.0,
            "flow": 1.0 / 4.0,
        }

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    def composite_score(self, symbol, signal_stack, regime_engine) -> dict[str, float | str]:
        parts = signal_stack.component_scores(symbol)
        mult = signal_stack.stablecoin.multiplier()
        raw = self.W_CVD * parts["cvd"] + self.W_OFI * parts["ofi"] + self.W_VOLC * parts["volc"] + self.W_ROT * parts["rot"]
        h = regime_engine.hurst.hurst(symbol)
        hreg = regime_engine.hurst.regime(symbol)
        final = 0.0 if hreg == "random" else self._clip(raw * mult)
        return {"cvd": parts["cvd"], "ofi": parts["ofi"], "volc": parts["volc"], "rot": parts["rot"], "mult": mult, "raw": raw, "hurst": h, "hurst_regime": hreg, "final": final}

    def indicator_score(self, symbol: str, features: dict[str, float], regime_state: str, btc_context: dict[str, float]) -> float:
        _ = symbol
        btc_trend = float(btc_context.get("btc_trend", 0.0))
        if regime_state == "risk_off":
            return 0.0
        if regime_state == "risk_on":
            trend = 1.0 if features.get("ema20", 0.0) > features.get("ema50", 0.0) else -1.0
            momentum = 1.0 if features.get("mom_24", 0.0) > 0 else -1.0
            strength = 1.0 if features.get("adx", 0.0) > self.ADX_TREND_THRESHOLD else -1.0
            flow = 1.0 if features.get("ofi", 0.0) > 0 else -1.0
            raw = (
                self.weights["trend"] * trend
                + self.weights["momentum"] * momentum
                + self.weights["trend_strength"] * strength
                + self.weights["flow"] * flow
            )
            return self._clip(raw * (1.0 if btc_trend >= -0.01 else 0.8))
        if regime_state == "chop":
            long_votes = [
                1.0 if features.get("rsi", 50.0) < 35 else 0.0,
                1.0 if features.get("cci", 0.0) < -100 else 0.0,
                1.0 if features.get("bb_pos", 0.5) < 0.2 else 0.0,
                1.0 if features.get("mfi", 50.0) < 35 else 0.0,
            ]
            short_votes = [
                1.0 if features.get("rsi", 50.0) > 65 else 0.0,
                1.0 if features.get("cci", 0.0) > 100 else 0.0,
                1.0 if features.get("bb_pos", 0.5) > 0.8 else 0.0,
                1.0 if features.get("mfi", 50.0) > 65 else 0.0,
            ]
            return self._clip((sum(long_votes) - sum(short_votes)) / 4.0)
        return 0.0

    def legacy_score(
        self,
        symbol: str,
        features: dict[str, float],
        regime_state: str,
        btc_context: dict[str, float],
        rank_24h: float = 0.5,
        rank_168h: float = 0.5,
        cross_section_weight: float = 0.40,
    ) -> float:
        base = self.indicator_score(symbol, features, regime_state, btc_context)
        cross_section_score = 0.5 * (float(rank_24h) + float(rank_168h)) - 0.5
        w = max(0.0, min(1.0, float(cross_section_weight)))
        return self._clip((1.0 - w) * base + w * cross_section_score)

    def score(self, *, symbol, features, regime_state, btc_context, rank_24h=0.5, rank_168h=0.5, signal_stack=None, regime_engine=None):
        if self.config.signal_mode == "legacy":
            val = self.legacy_score(
                str(getattr(symbol, "Value", symbol)),
                features,
                regime_state,
                btc_context,
                rank_24h=rank_24h,
                rank_168h=rank_168h,
                cross_section_weight=self.config.cross_section_weight,
            )
            return {"cvd": 0.0, "ofi": float(features.get("ofi", 0.0)), "volc": 0.0, "rot": 0.0, "mult": 1.0, "raw": val, "hurst": 0.5, "hurst_regime": "legacy", "final": val}
        return self.composite_score(symbol, signal_stack, regime_engine)


def composite_score(scorer: Scorer, symbol, signal_stack, regime_engine) -> dict[str, float | str]:
    return scorer.composite_score(symbol, signal_stack, regime_engine)
