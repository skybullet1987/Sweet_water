from __future__ import annotations

try:
    from config import CONFIG, StrategyConfig
except ModuleNotFoundError:  # pragma: no cover
    from .config import CONFIG, StrategyConfig  # type: ignore


class Scorer:
    ADX_TREND_THRESHOLD = 25.0
    BREADTH_PENALTY_THRESHOLD = 0.30
    MULT_MIN = 0.4
    MULT_MAX = 1.3
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

    def cross_sectional_momentum_score(self, features: dict[str, float]) -> dict[str, float | str]:
        rv_floor = max(float(getattr(self.config, "min_rv_floor", 1e-4) or 1e-4), 1e-9)
        clip = max(float(getattr(self.config, "score_clip_value", 5.0) or 5.0), 1.0)
        mom_21d = float(features.get("mom_21d", 0.0) or 0.0)
        mom_63d = float(features.get("mom_63d", 0.0) or 0.0)
        rv_21d = max(float(features.get("rv_21d", 0.0) or 0.0), rv_floor)
        dd_63d = self._clamp(float(features.get("dd_63d", 0.0) or 0.0), 0.0, 1.0)
        m21_adj = self._clamp(mom_21d / rv_21d, -clip, clip)
        m63_adj = self._clamp(mom_63d / rv_21d, -clip, clip)
        score = self._clamp(0.6 * m21_adj + 0.4 * m63_adj - 0.3 * dd_63d, -clip, clip)
        return {
            "cvd": 0.0,
            "ofi": 0.0,
            "volc": 0.0,
            "rot": 0.0,
            "mult": 1.0,
            "raw": score,
            "vr": 1.0,
            "vr_regime": "daily_cs_mom",
            "hurst": 0.5,
            "hurst_regime": "daily_cs_mom",
            "final": score,
        }

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(float(lo), min(float(hi), float(value)))

    @staticmethod
    def _vr_multiplier(vr_regime: str) -> float:
        if vr_regime == "trend":
            return 1.0
        if vr_regime == "meanrev":
            return 0.8
        return 0.6

    @staticmethod
    def _vol_stress_multiplier(vol_stress: float, threshold: float) -> float:
        return 0.7 if float(vol_stress) > float(threshold) else 1.0

    @staticmethod
    def _breadth_multiplier(breadth: float) -> float:
        return 0.7 if float(breadth) < Scorer.BREADTH_PENALTY_THRESHOLD else 1.0

    def composite_score(self, symbol, signal_stack, regime_engine, breadth: float = 0.5) -> dict[str, float | str]:
        parts = signal_stack.component_scores(symbol)
        raw = self.W_CVD * parts["cvd"] + self.W_OFI * parts["ofi"] + self.W_VOLC * parts["volc"] + self.W_ROT * parts["rot"]
        vr = regime_engine.vr.variance_ratio(symbol)
        vr_reg = regime_engine.vr.regime(symbol)
        mult = 1.0
        mult *= self._vr_multiplier(vr_reg)
        mult *= self._vol_stress_multiplier(regime_engine.vol_stress, self.config.vol_stress_threshold)
        mult *= self._breadth_multiplier(breadth)
        mult = self._clamp(mult, self.MULT_MIN, self.MULT_MAX)
        h = regime_engine.hurst.hurst(symbol)
        hreg = regime_engine.hurst.regime(symbol)
        final = self._clip(raw * mult)
        return {
            "cvd": parts["cvd"],
            "ofi": parts["ofi"],
            "volc": parts["volc"],
            "rot": parts["rot"],
            "mult": mult,
            "raw": raw,
            "vr": vr,
            "vr_regime": vr_reg,
            "hurst": h,
            "hurst_regime": hreg,
            "final": final,
        }

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

    def score(
        self,
        *,
        symbol,
        features,
        regime_state,
        btc_context,
        rank_24h=0.5,
        rank_168h=0.5,
        breadth=0.5,
        signal_stack=None,
        regime_engine=None,
    ):
        mode = str(getattr(self.config, "signal_mode", "cross_sectional_momentum"))
        if mode == "legacy":
            val = self.legacy_score(
                str(getattr(symbol, "Value", symbol)),
                features,
                regime_state,
                btc_context,
                rank_24h=rank_24h,
                rank_168h=rank_168h,
                cross_section_weight=self.config.cross_section_weight,
            )
            return {
                "cvd": 0.0,
                "ofi": float(features.get("ofi", 0.0)),
                "volc": 0.0,
                "rot": 0.0,
                "mult": 1.0,
                "raw": val,
                "vr": 1.0,
                "vr_regime": "legacy",
                "hurst": 0.5,
                "hurst_regime": "legacy",
                "final": val,
            }
        if mode == "microstructure":
            return self.composite_score(symbol, signal_stack, regime_engine, breadth=breadth)
        return self.cross_sectional_momentum_score(features)
