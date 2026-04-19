from __future__ import annotations

try:
    from config import CONFIG, StrategyConfig
except ModuleNotFoundError:  # pragma: no cover
    from .config import CONFIG, StrategyConfig  # type: ignore


class Scorer:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config

    def cross_sectional_momentum_score(
        self, features: dict[str, float], cross_section_zscore: float | None = None
    ) -> dict[str, float | str]:
        clip = max(float(getattr(self.config, "score_clip_value", 5.0) or 5.0), 1.0)
        if cross_section_zscore is not None:
            score = self._clamp(float(cross_section_zscore), -clip, clip)
            return {"raw": score, "final": score}
        rv_floor = max(float(getattr(self.config, "min_rv_floor", 1e-4) or 1e-4), 1e-9)
        mom_21d = float(features.get("mom_21d", 0.0) or 0.0)
        mom_63d = float(features.get("mom_63d", 0.0) or 0.0)
        rv_21d = max(float(features.get("rv_21d", 0.0) or 0.0), rv_floor)
        dd_63d = self._clamp(float(features.get("dd_63d", 0.0) or 0.0), 0.0, 1.0)
        m21_adj = self._clamp(mom_21d / rv_21d, -clip, clip)
        m63_adj = self._clamp(mom_63d / rv_21d, -clip, clip)
        w21 = float(getattr(self.config, "score_mom21_weight", 0.6) or 0.6)
        w63 = float(getattr(self.config, "score_mom63_weight", 0.4) or 0.4)
        dd_penalty = float(getattr(self.config, "score_dd_penalty", 0.3) or 0.3)
        score = self._clamp(w21 * m21_adj + w63 * m63_adj - dd_penalty * dd_63d, -clip, clip)
        return {"raw": score, "final": score}

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(float(lo), min(float(hi), float(value)))

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
        cross_section_zscore=None,
    ):
        _ = (symbol, regime_state, btc_context, rank_24h, rank_168h, breadth, signal_stack, regime_engine)
        return self.cross_sectional_momentum_score(features, cross_section_zscore=cross_section_zscore)
