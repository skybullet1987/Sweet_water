from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional, Sequence

from nextgen.core.types import FeatureOutput, RegimeState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class MarketContext:
    """
    Aggregate market-level signals to inform regime classification.

    vol_ratio            – short-window / long-window realised vol; >1 = rising stress
    breadth              – fraction of universe where ema_short > ema_long (0–1)
    funding              – funding-rate proxy (positive = longs paying premium)
    btc_corr_concentration – fraction of universe with high BTC correlation (0–1)
    """
    vol_ratio: float = 1.0
    breadth: float = 0.5
    funding: float = 0.0
    btc_corr_concentration: float = 0.5


@dataclass
class RegimeEngineConfig:
    smoothing: float = 0.25
    switch_threshold: float = 0.15
    min_persistence_steps: int = 2


class ProbabilisticRegimeEngine:
    def __init__(self, config: RegimeEngineConfig | None = None) -> None:
        self.config = config or RegimeEngineConfig()
        self._prev: RegimeState | None = None

    def _label(self, trend: float, mean_rev: float) -> str:
        return "trend" if trend >= mean_rev else "mean_reversion"

    # ── public API ────────────────────────────────────────────────────────

    def update(
        self,
        features: Sequence[FeatureOutput],
        context: Optional[MarketContext] = None,
    ) -> RegimeState:
        """
        Update regime state.

        Parameters
        ----------
        features : sequence of FeatureOutput from the symbol universe
        context  : optional MarketContext (vol_ratio, breadth, funding, btc_corr)
                   When provided, the raw regime signals are blended with context-
                   derived signals, replacing the simple momentum-comparison rule
                   with empirically more robust market-structure signals.
        """
        if not features:
            now = datetime.now(UTC)
            return RegimeState(now, 0.5, 0.5, 0.5, 0.5, 0.5, "neutral", 1)

        # ── Per-symbol feature aggregates ────────────────────────────────
        trend_strength = sum(f.values.get("trend_strength", 0.0) for f in features) / len(features)
        mean_rev_strength = sum(f.values.get("mean_reversion_score", 0.0) for f in features) / len(features)
        realized_vol = sum(f.values.get("realized_vol", 0.0) for f in features) / len(features)
        liquidity = sum(f.values.get("liquidity", 0.0) for f in features) / len(features)
        breadth = sum(f.values.get("breadth", 0.0) for f in features) / len(features)
        vol_ratio_avg = sum(f.values.get("vol_ratio", 1.0) for f in features) / len(features)

        # ── Raw signals (per-symbol features) ────────────────────────────
        raw_trend = _clamp01(0.5 + 0.7 * trend_strength - 0.3 * mean_rev_strength)
        raw_mean_rev = _clamp01(0.5 + 0.7 * mean_rev_strength - 0.3 * abs(trend_strength))
        raw_vol_stress = _clamp01(realized_vol)
        raw_liq_quality = _clamp01(liquidity)
        raw_breadth = _clamp01(0.5 + 0.5 * breadth)

        # ── Context-based regime signals ──────────────────────────────────
        # When a MarketContext is provided we blend additional market-structure
        # signals that are more reliable than per-bar momentum comparisons.
        if context is not None:
            # Vol-ratio: short/long realised-vol expansion indicates stress.
            # vol_ratio > 1.5 → strong stress signal; < 0.7 → calm / mean-rev.
            vol_stress_context = _clamp01((context.vol_ratio - 1.0) / 1.5)
            raw_vol_stress = _clamp01(0.5 * raw_vol_stress + 0.5 * vol_stress_context)

            # Breadth: direct market-breadth estimate overrides per-symbol proxy.
            # breadth > 0.6 → trend-supportive; < 0.4 → mean-reversion-supportive.
            ctx_breadth_signal = _clamp01(0.5 + 1.0 * (context.breadth - 0.5))
            raw_breadth = _clamp01(0.4 * raw_breadth + 0.6 * ctx_breadth_signal)

            # High breadth → boost trend; low breadth → boost mean-reversion.
            breadth_adj = context.breadth - 0.5           # −0.5 … +0.5
            raw_trend = _clamp01(raw_trend + 0.3 * breadth_adj)
            raw_mean_rev = _clamp01(raw_mean_rev - 0.3 * breadth_adj)

            # Funding: extreme positive funding → crowded longs, fade trend.
            # Values > 0.1% per 8h are elevated; we scale to a 0-1 signal.
            funding_extreme = _clamp01(abs(context.funding) / 0.002)  # saturates at 0.2%
            if context.funding > 0:
                # Crowded longs — fade trend, boost mean-reversion
                raw_trend = _clamp01(raw_trend - 0.2 * funding_extreme)
                raw_mean_rev = _clamp01(raw_mean_rev + 0.2 * funding_extreme)
            elif context.funding < 0:
                # Crowded shorts — mean-reversion less reliable
                raw_mean_rev = _clamp01(raw_mean_rev - 0.15 * funding_extreme)

            # BTC correlation concentration: when most assets move with BTC,
            # idiosyncratic signals weaker → reduce liquidity quality proxy.
            corr_penalty = _clamp01(context.btc_corr_concentration - 0.5) * 2.0  # 0-1
            raw_liq_quality = _clamp01(raw_liq_quality * (1.0 - 0.3 * corr_penalty))

            # Also incorporate per-feature vol_ratio average alongside context.
            combined_vol_ratio = 0.5 * vol_ratio_avg + 0.5 * context.vol_ratio
            extra_stress = _clamp01((combined_vol_ratio - 1.0) / 1.5)
            raw_vol_stress = _clamp01(0.6 * raw_vol_stress + 0.4 * extra_stress)
        else:
            # Without context: still use the per-symbol vol_ratio aggregate.
            extra_stress = _clamp01((vol_ratio_avg - 1.0) / 1.5)
            raw_vol_stress = _clamp01(0.6 * raw_vol_stress + 0.4 * extra_stress)

        # ── EMA smoothing + persistence ───────────────────────────────────
        timestamp = max(f.timestamp for f in features)
        if self._prev is None:
            active = self._label(raw_trend, raw_mean_rev)
            state = RegimeState(
                timestamp, raw_trend, raw_mean_rev,
                raw_vol_stress, raw_liq_quality, raw_breadth,
                active, 1,
            )
            self._prev = state
            return state

        alpha = self.config.smoothing
        trend = _clamp01((1 - alpha) * self._prev.trend_confidence + alpha * raw_trend)
        mean_rev = _clamp01((1 - alpha) * self._prev.mean_reversion_confidence + alpha * raw_mean_rev)
        vol_stress = _clamp01((1 - alpha) * self._prev.volatility_stress + alpha * raw_vol_stress)
        liq_quality = _clamp01((1 - alpha) * self._prev.liquidity_quality + alpha * raw_liq_quality)
        breadth_strength = _clamp01((1 - alpha) * self._prev.breadth_strength + alpha * raw_breadth)

        candidate = self._label(trend, mean_rev)
        margin = abs(trend - mean_rev)
        active = self._prev.active_regime
        persistence = self._prev.regime_persistence_steps + 1

        can_switch = self._prev.regime_persistence_steps >= self.config.min_persistence_steps
        if candidate != active and margin >= self.config.switch_threshold and can_switch:
            active = candidate
            persistence = 1

        state = RegimeState(
            timestamp, trend, mean_rev, vol_stress, liq_quality,
            breadth_strength, active, persistence,
        )
        self._prev = state
        return state
