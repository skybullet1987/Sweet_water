"""Tests for the enhanced ProbabilisticRegimeEngine with MarketContext (item 2)."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from nextgen.core.models import FeatureOutput
from nextgen.regime.engine import MarketContext, ProbabilisticRegimeEngine, RegimeEngineConfig


def _f(trend: float, mean_rev: float, vol: float = 0.1, vol_ratio: float = 1.0, breadth: float = 0.5) -> FeatureOutput:
    return FeatureOutput(
        "BTC",
        datetime(2026, 1, 1),
        {
            "trend_strength": trend,
            "mean_reversion_score": mean_rev,
            "realized_vol": vol,
            "liquidity": 1.0,
            "breadth": breadth,
            "vol_ratio": vol_ratio,
        },
    )


class RegimeEngineWithContextTest(unittest.TestCase):

    def test_market_context_accepted(self) -> None:
        engine = ProbabilisticRegimeEngine()
        ctx = MarketContext(vol_ratio=1.5, breadth=0.7, funding=0.0, btc_corr_concentration=0.5)
        state = engine.update([_f(0.5, 0.2)], context=ctx)
        self.assertIsNotNone(state)

    def test_high_vol_ratio_boosts_volatility_stress(self) -> None:
        """vol_ratio > 1 should increase volatility_stress relative to baseline."""
        baseline_engine = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        ctx_engine = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        feat = _f(0.3, 0.2, vol=0.05, vol_ratio=1.0)
        ctx_feat = _f(0.3, 0.2, vol=0.05, vol_ratio=3.0)

        baseline_state = baseline_engine.update([feat])
        ctx_state = ctx_engine.update([ctx_feat], context=MarketContext(vol_ratio=3.0))
        self.assertGreater(ctx_state.volatility_stress, baseline_state.volatility_stress)

    def test_high_breadth_boosts_trend_confidence(self) -> None:
        """High market breadth (most assets in uptrend) should push trend_confidence higher."""
        low_breadth_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        high_breadth_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        feat = _f(0.3, 0.2, breadth=0.5)

        low_state = low_breadth_eng.update([feat], context=MarketContext(breadth=0.1))
        high_state = high_breadth_eng.update([feat], context=MarketContext(breadth=0.9))
        self.assertGreater(high_state.trend_confidence, low_state.trend_confidence)

    def test_positive_funding_fades_trend(self) -> None:
        """Large positive funding (crowded longs) should reduce trend_confidence."""
        no_fund_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        high_fund_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        feat = _f(0.6, 0.1)

        no_fund_state = no_fund_eng.update([feat], context=MarketContext(funding=0.0))
        high_fund_state = high_fund_eng.update([feat], context=MarketContext(funding=0.005))
        self.assertGreater(no_fund_state.trend_confidence, high_fund_state.trend_confidence)

    def test_high_btc_corr_reduces_liquidity_quality(self) -> None:
        """High BTC correlation concentration should reduce liquidity_quality."""
        low_corr_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        high_corr_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        feat = _f(0.5, 0.2)

        low_state = low_corr_eng.update([feat], context=MarketContext(btc_corr_concentration=0.1))
        high_state = high_corr_eng.update([feat], context=MarketContext(btc_corr_concentration=0.95))
        self.assertGreater(low_state.liquidity_quality, high_state.liquidity_quality)

    def test_no_context_still_uses_vol_ratio_from_features(self) -> None:
        """Without explicit MarketContext, high vol_ratio in features still raises vol_stress."""
        low_vr_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))
        high_vr_eng = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0))

        low_state = low_vr_eng.update([_f(0.3, 0.2, vol=0.1, vol_ratio=0.5)])
        high_state = high_vr_eng.update([_f(0.3, 0.2, vol=0.1, vol_ratio=3.5)])
        self.assertGreater(high_state.volatility_stress, low_state.volatility_stress)

    def test_existing_persistence_test_still_passes(self) -> None:
        """Backward compatibility: the original hysteresis test must still hold."""
        engine = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0, switch_threshold=0.05, min_persistence_steps=2))
        t0 = datetime(2026, 1, 1)
        trend = FeatureOutput("BTC", t0, {"trend_strength": 0.8, "mean_reversion_score": 0.0, "realized_vol": 0.2, "liquidity": 0.8, "breadth": 0.7})
        first = engine.update([trend])
        self.assertEqual(first.active_regime, "trend")

        mean_rev = FeatureOutput("BTC", t0 + timedelta(minutes=5), {"trend_strength": -0.8, "mean_reversion_score": 0.9, "realized_vol": 0.2, "liquidity": 0.8, "breadth": -0.7})
        second = engine.update([mean_rev])
        self.assertEqual(second.active_regime, "trend")

        third = engine.update([mean_rev])
        self.assertEqual(third.active_regime, "mean_reversion")


if __name__ == "__main__":
    unittest.main()
