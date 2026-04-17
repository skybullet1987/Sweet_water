"""Tests for the expanded BasicFeatureEngine (items 1 of the plan)."""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

from nextgen.core.models import Bar
from nextgen.features.basic import BasicFeatureEngine


def _bar(symbol: str, close: float, t: datetime, volume: float = 1000.0, high: float = 0, low: float = 0) -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=t,
        open=close,
        high=high or close * 1.002,
        low=low or close * 0.998,
        close=close,
        volume=volume,
    )


class FeatureEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.t0 = datetime(2026, 1, 1, 0, 0)
        self.engine = BasicFeatureEngine(short_window=5, long_window=10, ema_short_bars=3, ema_long_bars=6)

    def _feed(self, symbol: str, closes: list, base_time: datetime | None = None) -> list:
        base = base_time or self.t0
        outputs = []
        for i, c in enumerate(closes):
            t = base + timedelta(minutes=5 * i)
            outputs.append(self.engine.update(_bar(symbol, c, t)))
        return outputs

    # ── output shape ────────────────────────────────────────────────────────

    def test_all_required_keys_present(self) -> None:
        out = self._feed("BTC", [100.0])[-1]
        for key in (
            "momentum", "momentum_short", "momentum_long",
            "trend_strength", "mean_reversion_score",
            "realized_vol", "realized_vol_short", "realized_vol_long",
            "vol_ratio", "rsi_norm", "vwap_deviation",
            "volume_ratio", "atr_proxy", "breadth", "liquidity",
            "breakout_strength", "pullback_depth",
        ):
            self.assertIn(key, out.values, f"Missing key: {key}")

    def test_liquidity_is_one_for_positive_volume(self) -> None:
        out = self._feed("BTC", [100.0])[-1]
        self.assertEqual(out.values["liquidity"], 1.0)

    def test_liquidity_is_zero_for_zero_volume(self) -> None:
        bar = Bar("BTC", self.t0, 100, 101, 99, 100, 0.0)
        out = self.engine.update(bar)
        self.assertEqual(out.values["liquidity"], 0.0)

    # ── momentum ────────────────────────────────────────────────────────────

    def test_momentum_positive_on_uptrend(self) -> None:
        closes = [100.0 + i for i in range(12)]
        outs = self._feed("BTC", closes)
        self.assertGreater(outs[-1].values["momentum_short"], 0.0)
        self.assertGreater(outs[-1].values["momentum_long"], 0.0)

    def test_momentum_negative_on_downtrend(self) -> None:
        closes = [100.0 - i * 0.5 for i in range(12)]
        outs = self._feed("BTC", closes)
        self.assertLess(outs[-1].values["momentum_short"], 0.0)

    # ── vol_ratio ────────────────────────────────────────────────────────────

    def test_vol_ratio_above_one_on_expanding_vol(self) -> None:
        # Calm prices followed by big moves → short vol > long vol
        import random
        random.seed(0)
        calm = [100.0 + random.uniform(-0.01, 0.01) for _ in range(20)]
        explosive = [calm[-1] + random.uniform(-2.0, 2.0) for _ in range(5)]
        outs = self._feed("BTC", calm + explosive)
        vr = outs[-1].values["vol_ratio"]
        self.assertGreater(vr, 1.0)

    # ── EMA / trend_strength ─────────────────────────────────────────────────

    def test_trend_strength_positive_on_uptrend(self) -> None:
        closes = [100.0 + i * 2.0 for i in range(15)]
        outs = self._feed("ETH", closes)
        self.assertGreater(outs[-1].values["trend_strength"], 0.0)

    def test_trend_strength_negative_on_downtrend(self) -> None:
        closes = [200.0 - i * 2.0 for i in range(15)]
        outs = self._feed("ETH", closes)
        self.assertLess(outs[-1].values["trend_strength"], 0.0)

    def test_breadth_one_when_ema_short_above_long(self) -> None:
        closes = [100.0 + i * 3.0 for i in range(15)]
        outs = self._feed("SOL", closes)
        # After strong uptrend short EMA > long EMA
        self.assertEqual(outs[-1].values["breadth"], 1.0)

    # ── VWAP / mean_reversion_score ──────────────────────────────────────────

    def test_mean_reversion_score_negative_when_price_above_vwap(self) -> None:
        # Gradually rising prices: last bar is well above early VWAP
        closes = [100.0, 100.0, 100.0, 100.0, 100.0, 120.0]
        outs = self._feed("LTC", closes)
        # Price jumped above VWAP → deviation > 0 → mean_rev_score < 0
        self.assertLess(outs[-1].values["mean_reversion_score"], 0.0)

    # ── volume_ratio ─────────────────────────────────────────────────────────

    def test_volume_ratio_above_one_on_surge(self) -> None:
        t = self.t0
        for i in range(10):
            self.engine.update(_bar("XRP", 1.0, t + timedelta(minutes=5 * i), volume=100.0))
        # Now a volume surge
        surge = self.engine.update(_bar("XRP", 1.0, t + timedelta(minutes=55), volume=1000.0))
        self.assertGreater(surge.values["volume_ratio"], 1.0)

    # ── RSI normalised ───────────────────────────────────────────────────────

    def test_rsi_norm_positive_on_all_up_moves(self) -> None:
        closes = [100.0 + i for i in range(12)]  # pure up trend
        outs = self._feed("BNB", closes)
        self.assertGreater(outs[-1].values["rsi_norm"], 0.0)

    # ── backward compatibility ───────────────────────────────────────────────

    def test_lookback_kwarg_accepted(self) -> None:
        eng = BasicFeatureEngine(lookback=5)
        out = eng.update(_bar("BTC", 100.0, self.t0))
        self.assertIn("momentum", out.values)

    # ── VWAP reset ───────────────────────────────────────────────────────────

    def test_vwap_reset_clears_accumulator(self) -> None:
        for i in range(5):
            self.engine.update(_bar("AVAX", 100.0, self.t0 + timedelta(minutes=5 * i)))
        self.engine.reset_vwap("AVAX")
        # After reset next bar restarts VWAP from close
        out = self.engine.update(_bar("AVAX", 200.0, self.t0 + timedelta(minutes=30)))
        # vwap_deviation should be 0 because only one bar is in the new accumulator
        self.assertAlmostEqual(out.values["vwap_deviation"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
