"""Tests for DiagonalCovarianceEstimator (item 4)."""
from __future__ import annotations

import math
import unittest

from nextgen.portfolio.covariance import DiagonalCovarianceEstimator


class CovarianceEstimatorTest(unittest.TestCase):

    def setUp(self) -> None:
        # Use annualization_factor=1 so realized_vol == per-bar std (easier to verify)
        self.est = DiagonalCovarianceEstimator(vol_window=10, annualization_factor=1)

    def test_zero_vol_when_insufficient_history(self) -> None:
        self.est.update("BTC", 0.01)
        self.assertEqual(self.est.realized_vol("BTC"), 0.0)

    def test_zero_vol_for_unknown_symbol(self) -> None:
        self.assertEqual(self.est.realized_vol("UNKNOWN"), 0.0)

    def test_realized_vol_matches_sample_std(self) -> None:
        returns = [0.01, -0.02, 0.03, -0.01, 0.00, 0.02, -0.03, 0.01, 0.02, -0.01]
        for r in returns:
            self.est.update("ETH", r)

        n = len(returns)
        mean = sum(returns) / n
        expected_std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        # annualization_factor=1 → no scaling
        self.assertAlmostEqual(self.est.realized_vol("ETH"), expected_std, places=10)

    def test_rolling_window_respects_maxlen(self) -> None:
        """Oldest returns should fall out of the window."""
        for _ in range(20):
            self.est.update("SOL", 0.05)
        # Most recent 10 bars: all 0.05 → zero variance → zero vol
        self.assertAlmostEqual(self.est.realized_vol("SOL"), 0.0, places=10)

    def test_portfolio_vol_scales_with_weight(self) -> None:
        # Use returns with enough spread so realized_vol > vol_floor
        returns = [0.01 * ((-1) ** i * (1 + i * 0.5)) for i in range(10)]
        for r in returns:
            self.est.update("BTC", r)
        vol_single = self.est.realized_vol("BTC")

        # portfolio_vol_estimate applies the same vol (no floor needed if vol > floor)
        # Set vol_floor below the actual vol to test pure scaling
        port_vol = self.est.portfolio_vol_estimate({"BTC": 0.5}, vol_floor=0.0)
        self.assertAlmostEqual(port_vol, 0.5 * vol_single, places=10)

    def test_portfolio_vol_uses_vol_floor(self) -> None:
        """Symbol with only 1 return uses vol_floor in portfolio vol estimate."""
        self.est.update("NEW", 0.01)   # only 1 observation
        floor = 0.05
        port_vol = self.est.portfolio_vol_estimate({"NEW": 0.2}, vol_floor=floor)
        self.assertAlmostEqual(port_vol, 0.2 * floor, places=10)

    def test_multiple_symbols_independent(self) -> None:
        for r in [0.01, -0.02, 0.03, -0.01, 0.02, -0.02, 0.01, 0.00, 0.02, -0.01]:
            self.est.update("BTC", r)
        for r in [0.05] * 10:
            self.est.update("ETH", r)

        btc_vol = self.est.realized_vol("BTC")
        eth_vol = self.est.realized_vol("ETH")
        self.assertGreater(btc_vol, 0.0)
        self.assertAlmostEqual(eth_vol, 0.0, places=10)  # constant returns → 0 vol

    def test_update_price_computes_return(self) -> None:
        # Feed 10 prices starting at 100, each up by 1
        prev = 100.0
        for i in range(1, 11):
            new = 100.0 + i
            self.est.update_price("XRP", new, prev)
            prev = new
        # All returns are ~1/100..1/109 ≈ constant → very low vol
        self.assertGreater(self.est.realized_vol("XRP"), 0.0)

    def test_known_symbols(self) -> None:
        self.est.update("BTC", 0.01)
        self.est.update("ETH", 0.02)
        syms = self.est.known_symbols()
        self.assertIn("BTC", syms)
        self.assertIn("ETH", syms)

    def test_annualization_applied(self) -> None:
        """With annualization_factor=100, vol is scaled by sqrt(100)=10."""
        est = DiagonalCovarianceEstimator(vol_window=10, annualization_factor=100)
        returns = [0.01, -0.02, 0.03, -0.01, 0.00, 0.02, -0.03, 0.01, 0.02, -0.01]
        for r in returns:
            est.update("BTC", r)

        n = len(returns)
        mean = sum(returns) / n
        base_std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        expected = base_std * math.sqrt(100)
        self.assertAlmostEqual(est.realized_vol("BTC"), expected, places=10)


if __name__ == "__main__":
    unittest.main()
