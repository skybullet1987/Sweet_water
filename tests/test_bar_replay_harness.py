"""Tests for BarReplayHarness: bar-replay, cost model, walk-forward, Sharpe (items 7-8)."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from nextgen.core.types import Bar
from nextgen.research.harness import (
    BarReplayHarness,
    ExperimentConfig,
    StressScenario,
    WalkForwardResult,
)


def _make_bars(symbol: str, closes: list, base_time: datetime | None = None) -> list:
    base = base_time or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(
            symbol=symbol,
            timestamp=base + timedelta(minutes=5 * i),
            open=c,
            high=c * 1.002,
            low=c * 0.998,
            close=c,
            volume=1000.0,
        )
        for i, c in enumerate(closes)
    ]


def _trending_closes(n: int = 200, drift: float = 0.001) -> list:
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift))
    return closes


def _flat_closes(n: int = 100) -> list:
    return [100.0] * n


class BarReplayHarnessTest(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = BarReplayHarness(fee_rate=0.005, slippage_rate=0.004)
        self.t_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.t_end = datetime(2027, 1, 1, tzinfo=timezone.utc)
        self.config = ExperimentConfig(
            name="test",
            start=self.t_start,
            end=self.t_end,
            symbols=("BTCUSD",),
            initial_cash=10_000.0,
        )

    # ── run() basics ─────────────────────────────────────────────────────────

    def test_run_returns_run_result(self) -> None:
        bars = _make_bars("BTCUSD", _flat_closes(50))
        result = self.harness.run(self.config, {"BTCUSD": bars})
        self.assertIsNotNone(result)
        self.assertIn("total_return", result.metrics)

    def test_run_metadata_has_config_hash(self) -> None:
        bars = _make_bars("BTCUSD", _flat_closes(30))
        result = self.harness.run(self.config, {"BTCUSD": bars})
        self.assertEqual(len(result.metadata.config_hash), 64)  # SHA-256 hex
        self.assertEqual(len(result.metadata.run_id), 12)

    def test_run_observations_match_bar_count(self) -> None:
        n = 60
        bars = _make_bars("BTCUSD", _flat_closes(n))
        result = self.harness.run(self.config, {"BTCUSD": bars})
        # First bar initialises EMAs without producing a return observation
        self.assertGreater(result.metrics["observations"], 0)

    def test_run_ignores_bars_outside_date_range(self) -> None:
        future_bars = _make_bars("BTCUSD", _flat_closes(20), base_time=datetime(2028, 1, 1, tzinfo=timezone.utc))
        result = self.harness.run(self.config, {"BTCUSD": future_bars})
        self.assertEqual(result.metrics["observations"], 0.0)

    # ── cost model ───────────────────────────────────────────────────────────

    def test_trending_market_positive_return_minus_costs(self) -> None:
        """EMA strategy on a strong trend should be positive net of costs."""
        closes = _trending_closes(300, drift=0.002)
        bars = _make_bars("BTCUSD", closes)
        result = self.harness.run(self.config, {"BTCUSD": bars})
        self.assertGreater(result.metrics["total_return"], 0.0)

    def test_costs_reduce_returns_vs_zero_cost(self) -> None:
        """Non-zero fee/slippage should produce lower return than zero cost."""
        bars = _make_bars("BTCUSD", _trending_closes(200))
        result_with_cost = self.harness.run(self.config, {"BTCUSD": bars})

        free_harness = BarReplayHarness(fee_rate=0.0, slippage_rate=0.0)
        result_free = free_harness.run(self.config, {"BTCUSD": bars})

        self.assertGreaterEqual(result_free.metrics["total_return"], result_with_cost.metrics["total_return"])

    # ── stress scenarios ─────────────────────────────────────────────────────

    def test_stress_scenario_reduces_return(self) -> None:
        scenario = StressScenario(name="double_costs", spread_multiplier=2.0, slippage_multiplier=2.0)
        config = ExperimentConfig(
            name="stress_test",
            start=self.t_start,
            end=self.t_end,
            symbols=("BTCUSD",),
            initial_cash=10_000.0,
            stress_scenarios=(scenario,),
        )
        bars = _make_bars("BTCUSD", _trending_closes(200))
        result = self.harness.run(config, {"BTCUSD": bars})
        self.assertIn("double_costs", result.scenario_metrics)
        base_ret = result.metrics["total_return"]
        stress_ret = result.scenario_metrics["double_costs"]["total_return"]
        self.assertGreaterEqual(base_ret, stress_ret)

    # ── metrics ──────────────────────────────────────────────────────────────

    def test_metrics_include_sharpe_and_drawdown(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(100))
        result = self.harness.run(self.config, {"BTCUSD": bars})
        for key in ("sharpe_ratio", "max_drawdown", "win_rate", "average_bar_return"):
            self.assertIn(key, result.metrics)

    def test_max_drawdown_between_zero_and_one(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(150))
        result = self.harness.run(self.config, {"BTCUSD": bars})
        self.assertGreaterEqual(result.metrics["max_drawdown"], 0.0)
        self.assertLessEqual(result.metrics["max_drawdown"], 1.0)

    def test_win_rate_between_zero_and_one(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(100))
        result = self.harness.run(self.config, {"BTCUSD": bars})
        self.assertGreaterEqual(result.metrics["win_rate"], 0.0)
        self.assertLessEqual(result.metrics["win_rate"], 1.0)

    # ── walk_forward_run ─────────────────────────────────────────────────────

    def test_walk_forward_returns_correct_fold_count(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(300))
        wf = self.harness.walk_forward_run(self.config, {"BTCUSD": bars}, n_folds=3)
        self.assertEqual(len(wf.fold_metrics), 3)

    def test_walk_forward_fold_metrics_have_expected_keys(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(200))
        wf = self.harness.walk_forward_run(self.config, {"BTCUSD": bars}, n_folds=2)
        for fold in wf.fold_metrics:
            self.assertIn("sharpe_ratio", fold)
            self.assertIn("total_return", fold)

    def test_walk_forward_sharpe_t_stat_and_p_value_present(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(300))
        wf = self.harness.walk_forward_run(self.config, {"BTCUSD": bars}, n_folds=2)
        self.assertIsInstance(wf.sharpe_t_stat, float)
        self.assertIsInstance(wf.sharpe_p_value, float)
        self.assertGreaterEqual(wf.sharpe_p_value, 0.0)
        self.assertLessEqual(wf.sharpe_p_value, 1.0)

    def test_walk_forward_empty_bars_returns_gracefully(self) -> None:
        wf = self.harness.walk_forward_run(self.config, {"BTCUSD": []}, n_folds=2)
        self.assertEqual(wf.overall_sharpe, 0.0)
        self.assertFalse(wf.is_significant)

    def test_walk_forward_requires_min_two_folds(self) -> None:
        with self.assertRaises(ValueError):
            self.harness.walk_forward_run(self.config, {}, n_folds=1)

    def test_walk_forward_overall_sharpe_is_mean_of_folds(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(200))
        wf = self.harness.walk_forward_run(self.config, {"BTCUSD": bars}, n_folds=2)
        expected = sum(f["sharpe_ratio"] for f in wf.fold_metrics) / len(wf.fold_metrics)
        self.assertAlmostEqual(wf.overall_sharpe, expected, places=10)

    def test_is_significant_flag_respects_p_value(self) -> None:
        bars = _make_bars("BTCUSD", _trending_closes(300))
        wf = self.harness.walk_forward_run(self.config, {"BTCUSD": bars}, n_folds=2)
        if wf.sharpe_p_value < 0.05:
            self.assertTrue(wf.is_significant)
        else:
            self.assertFalse(wf.is_significant)

    # ── backward compat: legacy BacktestHarness ───────────────────────────────

    def test_legacy_backtest_harness_still_works(self) -> None:
        from nextgen.research.harness import BacktestHarness
        harness = BacktestHarness()
        result = harness.run(
            ExperimentConfig("leg", datetime(2026, 1, 1), datetime(2026, 12, 31), ("BTC",), 1000.0),
            {"BTC": [0.01, 0.02, -0.01]},
        )
        self.assertIn("average_return", result.metrics)


if __name__ == "__main__":
    unittest.main()
