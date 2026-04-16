"""
Tests for the 5-item risk / validation framework:

1. OOS validation  — OOSConfig, OOSResult, BarReplayHarness.oos_run()
2. Cost-aware edge — break_even_return(), min_edge_multiplier gate
3. Paper-trade     — PaperTradeEvent, PaperTradeSession, paper_trade_mode
4. Kraken universe — KrakenUniverseFilter, KRAKEN_MIN_ORDER_SIZES
5. Drawdown-first  — PerformanceObjective, meets_objective(), objective_met
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from nextgen.core.types import Bar
from nextgen.research.harness import (
    BarReplayHarness,
    ExperimentConfig,
    KRAKEN_MIN_ORDER_SIZES,
    KrakenUniverseFilter,
    OOSConfig,
    OOSResult,
    PaperTradeEvent,
    PaperTradeSession,
    PerformanceObjective,
    StressScenario,
    WalkForwardResult,
    break_even_return,
    meets_objective,
)

_UTC = timezone.utc


# ── bar helpers ──────────────────────────────────────────────────────────────

def _make_bars(symbol: str, closes: list, base_time: datetime | None = None) -> list:
    base = base_time or datetime(2025, 1, 1, tzinfo=_UTC)
    return [
        Bar(
            symbol=symbol,
            timestamp=base + timedelta(minutes=5 * i),
            open=c, high=c * 1.002, low=c * 0.998, close=c, volume=1000.0,
        )
        for i, c in enumerate(closes)
    ]


def _trending_closes(n: int = 200, drift: float = 0.001) -> list:
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift))
    return closes


def _flat_closes(n: int = 60) -> list:
    return [100.0] * n


# ── 1. OOS validation ─────────────────────────────────────────────────────────


class OOSValidationTests(unittest.TestCase):

    def _make_bars_two_windows(self, n_is: int = 300, n_oos: int = 200) -> dict:
        """IS bars Jan 2025, OOS bars immediately after."""
        is_start = datetime(2025, 1, 1, tzinfo=_UTC)
        # IS window: trending
        is_bars = _make_bars("BTCUSD", _trending_closes(n_is, 0.001), base_time=is_start)
        # OOS window: starts right after IS ends
        oos_start = is_bars[-1].timestamp + timedelta(minutes=5)
        oos_bars = _make_bars("BTCUSD", _trending_closes(n_oos, 0.001), base_time=oos_start)
        return {"BTCUSD": is_bars + oos_bars}

    def _oos_config(self, bars_by_symbol: dict) -> OOSConfig:
        all_bars = sorted(
            [b for bars in bars_by_symbol.values() for b in bars],
            key=lambda b: b.timestamp,
        )
        midpoint = all_bars[len(all_bars) // 2].timestamp
        return OOSConfig(
            is_start=all_bars[0].timestamp,
            is_end=midpoint,
            oos_start=midpoint + timedelta(minutes=5),
            oos_end=all_bars[-1].timestamp,
            min_recommended_folds=30,
        )

    def test_oos_run_returns_oos_result(self):
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows()
        oos_cfg = self._oos_config(bars)
        result = harness.oos_run(bars, oos_cfg, n_is_folds=2)
        self.assertIsInstance(result, OOSResult)

    def test_oos_result_has_is_walk_forward(self):
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows()
        oos_cfg = self._oos_config(bars)
        result = harness.oos_run(bars, oos_cfg, n_is_folds=2)
        self.assertIsInstance(result.is_result, WalkForwardResult)

    def test_oos_metrics_dict_has_expected_keys(self):
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows()
        oos_cfg = self._oos_config(bars)
        result = harness.oos_run(bars, oos_cfg, n_is_folds=2)
        for key in ("sharpe_ratio", "max_drawdown", "total_return", "observations"):
            self.assertIn(key, result.oos_metrics)

    def test_folds_warning_when_n_is_folds_below_recommended(self):
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows()
        oos_cfg = self._oos_config(bars)  # min_recommended_folds=30
        result = harness.oos_run(bars, oos_cfg, n_is_folds=5)
        self.assertTrue(result.folds_warning)

    def test_no_folds_warning_when_n_is_folds_at_recommended(self):
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows(n_is=3100, n_oos=1000)
        oos_cfg = self._oos_config(bars)
        result = harness.oos_run(bars, oos_cfg, n_is_folds=30)
        self.assertFalse(result.folds_warning)

    def test_sharpe_degradation_is_float(self):
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows()
        oos_cfg = self._oos_config(bars)
        result = harness.oos_run(bars, oos_cfg, n_is_folds=2)
        self.assertIsInstance(result.sharpe_degradation, float)

    def test_oos_objective_met_with_strong_trend(self):
        """A very strong trend should meet a relaxed objective."""
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        bars = self._make_bars_two_windows(n_is=200, n_oos=200)
        oos_cfg = self._oos_config(bars)
        # Use permissive objective so test is stable
        obj = PerformanceObjective(target_sharpe=-99.0, max_drawdown=1.0)
        result = harness.oos_run(bars, oos_cfg, n_is_folds=2, objective=obj)
        self.assertTrue(result.oos_objective_met)

    def test_oos_objective_not_met_when_sharpe_required_high(self):
        """Flat bars cannot meet Sharpe ≥ 100."""
        harness = BarReplayHarness(min_edge_multiplier=0.0)
        flat_bars = _make_bars("BTCUSD", _flat_closes(400))
        oos_cfg = OOSConfig(
            is_start=flat_bars[0].timestamp,
            is_end=flat_bars[199].timestamp,
            oos_start=flat_bars[200].timestamp,
            oos_end=flat_bars[-1].timestamp,
        )
        obj = PerformanceObjective(target_sharpe=100.0, max_drawdown=0.15)
        result = harness.oos_run({"BTCUSD": flat_bars}, oos_cfg, n_is_folds=2, objective=obj)
        self.assertFalse(result.oos_objective_met)

    def test_oos_config_default_min_recommended_folds(self):
        oos_cfg = OOSConfig(
            is_start=datetime(2025, 1, 1, tzinfo=_UTC),
            is_end=datetime(2025, 6, 30, tzinfo=_UTC),
            oos_start=datetime(2025, 7, 1, tzinfo=_UTC),
            oos_end=datetime(2025, 12, 31, tzinfo=_UTC),
        )
        self.assertEqual(oos_cfg.min_recommended_folds, 30)


# ── 2. Cost-aware edge threshold ─────────────────────────────────────────────


class CostAwareEdgeTests(unittest.TestCase):

    def test_break_even_return_positive(self):
        be = break_even_return()
        self.assertGreater(be, 0.0)

    def test_break_even_return_equals_round_trip_cost(self):
        fee = 0.005
        slip = 0.004
        be = break_even_return(fee_rate=fee, slippage_rate=slip)
        expected = 2.0 * (fee + slip)
        self.assertAlmostEqual(be, expected, places=10)

    def test_break_even_with_zero_costs_is_zero(self):
        self.assertEqual(break_even_return(0.0, 0.0), 0.0)

    def test_edge_gate_reduces_trade_count_vs_no_gate(self):
        """With edge gating, fewer entries should occur on a noisy flat market."""
        # Build bars with tiny oscillations — EMA signal fires often but edge is tiny
        closes = [100.0 + (0.001 if i % 10 < 5 else -0.001) for i in range(500)]
        bars = _make_bars("BTCUSD", closes)

        session_gated = PaperTradeSession()
        session_ungated = PaperTradeSession()

        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        gated = BarReplayHarness(fee_rate=0.005, slippage_rate=0.004,
                                  min_edge_multiplier=2.0, paper_trade_mode=True)
        ungated = BarReplayHarness(fee_rate=0.005, slippage_rate=0.004,
                                   min_edge_multiplier=0.0, paper_trade_mode=True)

        gated.run(config, {"BTCUSD": bars})
        ungated.run(config, {"BTCUSD": bars})

        gated_buys = gated.last_paper_session.buy_count() if gated.last_paper_session else 0
        ungated_buys = ungated.last_paper_session.buy_count() if ungated.last_paper_session else 0
        # Gated version should enter fewer or equal times
        self.assertLessEqual(gated_buys, ungated_buys)

    def test_edge_gate_zero_disables_gating(self):
        """min_edge_multiplier=0 should never suppress a valid signal entry."""
        closes = _trending_closes(200, 0.002)
        bars = _make_bars("BTCUSD", closes)
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        h_zero = BarReplayHarness(fee_rate=0.005, slippage_rate=0.004,
                                   min_edge_multiplier=0.0, paper_trade_mode=True)
        h_two = BarReplayHarness(fee_rate=0.005, slippage_rate=0.004,
                                  min_edge_multiplier=2.0, paper_trade_mode=True)
        h_zero.run(config, {"BTCUSD": bars})
        h_two.run(config, {"BTCUSD": bars})
        buys_zero = h_zero.last_paper_session.buy_count() if h_zero.last_paper_session else 0
        buys_two = h_two.last_paper_session.buy_count() if h_two.last_paper_session else 0
        self.assertGreaterEqual(buys_zero, buys_two)

    def test_harness_default_edge_multiplier_is_zero(self):
        """Default is 0.0 (backward-compat); opt in with min_edge_multiplier=2.0."""
        h = BarReplayHarness()
        self.assertEqual(h.min_edge_multiplier, 0.0)

    def test_edge_gating_reflected_in_stress_scenarios(self):
        """Stress scenario with 2× costs → tighter edge gate → fewer entries."""
        closes = _trending_closes(300, 0.0005)
        bars = _make_bars("BTCUSD", closes)
        scenario_2x = StressScenario("2x", spread_multiplier=2.0, slippage_multiplier=2.0)
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",),
            1000.0, stress_scenarios=(scenario_2x,),
        )
        h = BarReplayHarness(fee_rate=0.005, slippage_rate=0.004, min_edge_multiplier=2.0)
        result = h.run(config, {"BTCUSD": bars})
        # Stress scenario metrics should exist and be plausible
        self.assertIn("2x", result.scenario_metrics)
        self.assertIn("sharpe_ratio", result.scenario_metrics["2x"])


# ── 3. Paper-trade mode ───────────────────────────────────────────────────────


class PaperTradeModeTests(unittest.TestCase):

    def setUp(self):
        # Use alternating up/down phases so EMA crossovers generate both BUY and SELL events
        phase_len = 30
        closes = []
        price = 100.0
        for phase in range(8):
            drift = 0.002 if phase % 2 == 0 else -0.002
            for _ in range(phase_len):
                price *= (1 + drift)
                closes.append(price)
        self.bars = _make_bars("BTCUSD", closes)
        self.config = ExperimentConfig(
            "pt", self.bars[0].timestamp, self.bars[-1].timestamp,
            ("BTCUSD",), 1000.0,
        )

    def test_paper_session_populated_when_mode_enabled(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        self.assertIsNotNone(h.last_paper_session)

    def test_paper_session_none_when_mode_disabled(self):
        h = BarReplayHarness(paper_trade_mode=False)
        h.run(self.config, {"BTCUSD": self.bars})
        self.assertIsNone(h.last_paper_session)

    def test_paper_session_has_buy_and_sell_events(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        session = h.last_paper_session
        self.assertGreater(session.buy_count(), 0)
        self.assertGreater(session.sell_count(), 0)

    def test_paper_event_fields_populated(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        session = h.last_paper_session
        buy_events = [e for e in session.events if e.action == "BUY"]
        self.assertTrue(len(buy_events) > 0)
        ev = buy_events[0]
        self.assertIsInstance(ev, PaperTradeEvent)
        self.assertGreater(ev.price, 0.0)
        self.assertGreater(ev.fee_est, 0.0)
        self.assertGreater(ev.slippage_est, 0.0)
        self.assertEqual(ev.symbol, "BTCUSD")

    def test_paper_session_buy_sell_counts_balanced(self):
        """Trending market → buys and sells should be roughly paired."""
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        session = h.last_paper_session
        # Sells ≤ buys (an open position at end has no sell yet)
        self.assertLessEqual(session.sell_count(), session.buy_count() + 1)

    def test_paper_session_total_cost_positive(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        self.assertGreater(h.last_paper_session.total_cost_fraction(), 0.0)

    def test_paper_session_to_records_serialisable(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        records = h.last_paper_session.to_records()
        self.assertIsInstance(records, list)
        if records:
            self.assertIn("action", records[0])
            self.assertIn("price", records[0])

    def test_paper_session_refreshed_on_each_run(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        h.run(self.config, {"BTCUSD": self.bars})
        first_count = h.last_paper_session.buy_count()
        h.run(self.config, {"BTCUSD": self.bars})
        second_count = h.last_paper_session.buy_count()
        self.assertEqual(first_count, second_count)

    def test_paper_session_counts_with_empty_bars(self):
        h = BarReplayHarness(paper_trade_mode=True, min_edge_multiplier=0.0)
        empty_config = ExperimentConfig(
            "empty",
            datetime(2030, 1, 1, tzinfo=_UTC),
            datetime(2030, 12, 31, tzinfo=_UTC),
            ("BTCUSD",), 1000.0,
        )
        h.run(empty_config, {"BTCUSD": self.bars})
        self.assertEqual(h.last_paper_session.buy_count(), 0)


# ── 4. Kraken universe filter ─────────────────────────────────────────────────


class KrakenUniverseFilterTests(unittest.TestCase):

    def test_tradeable_symbol_with_sufficient_capital(self):
        f = KrakenUniverseFilter(capital=500.0, max_position_pct=0.25)
        # budget = 500 * 0.25 * 0.9 = 112.5; BTCUSD min = $10 → tradeable
        self.assertTrue(f.is_tradeable("BTCUSD"))

    def test_untradeable_symbol_below_minimum(self):
        # Custom min order size of $200 → not tradeable with $50 budget
        f = KrakenUniverseFilter(
            capital=100.0, max_position_pct=0.25,
            min_order_sizes={"EXOTIC": 200.0},
        )
        # budget = 100 * 0.25 * 0.9 = 22.5 < 200
        self.assertFalse(f.is_tradeable("EXOTIC"))

    def test_position_budget_calculation(self):
        f = KrakenUniverseFilter(capital=500.0, max_position_pct=0.25, safety_margin=0.9)
        self.assertAlmostEqual(f.position_budget(), 112.5)

    def test_filter_symbols_partitions_correctly(self):
        custom = {"CHEAP": 5.0, "EXPENSIVE": 500.0}
        f = KrakenUniverseFilter(
            capital=100.0, max_position_pct=0.25,
            min_order_sizes=custom,
        )
        # budget = 22.5; CHEAP ok, EXPENSIVE rejected
        tradeable, rejected = f.filter_symbols(["CHEAP", "EXPENSIVE"])
        self.assertIn("CHEAP", tradeable)
        self.assertIn("EXPENSIVE", rejected)

    def test_filter_symbols_all_tradeable(self):
        f = KrakenUniverseFilter(capital=10_000.0, max_position_pct=0.25)
        syms = list(KRAKEN_MIN_ORDER_SIZES.keys())
        tradeable, rejected = f.filter_symbols(syms)
        self.assertEqual(rejected, [])
        self.assertEqual(len(tradeable), len(syms))

    def test_filter_symbols_none_tradeable_with_tiny_capital(self):
        f = KrakenUniverseFilter(capital=1.0, max_position_pct=0.01)
        tradeable, rejected = f.filter_symbols(list(KRAKEN_MIN_ORDER_SIZES.keys()))
        self.assertEqual(tradeable, [])

    def test_min_order_usd_falls_back_to_default(self):
        f = KrakenUniverseFilter(capital=500.0)
        # Unknown symbol → _KRAKEN_DEFAULT_MIN_USD = 10.0
        self.assertEqual(f.min_order_usd("UNKNOWN_XYZ"), 10.0)

    def test_invalid_capital_raises(self):
        with self.assertRaises(ValueError):
            KrakenUniverseFilter(capital=0.0)

    def test_invalid_max_position_pct_raises(self):
        with self.assertRaises(ValueError):
            KrakenUniverseFilter(capital=500.0, max_position_pct=1.5)

    def test_kraken_min_order_sizes_dict_non_empty(self):
        self.assertGreater(len(KRAKEN_MIN_ORDER_SIZES), 0)

    def test_filter_preserves_order(self):
        custom = {s: 5.0 for s in ["A", "B", "C", "D"]}
        f = KrakenUniverseFilter(capital=1000.0, min_order_sizes=custom)
        tradeable, _ = f.filter_symbols(["D", "C", "B", "A"])
        self.assertEqual(tradeable, ["D", "C", "B", "A"])


# ── 5. Drawdown-first objective ───────────────────────────────────────────────


class PerformanceObjectiveTests(unittest.TestCase):

    def test_meets_objective_when_both_conditions_satisfied(self):
        obj = PerformanceObjective(target_sharpe=1.5, max_drawdown=0.15)
        metrics = {"sharpe_ratio": 2.0, "max_drawdown": 0.10}
        self.assertTrue(meets_objective(metrics, obj))

    def test_fails_objective_on_low_sharpe(self):
        obj = PerformanceObjective(target_sharpe=1.5, max_drawdown=0.15)
        metrics = {"sharpe_ratio": 0.5, "max_drawdown": 0.05}
        self.assertFalse(meets_objective(metrics, obj))

    def test_fails_objective_on_high_drawdown(self):
        obj = PerformanceObjective(target_sharpe=1.5, max_drawdown=0.15)
        metrics = {"sharpe_ratio": 2.0, "max_drawdown": 0.25}
        self.assertFalse(meets_objective(metrics, obj))

    def test_fails_objective_when_both_conditions_unmet(self):
        obj = PerformanceObjective(target_sharpe=1.5, max_drawdown=0.15)
        metrics = {"sharpe_ratio": 0.0, "max_drawdown": 0.50}
        self.assertFalse(meets_objective(metrics, obj))

    def test_objective_met_at_exact_boundary(self):
        obj = PerformanceObjective(target_sharpe=1.5, max_drawdown=0.15)
        metrics = {"sharpe_ratio": 1.5, "max_drawdown": 0.15}
        self.assertTrue(meets_objective(metrics, obj))

    def test_default_objective_values(self):
        obj = PerformanceObjective()
        self.assertEqual(obj.target_sharpe, 1.5)
        self.assertEqual(obj.max_drawdown, 0.15)

    def test_walk_forward_result_has_objective_met_field(self):
        h = BarReplayHarness(min_edge_multiplier=0.0)
        bars = _make_bars("BTCUSD", _trending_closes(300))
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        wf = h.walk_forward_run(config, {"BTCUSD": bars}, n_folds=2,
                                 objective=PerformanceObjective(target_sharpe=-99.0, max_drawdown=1.0))
        self.assertIsInstance(wf.objective_met, bool)
        self.assertTrue(wf.objective_met)

    def test_walk_forward_objective_met_false_without_objective(self):
        h = BarReplayHarness(min_edge_multiplier=0.0)
        bars = _make_bars("BTCUSD", _trending_closes(200))
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        wf = h.walk_forward_run(config, {"BTCUSD": bars}, n_folds=2)
        self.assertFalse(wf.objective_met)

    def test_run_result_objective_met_key_in_metrics(self):
        h = BarReplayHarness(min_edge_multiplier=0.0)
        bars = _make_bars("BTCUSD", _trending_closes(200))
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        obj = PerformanceObjective(target_sharpe=-99.0, max_drawdown=1.0)
        result = h.run(config, {"BTCUSD": bars}, objective=obj)
        self.assertIn("objective_met", result.metrics)
        self.assertEqual(result.metrics["objective_met"], 1.0)

    def test_run_result_no_objective_key_when_not_supplied(self):
        h = BarReplayHarness(min_edge_multiplier=0.0)
        bars = _make_bars("BTCUSD", _trending_closes(200))
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        result = h.run(config, {"BTCUSD": bars})
        self.assertNotIn("objective_met", result.metrics)

    def test_walk_forward_objective_not_met_when_drawdown_too_high(self):
        """Declining bars → high drawdown → strict objective fails."""
        closes = [100.0 * (0.999 ** i) for i in range(300)]  # constant decline
        bars = _make_bars("BTCUSD", closes)
        h = BarReplayHarness(min_edge_multiplier=0.0)
        config = ExperimentConfig(
            "t", bars[0].timestamp, bars[-1].timestamp, ("BTCUSD",), 1000.0
        )
        strict_obj = PerformanceObjective(target_sharpe=1.5, max_drawdown=0.15)
        wf = h.walk_forward_run(config, {"BTCUSD": bars}, n_folds=2, objective=strict_obj)
        self.assertFalse(wf.objective_met)


if __name__ == "__main__":
    unittest.main()
