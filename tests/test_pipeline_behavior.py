import unittest
from datetime import datetime, timedelta, timezone

from nextgen.core.models import Bar
from nextgen.research.harness import BarReplayHarness, ExperimentConfig
from regime_router import RegimeRouter


class _Indicator:
    class _Value:
        def __init__(self, value):
            self.Value = value

    def __init__(self, value, ready=True):
        self.IsReady = ready
        self.Current = self._Value(value)
        self.PositiveDirectionalIndex = self._Value(20)
        self.NegativeDirectionalIndex = self._Value(10)


class _AlgoStub:
    def __init__(self):
        self.market_regime = "sideways"
        self.min_signal_count = 2
        self.crypto_data = {}
        self.market_breadth = 0.5
        self._btc_sma48_window = [0] * 48

    def Debug(self, *_args, **_kwargs):
        return None


def _bars(closes):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(
            symbol="BTCUSD",
            timestamp=base + timedelta(minutes=5 * i),
            open=c,
            high=c * 1.001,
            low=c * 0.999,
            close=c,
            volume=1000.0,
        )
        for i, c in enumerate(closes)
    ]


class PipelineBehaviorTests(unittest.TestCase):
    def test_score_to_order_behavior_places_expected_round_trip(self) -> None:
        closes = [100 + i * 0.5 for i in range(80)] + [140 - i * 0.7 for i in range(80)]
        bars = _bars(closes)
        cfg = ExperimentConfig(
            name="score_to_order",
            start=bars[0].timestamp,
            end=bars[-1].timestamp,
            symbols=("BTCUSD",),
            initial_cash=10_000.0,
        )
        harness = BarReplayHarness(fee_rate=0.001, slippage_rate=0.001, paper_trade_mode=True)
        harness.run(cfg, {"BTCUSD": bars})
        session = harness.last_paper_session
        self.assertEqual(1, session.buy_count())
        self.assertEqual(1, session.sell_count())
        self.assertEqual(["BUY", "SELL"], [e.action for e in session.events])
        self.assertAlmostEqual(0.004, session.total_cost_fraction(), places=6)

    def test_regime_router_transitions_trend_transition_chop(self) -> None:
        algo = _AlgoStub()
        for i in range(8):
            algo.crypto_data[f"S{i}"] = {"adx": _Indicator(24)}
        router = RegimeRouter(algo)

        algo.market_regime = "bull"
        for _ in range(5):
            router.update()
        self.assertEqual("trend", router.route())

        algo.market_regime = "unknown"
        for _ in range(4):
            router.update()
        self.assertEqual("transition", router.route())

        algo.market_regime = "sideways"
        algo.market_breadth = 0.5
        for v in algo.crypto_data.values():
            v["adx"] = _Indicator(14)
        for _ in range(5):
            router.update()
        self.assertEqual("chop", router.route())

    def test_cost_accounting_matches_gross_minus_modeled_costs(self) -> None:
        closes = [100] * 30 + [101] * 10 + [100] * 10
        bars = _bars(closes)
        cfg = ExperimentConfig(
            name="cost_accounting",
            start=bars[0].timestamp,
            end=bars[-1].timestamp,
            symbols=("BTCUSD",),
            initial_cash=10_000.0,
        )
        gross_harness = BarReplayHarness(fee_rate=0.0, slippage_rate=0.0, paper_trade_mode=True)
        gross = gross_harness.run(cfg, {"BTCUSD": bars}).metrics["total_return"]

        net_harness = BarReplayHarness(fee_rate=0.001, slippage_rate=0.001, paper_trade_mode=True)
        net = net_harness.run(cfg, {"BTCUSD": bars}).metrics["total_return"]
        modeled_cost = net_harness.last_paper_session.total_cost_fraction()

        self.assertAlmostEqual(gross - net, modeled_cost, delta=1e-4)

    def test_zero_trade_regression_guard_now_produces_trade(self) -> None:
        closes = [100 + i * 0.45 for i in range(60)] + [127 - i * 0.6 for i in range(50)]
        bars = _bars(closes)
        cfg = ExperimentConfig(
            name="zero_trade_guard",
            start=bars[0].timestamp,
            end=bars[-1].timestamp,
            symbols=("BTCUSD",),
            initial_cash=10_000.0,
        )
        harness = BarReplayHarness(fee_rate=0.001, slippage_rate=0.001, paper_trade_mode=True)
        harness.run(cfg, {"BTCUSD": bars})
        self.assertGreater(harness.last_paper_session.buy_count(), 0)


if __name__ == "__main__":
    unittest.main()
