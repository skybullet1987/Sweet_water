import unittest
import unittest.mock
import sys
import types
from collections import deque
from datetime import datetime, timedelta, timezone

from nextgen.core.models import Bar
from nextgen.research.harness import BarReplayHarness, ExperimentConfig

if "AlgorithmImports" not in sys.modules:
    sys.modules["AlgorithmImports"] = types.ModuleType("AlgorithmImports")
from chop_engine import ChopEngine
from regime_router import RegimeRouter
from scoring import MicroScalpEngine
import entry_exec
import order_management


class _IndicatorValue:
    def __init__(self, value):
        self.Value = value


class _Indicator:
    class _Directional:
        def __init__(self, value):
            self.Current = _IndicatorValue(value)

    def __init__(self, value, ready=True):
        self.IsReady = ready
        self.Current = _IndicatorValue(value)
        self.PositiveDirectionalIndex = self._Directional(20)
        self.NegativeDirectionalIndex = self._Directional(10)


class _AlgoStub:
    def __init__(self):
        self.market_regime = "sideways"
        self.min_signal_count = 2
        self.crypto_data = {}
        self.market_breadth = 0.5
        self._btc_sma48_window = [0] * 48

    def Debug(self, *_args, **_kwargs):
        return None


class _Symbol:
    def __init__(self, value: str):
        self.Value = value
    def __hash__(self):
        return hash(self.Value)
    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


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

    def test_payoff_floor_keeps_chop_tp_at_least_1p5x_stop(self) -> None:
        algo = _AlgoStub()
        engine = ChopEngine(algo)
        crypto = {"atr": _Indicator(1.0)}
        params = engine.get_exit_params(_Symbol("ETHUSD"), entry_price=100.0, crypto=crypto)
        self.assertGreaterEqual(params["tp"], params["sl"] * 1.5)

    def test_confirmation_bar_blocks_reversal_chop_entry(self) -> None:
        symbol = _Symbol("SOLUSD")
        placed = []

        class _Sec:
            Price = 100.0

        class _Portfolio:
            Cash = 1000.0
            TotalPortfolioValue = 1000.0
            CashBook = {"USD": types.SimpleNamespace(Amount=1000.0)}

        class _ChopEngine:
            CHOP_ENTRY_THRESHOLD = 0.25
            CHOP_MAX_TRADES_PER_SYMBOL_DAY = 3
            def is_in_fail_cooldown(self, _s): return False
            def daily_trade_count(self, _s): return 0
            def calculate_score(self, _c): return 0.40, {"range_reversion": 0.2}
            def calculate_position_size(self, _score): return 0.2
            def register_entry(self, _s): return None

        algo = types.SimpleNamespace(
            _positions_synced=True,
            LiveMode=False,
            kraken_status="online",
            max_positions=3,
            Portfolio=_Portfolio(),
            Time=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            _pending_chop_signals={},
            _symbol_entry_cooldowns={},
            _symbol_loss_cooldowns={},
            _exit_cooldowns={},
            _session_blacklist=set(),
            _recent_entry_times={},
            _pending_orders={},
            _chop_engine=_ChopEngine(),
            crypto_data={symbol: {"prices": deque([100.0], maxlen=20), "atr": _Indicator(1.0, ready=False), "trade_count_today": 0}},
            Securities={symbol: _Sec()},
            min_price_usd=0.001,
            min_notional=15.0,
            max_position_pct=0.35,
            min_notional_fee_buffer=1.5,
            daily_trade_count=0,
            max_daily_trades=8,
            _consecutive_loss_halve_remaining=0,
            disable_performance_adaptive_risk=False,
            _entry_engine={},
            _choppy_regime_entries={},
            _entry_signal_combos={},
            _recent_tickets=[],
            trade_count=0,
            expected_round_trip_fees=0.0,
            min_expected_profit_pct=0.0,
            atr_tp_mult=1.0,
            _check_correlation=lambda _s: True,
            _is_ready=lambda _c: True,
            Debug=lambda *_a, **_k: None,
        )

        with unittest.mock.patch.multiple(
            entry_exec,
            get_actual_position_count=lambda _a: 0,
            has_open_orders=lambda _a, _s: False,
            is_invested_not_dust=lambda _a, _s: False,
            spread_ok=lambda _a, _s: True,
            get_min_quantity=lambda _a, _s: 0.01,
            get_min_notional_usd=lambda _a, _s: 5.0,
            round_quantity=lambda _a, _s, q: q,
            get_spread_pct=lambda _a, _s: 0.001,
            get_session_quality=lambda _a, _h: (0.0, 1.0, 1.0),
            cancel_stale_new_orders=lambda _a: None,
            record_trade_metadata_on_entry=lambda *_a, **_k: None,
            debug_limited=lambda *_a, **_k: None,
            place_limit_or_market=lambda *_a, **_k: placed.append(1) or object(),
        ):
            entry_exec.run_chop_rebalance(algo)  # bar T: create pending signal only
            algo.Time = algo.Time + timedelta(minutes=5)
            algo.crypto_data[symbol]["prices"].append(99.0)  # bar T+1 reversed lower
            algo.Securities[symbol].Price = 99.0
            entry_exec.run_chop_rebalance(algo)

        self.assertEqual(0, len(placed))

    def test_trend_gate_reachable_ideal_setup_scores_above_threshold_plus_buffer(self) -> None:
        algo = _AlgoStub()
        algo.market_regime = "sideways"
        engine = MicroScalpEngine(algo)
        crypto = {
            "volume": deque([100.0] * 19 + [420.0], maxlen=60),
            "volume_long": deque([100.0] * 60, maxlen=60),
            "_vol_long_sum": 6000.0,
            "prices": deque([100.0] * 30 + [101.0], maxlen=60),
            "adx": _Indicator(9.0),
            "rsi": _Indicator(45.0),
            "bb_lower": deque([99.0], maxlen=10),
            "ema_ultra_short": _Indicator(102.0),
            "ema_short": _Indicator(100.0),
            "vwap": 95.0,
            "vwap_sd": 1.0,
            "vwap_sd2_lower": 92.0,
            "vwap_sd3_lower": 90.0,
            "ema_5": _Indicator(103.0),
        }
        score, _ = engine.calculate_scalp_score(crypto)
        self.assertGreaterEqual(score, 0.71)

    def test_stale_limit_escalates_to_market_when_signal_still_valid(self) -> None:
        symbol = _Symbol("ETHUSD")
        now = datetime(2026, 1, 1, 0, 0)
        order = types.SimpleNamespace(
            Symbol=symbol,
            Time=now - timedelta(seconds=31),
            Id=123,
            Tag="Entry",
            Quantity=2.0,
            QuantityFilled=0.0,
        )
        canceled = []
        marketed = []
        tx = types.SimpleNamespace(
            GetOpenOrders=lambda *_a, **_k: [order],
            CancelOrder=lambda oid: canceled.append(oid),
        )
        algo = types.SimpleNamespace(
            Time=now,
            Transactions=tx,
            _session_blacklist=set(),
            _cancel_cooldowns={},
            cancel_cooldown_minutes=1,
            _symbol_entry_cooldowns={},
            _submitted_orders={
                symbol: {
                    "is_limit_entry": True,
                    "signal_engine": "trend",
                    "signal_threshold": 0.70,
                    "signal_regime": "trend",
                }
            },
            _regime_router=types.SimpleNamespace(route=lambda: "trend"),
            _scoring_engine=types.SimpleNamespace(calculate_scalp_score=lambda _c: (0.80, {})),
            _chop_engine=types.SimpleNamespace(calculate_score=lambda _c: (0.0, {})),
            crypto_data={symbol: {"prices": deque([100.0], maxlen=10)}},
            entry_prices={},
            Portfolio={},
            stale_limit_cancels=0,
            stale_limit_escalations=0,
            escalate_stale_limits_to_market=True,
            MarketOrder=lambda s, q, tag="": marketed.append((s, q, tag)),
            Debug=lambda *_a, **_k: None,
        )
        with unittest.mock.patch.multiple(
            order_management,
            effective_stale_timeout=lambda _a: 30,
            is_invested_not_dust=lambda _a, _s: False,
        ):
            order_management.cancel_stale_new_orders(algo)
        self.assertEqual([123], canceled)
        self.assertEqual(1, len(marketed))
        self.assertIn("[StaleEsc]", marketed[0][2])


if __name__ == "__main__":
    unittest.main()
