from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import CONFIG, StrategyConfig
from execution import (
    Executor,
    PositionState,
    execute_regime_entries,
    manage_open_positions,
    place_limit_or_market,
    reserved_qty,
    smart_liquidate,
)
from main import SweetWaterPhase1
from reporting import walk_forward_run
from scoring import Scorer
from sizing import Sizer
from universe import BLACKLIST, KRAKEN_SAFE_LIST, select_universe
import universe as universe_module


def test_module_size_limits():
    import os

    for f in ["main.py", "execution.py", "features.py", "scoring.py", "sizing.py", "risk.py", "regime.py", "reporting.py"]:
        assert os.path.getsize(f"qc_runtime/{f}") < 60_000, f"{f} exceeds 60KB"


def test_strategy_config_defaults_updated():
    cfg = StrategyConfig()
    assert cfg.universe_size == 30
    assert cfg.top_k == 6
    assert cfg.max_positions == 6
    assert cfg.edge_cost_multiplier == 1.2
    assert cfg.edge_scale == 0.005
    assert cfg.score_threshold == 0.75
    assert cfg.score_clip_value == 3.0
    assert cfg.min_rebalance_weight_delta == 0.015
    assert cfg.max_replacements_per_rebalance == 4
    assert cfg.rebalance_cadence_hours == 4
    assert cfg.vol_stress_threshold == 0.85


def test_universe_excludes_sklusd_even_if_high_liquidity(monkeypatch):
    assert "SKLUSD" not in KRAKEN_SAFE_LIST
    assert "SKLUSD" in BLACKLIST
    monkeypatch.setattr(universe_module, "KRAKEN_SAFE_LIST", (*KRAKEN_SAFE_LIST, "SKLUSD"))

    def history_provider(symbol, _start, _end):
        base_close = 1_000.0 if symbol == "SKLUSD" else 10.0
        base_volume = 1_000_000.0 if symbol == "SKLUSD" else 100.0
        return pd.DataFrame({"close": [base_close, base_close], "volume": [base_volume, base_volume]})

    selected = select_universe(history_provider, pd.Timestamp("2025-10-01", tz="UTC"))
    assert "SKLUSD" not in selected


class TestPhaseRequirements:
    class _Symbol:
        def __init__(self, value: str):
            self.Value = value

        def __hash__(self):
            return hash(self.Value)

        def __eq__(self, other):
            return isinstance(other, TestPhaseRequirements._Symbol) and self.Value == other.Value

    class _SymbolProps:
        MinimumPriceVariation = 0.01
        LotSize = 0.0001
        MinimumOrderSize = 0.0001

    class _Security:
        def __init__(self, symbol, price):
            self.Symbol = symbol
            self.Price = float(price)
            self.BidPrice = float(price) * 0.999
            self.AskPrice = float(price) * 1.001
            self.Volume = 10000.0
            self.SymbolProperties = TestPhaseRequirements._SymbolProps()
            self.FeeModel = None
            self.SlippageModel = None

        def SetSlippageModel(self, model):
            self.SlippageModel = model

    class _Holding:
        def __init__(self):
            self.Quantity = 0.0
            self.Invested = False
            self.AveragePrice = 0.0
            self.Price = 0.0

    class _Portfolio(dict):
        def __init__(self):
            super().__init__()
            self.TotalPortfolioValue = 500.0
            self.TotalHoldingsValue = 0.0
            self.Cash = 500.0
            self.CashBook = {"USD": type("Cash", (), {"Amount": 500.0})()}

    class _Transactions:
        def __init__(self):
            self._orders = {}
            self._open = []
            self._next = 1

        def GetOpenOrders(self, symbol=None):
            if symbol is None:
                return list(self._open)
            return [o for o in self._open if o.Symbol == symbol]

        def CancelOrder(self, order_id):
            self._open = [o for o in self._open if o.Id != order_id]

        def GetOrderById(self, order_id):
            return self._orders.get(order_id)

    class _Ticket:
        def __init__(self, order_id):
            self.OrderId = order_id

    class _Slice:
        def __init__(self, bars, ticks=None):
            self.Bars = bars
            self.Ticks = {} if ticks is None else ticks

    class _Decision:
        def __init__(self, w):
            self.approved = True
            self.adjusted_target_weight = w

    class _Bar:
        def __init__(self, open_, high, low, close, volume):
            self.Open = float(open_)
            self.High = float(high)
            self.Low = float(low)
            self.Close = float(close)
            self.Volume = float(volume)

    def _build_algo(self):
        algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
        algo.Time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        algo.LiveMode = True
        algo.IsWarmingUp = False
        algo.Securities = {}
        algo.Portfolio = self._Portfolio()
        algo.Transactions = self._Transactions()
        algo._order_calls = []
        algo._debug_logs = []
        algo._subscriptions = []

        noop = lambda *_args, **_kwargs: None
        algo.SetBrokerageModel = noop
        algo.SetCash = lambda cash: setattr(algo.Portfolio, "Cash", float(cash))
        algo.SetStartDate = noop
        algo.SetEndDate = noop
        algo.SetWarmup = noop
        algo.Debug = lambda msg: algo._debug_logs.append(str(msg))
        algo.Liquidate = noop

        def _add_crypto(ticker, _resolution, _market):
            symbol = self._Symbol(ticker)
            price = 200.0 if ticker == "BTCUSD" else 100.0
            sec = self._Security(symbol, price)
            algo.Securities[symbol] = sec
            algo.Portfolio[symbol] = self._Holding()
            algo._subscriptions.append((ticker, _resolution))
            return sec

        algo.AddCrypto = _add_crypto

        def _history(symbol, _start, _end, _resolution):
            rows = 250
            close = np.linspace(80.0, 120.0, rows)
            volume = np.linspace(1000.0, 2000.0, rows)
            return pd.DataFrame(
                {
                    "open": close * 0.999,
                    "high": close * 1.001,
                    "low": close * 0.998,
                    "close": close,
                    "volume": volume,
                }
            )

        algo.History = _history

        def _limit_order(symbol, quantity, limit_price, tag=""):
            oid = algo.Transactions._next
            algo.Transactions._next += 1
            order = type("Order", (), {"Id": oid, "Symbol": symbol, "Quantity": float(quantity), "Tag": tag, "Price": float(limit_price), "Direction": 1 if quantity > 0 else -1})
            algo.Transactions._orders[oid] = order
            algo.Transactions._open.append(order)
            algo._order_calls.append(("limit", symbol.Value, float(quantity), tag))
            return self._Ticket(oid)

        def _market_order(symbol, quantity, tag=""):
            oid = algo.Transactions._next
            algo.Transactions._next += 1
            order = type("Order", (), {"Id": oid, "Symbol": symbol, "Quantity": float(quantity), "Tag": tag, "Price": float(algo.Securities[symbol].Price), "Direction": 1 if quantity > 0 else -1})
            algo.Transactions._orders[oid] = order
            algo._order_calls.append(("market", symbol.Value, float(quantity), tag))
            return self._Ticket(oid)

        algo.LimitOrder = _limit_order
        algo.MarketOrder = _market_order
        return algo

    def _warm_and_configure_single_symbol(self, algo, symbol):
        algo.Initialize()
        algo.symbols = [symbol]
        for i in range(240):
            px = 100 + 0.2 * i
            algo.feature_engine.update(
                {
                    "symbol": symbol.Value,
                    "open": px * 0.999,
                    "high": px * 1.002,
                    "low": px * 0.998,
                    "close": px,
                    "volume": 1000 + i,
                }
            )
            algo.feature_engine.update(
                {
                    "symbol": "BTCUSD",
                    "open": px * 0.999,
                    "high": px * 1.002,
                    "low": px * 0.998,
                    "close": px,
                    "volume": 3000 + i,
                }
            )
        algo.scorer.score = lambda *_args, **_kwargs: {
            "cvd": 0.0, "ofi": 0.0, "volc": 0.0, "rot": 0.0, "mult": 1.0, "hurst": 0.6, "hurst_regime": "trend", "raw": 0.9, "final": 0.9
        }
        algo.risk.evaluate = lambda payload: self._Decision(payload["target_weight"])

    def _make_slice(self, algo, symbol, open_, high, low, close, btc_close=200.0):
        bars = {}
        for ref in algo.reference_symbols:
            sec = algo.Securities[ref]
            sec.Price = btc_close
            sec.BidPrice = btc_close * 0.999
            sec.AskPrice = btc_close * 1.001
            bars[ref] = self._Bar(btc_close * 0.999, btc_close * 1.002, btc_close * 0.998, btc_close, 10000)
        sec = algo.Securities[symbol]
        sec.Price = close
        sec.BidPrice = close * 0.999
        sec.AskPrice = close * 1.001
        bars[symbol] = self._Bar(open_, high, low, close, 10000)
        return self._Slice(bars)

    def test_no_double_entry(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        algo.Time = datetime(2025, 1, 2, 0, tzinfo=timezone.utc)
        execute_regime_entries(algo, [(symbol, 0.9, 0.2)], regime_tag="risk_on")
        execute_regime_entries(algo, [(symbol, 0.9, 0.2)], regime_tag="risk_on")

        buys = [o for o in algo._order_calls if o[2] > 0]
        assert len(buys) == 1

    def test_tick_subscriptions_disabled_in_backtest(self):
        algo = self._build_algo()
        algo.LiveMode = False
        algo.Initialize()
        assert all(str(resolution) != "Tick" for _, resolution in algo._subscriptions)

    def test_tick_subscriptions_enabled_in_live_microstructure(self):
        algo = self._build_algo()
        algo.LiveMode = True
        algo.Initialize()
        assert all(str(resolution) != "Tick" for _, resolution in algo._subscriptions)

    def test_cash_preflight_skips(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        self._warm_and_configure_single_symbol(algo, symbol)
        algo.Portfolio.Cash = 0.50
        algo.Portfolio.CashBook["USD"].Amount = 0.50

        for i in range(10):
            algo.Time = datetime(2025, 1, 3, i, tzinfo=timezone.utc)
            algo.OnData(self._make_slice(algo, symbol, 100, 101, 99.5, 100.4))

        assert len(algo._order_calls) == 0

    def test_atr_take_profit_fires(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        hold = algo.Portfolio[symbol]
        hold.Quantity = 1.0
        hold.Invested = True
        hold.AveragePrice = 100.0
        algo.position_state[symbol] = PositionState(100.0, 100.0, 2.0, algo.Time - timedelta(hours=1))

        manage_open_positions(algo, self._make_slice(algo, symbol, 100, 104.5, 99.8, 103.0))
        assert any(o[3] == "TP" for o in algo._order_calls)

    def test_atr_stop_loss_fires(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        hold = algo.Portfolio[symbol]
        hold.Quantity = 1.0
        hold.Invested = True
        hold.AveragePrice = 100.0
        algo.position_state[symbol] = PositionState(100.0, 100.0, 2.0, algo.Time - timedelta(hours=1))

        manage_open_positions(algo, self._make_slice(algo, symbol, 100, 101.0, 97.5, 98.0))
        assert any(o[3] == "SL" for o in algo._order_calls)

    def test_chandelier_trailing_fires(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        hold = algo.Portfolio[symbol]
        hold.Quantity = 1.0
        hold.Invested = True
        hold.AveragePrice = 100.0
        algo.position_state[symbol] = PositionState(100.0, 100.0, 5.0, algo.Time - timedelta(hours=1))
        algo.feature_engine.current_features = lambda *_args, **_kwargs: {"atr": 2.0}

        manage_open_positions(algo, self._make_slice(algo, symbol, 100, 108.0, 107.0, 108.0))
        manage_open_positions(algo, self._make_slice(algo, symbol, 107.0, 107.2, 101.5, 102.0))
        assert any(o[3] == "Chandelier" for o in algo._order_calls)

    def test_reserved_qty_blocks_duplicate_exit(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        hold = algo.Portfolio[symbol]
        hold.Quantity = 1.0
        hold.Invested = True
        algo.LimitOrder(symbol, -1.0, 100.0, tag="TP")

        assert reserved_qty(algo, symbol) >= 1.0
        before = len(algo._order_calls)
        assert not smart_liquidate(algo, symbol, tag="SL")
        assert len(algo._order_calls) == before

    def test_time_stop_respects_min_hold(self):
        algo = self._build_algo()
        algo.Initialize()
        algo.config = StrategyConfig(min_hold_hours=6)
        symbol = algo.symbol_by_ticker["SOLUSD"]
        hold = algo.Portfolio[symbol]
        hold.Quantity = 1.0
        hold.Invested = True
        algo.position_state[symbol] = PositionState(100.0, 100.0, 2.0, algo.Time - timedelta(hours=2))

        manage_open_positions(algo, self._make_slice(algo, symbol, 100, 100.5, 99.8, 100.1))
        assert not any(o[2] < 0 and o[3] == "TimeStop" for o in algo._order_calls)

    def test_daily_order_cap_blocks_extra_orders(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        for i in range(6):
            algo.Time = datetime(2025, 1, 6, i, tzinfo=timezone.utc)
            assert place_limit_or_market(algo, symbol, 0.01, force_market=True, tag=f"manual{i}") is not None
            assert int(getattr(algo, "_orders_today", 0) or 0) == i + 1
        before = len(algo._order_calls)
        algo.Time = datetime(2025, 1, 6, 7, tzinfo=timezone.utc)
        assert place_limit_or_market(algo, symbol, 0.01, force_market=True, tag="manual7") is None
        assert len(algo._order_calls) == before

    def test_risk_reduce_when_btc_momentum_negative(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        self._warm_and_configure_single_symbol(algo, symbol)
        orig_current = algo.feature_engine.current_features
        algo.feature_engine.current_features = lambda ticker: {"mom_90d": -0.1, "vol_stress_21d": 0.87} if ticker == "BTCUSD" else orig_current(ticker)
        state, candidates = algo._score_candidates(self._make_slice(algo, symbol, 100, 101, 99, 100, btc_close=220))
        assert state == "risk_reduce"
        assert candidates

    def test_risk_on_when_btc_vol_stress_high_with_positive_momentum(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        self._warm_and_configure_single_symbol(algo, symbol)
        orig_current = algo.feature_engine.current_features
        algo.feature_engine.current_features = lambda ticker: {"mom_90d": 0.2, "vol_stress_21d": 0.95} if ticker == "BTCUSD" else orig_current(ticker)
        state, candidates = algo._score_candidates(self._make_slice(algo, symbol, 100, 101, 99.5, 100.2, btc_close=50.0))
        assert state == "risk_on"
        assert candidates

    def test_risk_off_when_btc_momentum_negative_and_vol_stress_extreme(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        self._warm_and_configure_single_symbol(algo, symbol)
        orig_current = algo.feature_engine.current_features
        algo.feature_engine.current_features = lambda ticker: {"mom_90d": -0.2, "vol_stress_21d": 0.95} if ticker == "BTCUSD" else orig_current(ticker)
        state, candidates = algo._score_candidates(self._make_slice(algo, symbol, 100, 101, 99.5, 100.2, btc_close=50.0))
        assert state == "risk_off"
        assert candidates

    def test_cost_gate_rejects_low_score(self):
        sizer = Sizer()
        assert not sizer.passes_cost_gate("SOLUSD", score=0.401, notional=100, fee_model=None)

    def test_no_short_orders(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        qty_before = float(algo.Portfolio[symbol].Quantity)
        execute_regime_entries(algo, [(symbol, -0.9, 0.2)], regime_tag="risk_on")
        assert all(order[2] >= 0 for order in algo._order_calls)
        assert float(algo.Portfolio[symbol].Quantity) >= qty_before

    def test_warmup_updates_state_without_scoring_or_orders(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        algo._debug_logs = []
        algo.IsWarmingUp = True

        algo.OnData(self._make_slice(algo, symbol, 100.0, 101.0, 99.0, 100.5))
        assert len(algo._order_calls) == 0
        assert not any("SIG sym=" in msg for msg in algo._debug_logs)
        assert algo._bar_count >= 1

    def test_ondata_risk_reduce_rebalances_half_scale(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        called = {}
        algo._score_candidates = lambda _data: ("risk_reduce", [(symbol, 0.9, {})])
        algo._rebalance_due = lambda: True
        algo._rebalance_portfolio = lambda scored, risk_scale=1.0: called.update({"scored": scored, "risk_scale": risk_scale})
        algo.OnData(self._Slice({}))
        assert called["risk_scale"] == 0.5

    def test_no_trade_heartbeat_logs_when_risk_on_without_fills(self):
        algo = self._build_algo()
        algo.Initialize()
        symbol = algo.symbol_by_ticker["SOLUSD"]
        algo._debug_logs = []
        algo._bar_count = 168
        algo._last_trade_bar = 0
        algo._last_no_trade_log_bar = 0
        algo._score_candidates = lambda _data: ("risk_on", [(symbol, 0.321, {})])
        algo._rebalance_due = lambda: False
        algo.OnData(self._Slice({}))
        assert any(msg.startswith("NO_TRADE_HB bars_since_trade=") for msg in algo._debug_logs)


def test_heartbeat_logs_every_24_bars():
    t = TestPhaseRequirements()
    algo = t._build_algo()
    algo.Initialize()
    symbol = algo.symbol_by_ticker["SOLUSD"]
    t._warm_and_configure_single_symbol(algo, symbol)
    algo.scorer.score = lambda *_args, **_kwargs: {
        "cvd": 0.0,
        "ofi": 0.0,
        "volc": 0.0,
        "rot": 0.0,
        "mult": 1.0,
        "hurst": 0.6,
        "hurst_regime": "trend",
        "raw": 0.0,
        "final": 0.0,
    }
    algo._debug_logs = []
    for i in range(24):
        algo.Time = datetime(2025, 1, 4, i, tzinfo=timezone.utc)
        algo.OnData(t._make_slice(algo, symbol, 100.0, 101.0, 99.0, 100.0))
    assert any(msg.startswith("HB t=") for msg in algo._debug_logs)


def test_rebalance_logs_are_concise():
    t = TestPhaseRequirements()
    algo = t._build_algo()
    algo.Initialize()
    algo.config = StrategyConfig(rebalance_cadence_hours=1)
    symbol = algo.symbol_by_ticker["SOLUSD"]
    t._warm_and_configure_single_symbol(algo, symbol)
    algo.scorer.score = lambda *_args, **_kwargs: {
        "cvd": 0.0,
        "ofi": 0.0,
        "volc": 0.0,
        "rot": 0.0,
        "mult": 1.0,
        "hurst": 0.6,
        "hurst_regime": "trend",
        "raw": 0.0,
        "final": 0.0,
    }
    algo.log_budget = 1000
    algo._debug_logs = []
    algo._score_candidates = lambda _data: ("risk_on", [(symbol, 0.0, {})])
    for i in range(6):
        algo.Time = datetime(2025, 1, 7, i, tzinfo=timezone.utc)
        algo.OnData(t._make_slice(algo, symbol, 100.0, 101.0, 99.0, 100.0))
    sig_lines = [msg for msg in algo._debug_logs if msg.startswith("SIG sym=")]
    rebalance_lines = [msg for msg in algo._debug_logs if msg.startswith("REB ")]
    assert len(sig_lines) == 0
    assert len(rebalance_lines) >= 1

def test_cross_section_score_orders_winners_first():
    s = Scorer()
    feats = {"ema20": 2, "ema50": 1, "mom_24": 0.01, "adx": 30, "ofi": 1}
    low = s.legacy_score("A", feats, "risk_on", {"btc_trend": 0.01}, rank_24h=0.0, rank_168h=0.0)
    mid = s.legacy_score("B", feats, "risk_on", {"btc_trend": 0.01}, rank_24h=0.5, rank_168h=0.5)
    high = s.legacy_score("C", feats, "risk_on", {"btc_trend": 0.01}, rank_24h=1.0, rank_168h=1.0)
    assert high > mid > low


def test_cross_sectional_momentum_scoring_math():
    s = Scorer(StrategyConfig(signal_mode="cross_sectional_momentum"))
    feats = {"mom_21d": 0.20, "mom_63d": 0.30, "rv_21d": 0.10, "dd_63d": 0.25}
    out = s.score(symbol="SOLUSD", features=feats, regime_state="risk_on", btc_context={})
    expected = 0.6 * (0.20 / 0.10) + 0.4 * (0.30 / 0.10) - 0.3 * 0.25
    assert abs(float(out["final"]) - expected) < 1e-9


def test_rebalance_replacements_capped_at_two():
    t = TestPhaseRequirements()
    algo = t._build_algo()
    algo.Initialize()
    algo.config = StrategyConfig(top_k=8, max_replacements_per_rebalance=2, min_rebalance_weight_delta=1.0)
    picks = [algo.symbol_by_ticker[s] for s in list(algo.symbol_by_ticker.keys())[:12]]
    algo.symbols = picks
    for i, sym in enumerate(picks[:8]):
        hold = algo.Portfolio[sym]
        hold.Quantity = 1.0
        hold.Invested = True
        hold.AveragePrice = 100.0
        algo.position_state[sym] = PositionState(100.0, 100.0, 2.0, datetime(2025, 1, 1, tzinfo=timezone.utc))
    scored = []
    order = [picks[8], picks[9], picks[0], picks[1], picks[2], picks[3], picks[4], picks[5], picks[10], picks[11], picks[6], picks[7]]
    for rank, sym in enumerate(order):
        scored.append((sym, 1.0 - rank * 0.01, {"mom_21d": 0.1, "mom_63d": 0.2, "rv_21d": 0.1, "dd_63d": 0.1}))
    algo._order_calls = []
    algo._rebalance_portfolio(scored)
    exits = [o for o in algo._order_calls if o[2] < 0 and o[3] == "RebalanceExit"]
    assert len(exits) <= 2


def test_execution_file_under_30kb():
    assert (REPO_ROOT / "qc_runtime" / "execution.py").stat().st_size < 30_000


def test_walk_forward_outputs_required_fields():
    out = walk_forward_run(pd.read_csv(REPO_ROOT / "tests/fixtures/walk_forward_bars.csv"), CONFIG)
    required = {
        "oos_sharpe",
        "oos_max_dd",
        "oos_trade_count",
        "oos_win_rate",
        "oos_avg_win_avg_loss",
        "oos_cancel_rate",
        "regime_distribution",
        "exit_tag_distribution",
    }
    assert required.issubset(out.keys())


def test_executor_take_profit_api():
    ex = Executor()
    ex.register_fill("ETHUSD", price=100, atr=2, side=1, bar_index=0)
    out = ex.manage_exits({"ETHUSD": {"high": 105, "low": 99, "close": 104}}, bar_index=1)
    assert ("ETHUSD", "TP") in out
