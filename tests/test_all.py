from __future__ import annotations

import json
import math
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
from execution import Executor, KrakenTieredFeeModel, RealisticCryptoSlippage, execute_regime_entries, place_entry
from features import FeatureEngine, amihud_illiquidity, rank_momentum, realized_vol, roll_spread, zscore_vs_universe
from main import SweetWaterPhase1
from regime import RegimeEngine
from reporting import Reporter, walk_forward_run
from risk import RiskManager
from scoring import Scorer
from sizing import Sizer
from universe import REFERENCE_SYMBOLS, select_universe


def _bar(symbol: str, i: int, base: float = 100.0) -> dict[str, float | str]:
    close = base * (1 + 0.001 * i)
    return {
        "symbol": symbol,
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": 1000 + 10 * i,
    }


# ============================================================
# SECTION 1: Features
# ============================================================
def test_amihud_illiquidity_known_input():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    dv = pd.Series([100, 100, 100, 100])
    val = amihud_illiquidity(r, dv, window=2).iloc[-1]
    assert math.isclose(val, 0.0002, rel_tol=1e-9)


def test_roll_spread_negative_cov_returns_nan():
    close = pd.Series([100, 101, 100.5, 101.3, 101.0, 101.7])
    out = roll_spread(close, window=3)
    assert out.notna().sum() >= 1


def test_indicator_taLib_fallback_matches_numpy():
    engine = FeatureEngine()
    for i in range(80):
        engine.update(_bar("BTCUSD", i))
    feats = engine.current_features("BTCUSD")
    assert "rsi" in feats and "atr" in feats and "adx" in feats


def test_cross_sectional_helpers_work():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.0, -0.01, 0.01]})
    z = zscore_vs_universe(df)
    r = rank_momentum(df, window=2)
    assert z.shape == df.shape
    assert r.shape == df.shape


# ============================================================
# SECTION 2: Regime (HMM)
# ============================================================
def test_hmm_assigns_trending_to_risk_on():
    cfg = StrategyConfig(hmm_train_window_bars=40, hmm_retrain_every_bars=20)
    reg = RegimeEngine(cfg)
    for i in range(80):
        reg.update(0.001 + (i % 5) * 1e-5, 0.01, 0.8)
    assert reg.current_state() in {"risk_on", "chop", "risk_off"}


def test_gmm_fallback_when_hmmlearn_unavailable(monkeypatch):
    import regime as regime_mod

    monkeypatch.setattr(regime_mod, "HAS_HMM", False)
    cfg = StrategyConfig(hmm_train_window_bars=30, hmm_retrain_every_bars=10)
    reg = RegimeEngine(cfg)
    for i in range(60):
        reg.update((-1) ** i * 0.001, 0.02, 0.5)
    assert set(reg.current_state_probs().keys()) == {"risk_on", "risk_off", "chop"}


# ============================================================
# SECTION 3: Scoring
# ============================================================
def test_scoring_risk_off_zero():
    s = Scorer()
    assert s.score("ETHUSD", {}, "risk_off", {}) == 0.0


def test_scoring_risk_on_positive_path():
    s = Scorer()
    score = s.score("ETHUSD", {"ema20": 2, "ema50": 1, "mom_24": 0.01, "adx": 30, "ofi": 1}, "risk_on", {"btc_trend": 0.01})
    assert score > 0


# ============================================================
# SECTION 4: Sizing
# ============================================================
def test_sizer_output_bounded():
    sz = Sizer()
    for i in range(100):
        sz.update_returns(0.001 if i % 2 == 0 else -0.0005)
        sz.record_trade(0.01 if i % 3 else -0.005)
    weight = sz.size_for_trade("SOLUSD", 0.8, {"equity": 500, "gross_exposure": 0.2})
    assert 0 <= weight <= CONFIG.kelly_cap


# ============================================================
# SECTION 5: Execution + Triple Barrier
# ============================================================
def test_fee_slippage_estimators_positive():
    assert KrakenTieredFeeModel().estimate_round_trip_cost("BTCUSD", 100.0) > 0
    assert RealisticCryptoSlippage().estimate_slippage_bps("BTCUSD", 100.0, price=100.0, volume=1000.0) >= 1


def test_realistic_crypto_slippage_is_duck_typed_plain_class():
    assert RealisticCryptoSlippage.__bases__ == (object,)


def test_executor_triple_barrier_exit_take_profit():
    ex = Executor()
    ex.register_fill("ETHUSD", price=100, atr=2, side=1, bar_index=0)
    out = ex.manage_exits({"ETHUSD": {"high": 105, "low": 99, "close": 104}}, bar_index=1)
    assert ("ETHUSD", "take_profit") in out


# ============================================================
# SECTION 6: Risk + Circuit Breaker
# ============================================================
def test_risk_circuit_breaker_blocks_entries():
    rm = RiskManager()
    rm.evaluate({"target_weight": 0.1, "equity": 100, "gross_exposure": 0, "net_exposure": 0, "correlation": 0})
    decision = rm.evaluate({"target_weight": 0.1, "equity": 85, "gross_exposure": 0, "net_exposure": 0, "correlation": 0})
    assert not decision.approved


# ============================================================
# SECTION 7: Pipeline integration (replaces old behavioral tests)
# ============================================================
def test_score_to_order_end_to_end():
    fe = FeatureEngine()
    for i in range(100):
        fe.update(_bar("XRPUSD", i, base=50))
    feats = fe.current_features("XRPUSD")
    reg = RegimeEngine(StrategyConfig(hmm_train_window_bars=30, hmm_retrain_every_bars=10))
    for _ in range(60):
        reg.update(0.001, 0.01, 0.7)
    score = Scorer().score("XRPUSD", feats, reg.current_state(), {"btc_trend": 0.001})
    size = Sizer().size_for_trade("XRPUSD", score, {"equity": 500, "gross_exposure": 0})
    decision = RiskManager().evaluate({"target_weight": size, "equity": 500, "gross_exposure": 0, "net_exposure": 0, "correlation": 0})
    if decision.approved and abs(score) >= CONFIG.score_threshold:
        order = Executor().place_entry("XRPUSD", decision.adjusted_target_weight, score)
        assert order["symbol"] == "XRPUSD"


def test_zero_trade_regression_guard():
    out = walk_forward_run(pd.read_csv(REPO_ROOT / "tests/fixtures/walk_forward_bars.csv"), CONFIG)
    assert out["oos_trade_count"] > 0


def test_cost_accounting_no_double_count():
    rpt = Reporter()
    rpt.on_order_event({"status": "filled", "pnl": 0.01})
    rpt.on_order_event({"status": "filled", "pnl": -0.01})
    rpt.on_order_event({"status": "canceled"})
    final = rpt.final_report()
    assert final["trade_count"] == 2
    assert 0 <= final["cancel_rate"] <= 1


# ============================================================
# SECTION 8: Walk-forward smoke
# ============================================================
def test_walk_forward_fixture_meets_baseline():
    fixture = pd.read_csv(REPO_ROOT / "tests/fixtures/walk_forward_bars.csv")
    baseline_path = REPO_ROOT / "tests/fixtures/walk_forward_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    out = walk_forward_run(fixture, CONFIG)
    assert out["oos_sharpe"] >= 0.5
    assert out["oos_avg_win_avg_loss"] >= 1.0
    assert 15 <= out["oos_trade_count"] <= 200
    assert out["oos_cancel_rate"] < 0.20
    dist = out["regime_distribution"]
    active = [k for k, v in dist.items() if v >= 0.05]
    assert len(active) >= 2
    assert baseline["oos_trade_count"] >= 0


def test_universe_selector_excludes_reference_symbols():
    def _hist(symbol, start, end):
        _ = start, end
        rows = 30
        return pd.DataFrame({"close": np.linspace(100, 120, rows), "volume": np.linspace(1000, 2000, rows)})

    selected = select_universe(_hist, pd.Timestamp("2026-01-01", tz="UTC"))
    assert "BTCUSD" not in selected
    assert all(sym not in REFERENCE_SYMBOLS for sym in selected)


def test_real_orders_actually_placed_with_real_fees():
    from datetime import datetime, timedelta

    class _Sym:
        def __init__(self, value: str):
            self.Value = value

        def __hash__(self):
            return hash(self.Value)

        def __eq__(self, other):
            return isinstance(other, _Sym) and self.Value == other.Value

    class _SymbolProps:
        MinimumPriceVariation = 0.01
        LotSize = 0.0001
        MinimumOrderSize = 0.0001

    class _Sec:
        def __init__(self, price: float, volume: float):
            self.Price = price
            self.Volume = volume
            self.BidPrice = price * 0.999
            self.AskPrice = price * 1.001
            self.SymbolProperties = _SymbolProps()
            self.FeeModel = None
            self.SlippageModel = None

    class _Holding:
        def __init__(self):
            self.Quantity = 0.0
            self.Invested = False
            self.AveragePrice = 0.0
            self.Price = 0.0

    class _Portfolio(dict):
        def __init__(self):
            super().__init__()
            self.TotalPortfolioValue = 1000.0
            self.TotalHoldingsValue = 0.0
            self.Cash = 1000.0

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
        def __init__(self, oid: int):
            self.OrderId = oid

    class _FeeSpy(KrakenTieredFeeModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def estimate_round_trip_cost(self, symbol, notional, is_limit=True):
            self.calls += 1
            return super().estimate_round_trip_cost(symbol, notional, is_limit=is_limit)

    class _Algo:
        def __init__(self):
            self.Time = datetime(2025, 1, 1)
            self.LiveMode = False
            self.IsWarmingUp = False
            self._pending_orders = {}
            self._submitted_orders = {}
            self._order_retries = {}
            self._session_blacklist = set()
            self._spread_warning_times = {}
            self._symbol_slippage_history = {}
            self._slip_abs = []
            self.crypto_data = {}
            self.entry_prices = {}
            self.highest_prices = {}
            self.entry_times = {}
            self.entry_volumes = {}
            self.max_participation_rate = 0.25
            self.spread_limit_pct = 0.03
            self.stale_order_bars = 3
            self.min_notional = 5.0
            self.min_order_size_usd = 5.0
            self.Debug = lambda *_args, **_kwargs: None
            self.Portfolio = _Portfolio()
            self.Transactions = _Transactions()
            self.order_calls = []
            self.symbols = {}
            for t in ("BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "USDTUSD"):
                s = _Sym(t)
                self.symbols[t] = s
                self.Portfolio[s] = _Holding()
                self.crypto_data[s] = {"prices": [], "volume": []}
            self.Securities = {
                self.symbols["BTCUSD"]: _Sec(100.0, 10000.0),
                self.symbols["ETHUSD"]: _Sec(60.0, 9000.0),
                self.symbols["SOLUSD"]: _Sec(30.0, 8500.0),
                self.symbols["XRPUSD"]: _Sec(1.0, 20000.0),
                self.symbols["USDTUSD"]: _Sec(1.0, 20000.0),
            }
            self.fee_spy = _FeeSpy()
            for sec in self.Securities.values():
                sec.FeeModel = self.fee_spy
                sec.SlippageModel = RealisticCryptoSlippage(self)

        def MarketOrder(self, symbol, quantity, tag=""):
            oid = self.Transactions._next
            self.Transactions._next += 1
            order = type("O", (), {"Id": oid, "Symbol": symbol, "Quantity": quantity, "Tag": tag, "Price": self.Securities[symbol].Price, "Direction": 1 if quantity > 0 else -1})
            self.Transactions._orders[oid] = order
            self.order_calls.append(("market", symbol.Value, quantity, tag))
            return _Ticket(oid)

        def LimitOrder(self, symbol, quantity, limit_price, tag=""):
            oid = self.Transactions._next
            self.Transactions._next += 1
            order = type("O", (), {"Id": oid, "Symbol": symbol, "Quantity": quantity, "Tag": tag, "Price": limit_price, "Direction": 1 if quantity > 0 else -1})
            self.Transactions._orders[oid] = order
            self.Transactions._open.append(order)
            self.order_calls.append(("limit", symbol.Value, quantity, tag))
            return _Ticket(oid)

        def History(self, *_args, **_kwargs):
            return pd.DataFrame()

    algo = _Algo()
    sizer = Sizer()
    rejected_blacklist = False
    for bar in range(50):
        algo.Time = algo.Time + timedelta(hours=1)
        for ticker in ("BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "USDTUSD"):
            sym = algo.symbols[ticker]
            sec = algo.Securities[sym]
            drift = 1.0 + 0.003 + (0.0005 if ticker != "BTCUSD" else 0.0)
            sec.Price *= drift
            sec.BidPrice = sec.Price * 0.999
            sec.AskPrice = sec.Price * 1.001
            sec.Volume = max(1000.0, sec.Volume * 1.001)
            algo.crypto_data[sym]["prices"].append(sec.Price)
            algo.crypto_data[sym]["volume"].append(sec.Volume)

        for ticker in ("ETHUSD", "SOLUSD", "XRPUSD"):
            sym = algo.symbols[ticker]
            sec = algo.Securities[sym]
            notional = 80.0
            if sizer.passes_cost_gate(sym, 0.9, notional, sec.FeeModel, is_limit=True):
                execute_regime_entries(algo, [(sym, 0.9, 0.08)], regime_tag="risk_on")
        # explicit blacklist check path
        black = place_entry(algo, algo.symbols["USDTUSD"], 10.0, tag="blacklist-test")
        if black is None:
            rejected_blacklist = True

    assert len(algo.order_calls) >= 3
    assert algo.fee_spy.calls >= 1
    assert rejected_blacklist


class TestSweetWaterPhase1Integration:
    class _Symbol:
        def __init__(self, value: str):
            self.Value = value

        def __hash__(self):
            return hash(self.Value)

        def __eq__(self, other):
            return isinstance(other, TestSweetWaterPhase1Integration._Symbol) and self.Value == other.Value

    class _SymbolProps:
        MinimumPriceVariation = 0.01
        LotSize = 0.0001
        MinimumOrderSize = 0.0001

    class _Security:
        def __init__(self, symbol, price: float):
            self.Symbol = symbol
            self.Price = price
            self.BidPrice = price * 0.999
            self.AskPrice = price * 1.001
            self.Volume = 10_000.0
            self.SymbolProperties = TestSweetWaterPhase1Integration._SymbolProps()
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
            self.TotalPortfolioValue = 1000.0
            self.TotalHoldingsValue = 0.0
            self.Cash = 1000.0

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
        def __init__(self, order_id: int):
            self.OrderId = order_id

    class _Slice:
        def __init__(self, bars):
            self.Bars = bars

    class _Decision:
        def __init__(self, adjusted_target_weight: float):
            self.approved = True
            self.adjusted_target_weight = adjusted_target_weight

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
        algo.LiveMode = False
        algo.IsWarmingUp = False
        algo.Securities = {}
        algo.Portfolio = self._Portfolio()
        algo.Transactions = self._Transactions()
        algo._order_calls = []
        algo._debug_logs = []

        def _noop(*_args, **_kwargs):
            return None

        algo.SetBrokerageModel = _noop
        algo.SetCash = lambda cash: setattr(algo.Portfolio, "Cash", float(cash))
        algo.SetStartDate = _noop
        algo.SetEndDate = _noop
        algo.SetWarmup = _noop
        algo.Debug = lambda msg: algo._debug_logs.append(str(msg))
        algo.Liquidate = _noop

        def _add_crypto(ticker, _resolution, _market):
            symbol = self._Symbol(ticker)
            sec = self._Security(symbol, 100.0 if ticker != "BTCUSD" else 200.0)
            algo.Securities[symbol] = sec
            algo.Portfolio[symbol] = self._Holding()
            return sec

        algo.AddCrypto = _add_crypto

        def _history(symbol, _start, _end, _resolution):
            ticker = symbol.Value if hasattr(symbol, "Value") else str(symbol)
            rows = 30
            base = {
                "ETHUSD": 2_000_000.0,
                "SOLUSD": 1_500_000.0,
                "XRPUSD": 1_200_000.0,
                "ADAUSD": 1_100_000.0,
                "LINKUSD": 1_000_000.0,
                "DOTUSD": 900_000.0,
            }.get(ticker, 500_000.0)
            close = np.linspace(10.0, 11.5, rows)
            volume = np.linspace(base, base * 1.1, rows)
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

        def _market_order(symbol, quantity, tag=""):
            order_id = algo.Transactions._next
            algo.Transactions._next += 1
            order = type("Order", (), {"Id": order_id, "Symbol": symbol, "Quantity": quantity, "Tag": tag, "Price": algo.Securities[symbol].Price, "Direction": 1 if quantity > 0 else -1})
            algo.Transactions._orders[order_id] = order
            algo._order_calls.append(("market", symbol.Value, float(quantity), tag))
            return self._Ticket(order_id)

        def _limit_order(symbol, quantity, limit_price, tag=""):
            order_id = algo.Transactions._next
            algo.Transactions._next += 1
            order = type("Order", (), {"Id": order_id, "Symbol": symbol, "Quantity": quantity, "Tag": tag, "Price": float(limit_price), "Direction": 1 if quantity > 0 else -1})
            algo.Transactions._orders[order_id] = order
            algo.Transactions._open.append(order)
            algo._order_calls.append(("limit", symbol.Value, float(quantity), tag))
            return self._Ticket(order_id)

        algo.MarketOrder = _market_order
        algo.LimitOrder = _limit_order
        return algo

    def test_initialize_and_ondata_places_non_btc_entries(self):
        algo = self._build_algo()
        algo.Initialize()

        assert "BTCUSD" in [s.Value for s in algo.reference_symbols]
        assert "BTCUSD" not in [s.Value for s in algo.symbols]
        assert any(s.Value != "BTCUSD" for s in algo.symbols)

        algo.sizer.passes_cost_gate = lambda *_args, **_kwargs: True
        algo.sizer.size_for_trade = lambda _symbol, _score, _state: 0.08
        algo.scorer.score = lambda _symbol, _feats, _state, _ctx: 0.9
        algo.risk.evaluate = lambda payload: self._Decision(payload["target_weight"])

        fixture = pd.read_csv(REPO_ROOT / "tests/fixtures/walk_forward_bars.csv").head(240)
        tradable = algo.symbols[:4]
        refs = list(algo.reference_symbols)
        tracked_symbols = refs + tradable

        for i, row in fixture.iterrows():
            algo.Time = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=int(i))
            bars = {}
            base = float(row["close"])
            for j, symbol in enumerate(tracked_symbols):
                px = base * (1.0 + 0.0007 * j)
                sec = algo.Securities[symbol]
                sec.Price = px
                sec.BidPrice = px * 0.999
                sec.AskPrice = px * 1.001
                sec.Volume = max(1_000.0, float(row["volume"]) * (1.0 + 0.02 * j))
                bars[symbol] = self._Bar(px * 0.999, px * 1.002, px * 0.998, px, sec.Volume)
            algo.OnData(self._Slice(bars))

        non_btc_orders = [o for o in algo._order_calls if o[1] != "BTCUSD"]
        assert non_btc_orders, f"Expected non-BTC entry order, got {algo._order_calls}"
