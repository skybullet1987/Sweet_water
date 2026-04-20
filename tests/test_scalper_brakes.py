from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import main as main_module
import scalper as scalper_module
from execution import PositionState
from main import SweetWaterPhase1


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _FeatureEngine:
    def __init__(self, mapping):
        self.mapping = mapping

    def current_features(self, key):
        return dict(self.mapping.get(key, {}))


def _base_algo():
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    algo.Time = now
    algo.Debug = lambda *_args, **_kwargs: None
    algo._ingest_data = lambda _data: None
    algo._scalper_last_trade_time = {}
    algo._scalper_consec_losses = {}
    algo._scalper_daily_pnl = 0.0
    algo._scalper_daily_anchor_equity = 500.0
    algo._scalper_daily_anchor_date = now.date()
    algo._scalper_session_brake_until = None
    algo._scalper_recent_pnls = main_module.deque(maxlen=6)
    algo.symbol_by_ticker = {"BTCUSD": _Symbol("BTCUSD")}
    algo.config = SimpleNamespace(
        scalper_daily_loss_brake=-0.03,
        scalper_max_concurrent=4,
        scalper_universe_size=20,
        scalper_position_size_pct=0.15,
        min_position_floor_usd=5.0,
    )
    algo.Portfolio = SimpleNamespace(
        TotalPortfolioValue=500.0,
        Cash=500.0,
        CashBook={"USD": SimpleNamespace(Amount=500.0)},
    )
    algo.Securities = {}
    algo.symbols = []
    algo.position_state = {}
    return algo


def test_scalper_three_consecutive_losses_triggers_session_brake(monkeypatch):
    algo = _base_algo()
    sym = _Symbol("ETHUSD")
    algo.symbols = [sym]
    algo._current_holdings = lambda: [sym]
    algo.Portfolio = type(
        "_P",
        (),
        {
            "TotalPortfolioValue": 500.0,
            "Cash": 500.0,
            "CashBook": {"USD": SimpleNamespace(Amount=500.0)},
            "__getitem__": lambda _self, _symbol: SimpleNamespace(AveragePrice=100.0),
        },
    )()
    algo.Securities = {sym: SimpleNamespace(Price=99.0)}
    algo.position_state[sym] = PositionState(entry_price=100.0, highest_close=100.0, entry_atr=1.0, entry_time=algo.Time - timedelta(hours=1))
    algo.feature_engine = _FeatureEngine({"BTCUSD": {"ret_1h": 0.0, "ret_6h": 0.0}, "ETHUSD": {"z_20h": -2.5, "rsi_14": 30.0, "rv_21d": 0.3}})
    monkeypatch.setattr(main_module, "smart_liquidate", lambda _algo, _sym, tag="": True)
    monkeypatch.setattr(scalper_module, "evaluate_exit", lambda **kwargs: (True, "SL"))
    monkeypatch.setattr(scalper_module, "evaluate_entry", lambda **kwargs: (False, "skip"))

    algo._scalper_on_data(data=SimpleNamespace())
    algo._scalper_on_data(data=SimpleNamespace())
    algo._scalper_on_data(data=SimpleNamespace())

    assert algo._scalper_session_brake_until == algo.Time + timedelta(hours=6)


def test_scalper_entry_blocked_during_session_brake(monkeypatch):
    algo = _base_algo()
    sym = _Symbol("ETHUSD")
    algo.symbols = [sym]
    algo._current_holdings = lambda: []
    algo._scalper_session_brake_until = algo.Time + timedelta(hours=3)
    algo.feature_engine = _FeatureEngine({"BTCUSD": {"ret_1h": 0.0, "ret_6h": 0.0}, "ETHUSD": {"z_20h": -2.5, "rsi_14": 30.0, "rv_21d": 0.3}})
    algo.Securities = {sym: SimpleNamespace(Price=100.0)}
    submit_calls = []
    monkeypatch.setattr(scalper_module, "evaluate_exit", lambda **kwargs: (False, ""))
    monkeypatch.setattr(scalper_module, "evaluate_entry", lambda **kwargs: (True, "OK"))
    monkeypatch.setattr(main_module, "round_quantity", lambda _algo, _symbol, qty: qty)
    monkeypatch.setattr(main_module, "place_entry", lambda _algo, _symbol, _qty, tag="", signal_score=0.0: submit_calls.append(tag) or object())

    algo._scalper_on_data(data=SimpleNamespace())
    assert submit_calls == []
