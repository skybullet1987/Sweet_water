from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import execution as execution_module
from execution import PositionState, manage_open_positions


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


def _build_algo(*, symbol, price: float, state: PositionState):
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    algo = SimpleNamespace()
    algo.Time = now
    algo.config = SimpleNamespace(time_stop_hours=120.0, sl_atr_multiplier=1.5, tp_atr_multiplier=3.0, scalper_partial_tp_atr_mult=1.0)
    algo.Portfolio = {symbol: SimpleNamespace(Quantity=1.0, AveragePrice=state.entry_price)}
    algo.Securities = {symbol: SimpleNamespace(Price=price)}
    algo.feature_engine = SimpleNamespace(current_features=lambda *_args, **_kwargs: {"atr": state.entry_atr})
    algo.position_state = {symbol: state}
    algo.Transactions = SimpleNamespace(GetOpenOrders=lambda *_args, **_kwargs: [])
    algo._submitted_orders = {}
    algo._order_retries = {}
    algo._cancel_cooldowns = {}
    algo._failed_escalations = {}
    algo._failed_exit_counts = {}
    algo._abandoned_dust = set()
    algo.Debug = lambda *_args, **_kwargs: None
    algo.pnl_by_tag = {}
    algo.pnl_by_regime = {}
    return algo


def test_position_state_accepts_precomputed_price_kwargs():
    state = PositionState(
        entry_price=100.0,
        highest_close=100.0,
        entry_atr=2.0,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        stop_price=97.0,
        take_profit_price=104.0,
        partial_tp_price=102.0,
        trail_anchor_price=100.0,
    )
    assert state.stop_price == 97.0
    assert state.take_profit_price == 104.0
    assert state.partial_tp_price == 102.0
    assert state.trail_anchor_price == 100.0


def test_manage_open_positions_uses_precomputed_stop_price(monkeypatch):
    sym = _Symbol("SOLUSD")
    state = PositionState(
        entry_price=100.0,
        highest_close=100.0,
        entry_atr=10.0,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc) - timedelta(hours=1),
        stop_price=97.0,
        take_profit_price=110.0,
        partial_tp_price=102.0,
        trail_anchor_price=100.0,
    )
    algo = _build_algo(symbol=sym, price=96.5, state=state)
    calls: list[str] = []
    monkeypatch.setattr(execution_module, "smart_liquidate", lambda _algo, _symbol, tag="": calls.append(tag) or True)

    exits = manage_open_positions(algo, data=None)

    assert exits == [(sym, "SL")]
    assert calls == ["SL"]


def test_manage_open_positions_uses_precomputed_partial_tp_price(monkeypatch):
    sym = _Symbol("ETHUSD")
    state = PositionState(
        entry_price=100.0,
        highest_close=100.0,
        entry_atr=10.0,
        entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc) - timedelta(hours=1),
        stop_price=97.0,
        take_profit_price=110.0,
        partial_tp_price=102.0,
        trail_anchor_price=100.0,
    )
    algo = _build_algo(symbol=sym, price=102.5, state=state)
    calls: list[str] = []
    monkeypatch.setattr(execution_module, "smart_liquidate", lambda _algo, _symbol, tag="": calls.append(tag) or True)

    exits = manage_open_positions(algo, data=None)

    assert exits == [(sym, "TP1")]
    assert calls == ["TP1"]
