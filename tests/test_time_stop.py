from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


class _Holding:
    def __init__(self, qty=0.0):
        self.Quantity = float(qty)
        self.AveragePrice = 100.0


def test_time_stop_fires_after_121_hours(monkeypatch):
    sym = _Symbol("SOLUSD")
    algo = type("Algo", (), {})()
    algo.Time = datetime(2025, 1, 6, 1, tzinfo=timezone.utc)
    algo.Portfolio = {sym: _Holding(qty=1.0)}
    algo.Securities = {sym: type("Sec", (), {"Price": 100.0})()}
    algo.feature_engine = type("Feat", (), {"current_features": staticmethod(lambda *_args, **_kwargs: {"atr": 2.0})})()
    algo.position_state = {sym: PositionState(100.0, 100.0, 2.0, algo.Time - timedelta(hours=121))}
    algo.Transactions = type("Tx", (), {"GetOpenOrders": staticmethod(lambda *_args, **_kwargs: [])})()
    algo.Debug = lambda *_args, **_kwargs: None
    algo.pnl_by_tag = {}
    algo.pnl_by_regime = {}
    calls = []
    monkeypatch.setattr(execution_module, "smart_liquidate", lambda _algo, symbol, tag="": calls.append((symbol, tag)) or True)

    exits = manage_open_positions(algo, data=None)
    assert exits == [(sym, "TimeStop")]
    assert calls and calls[0][1] == "TimeStop"
