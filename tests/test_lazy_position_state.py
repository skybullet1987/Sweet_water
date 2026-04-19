from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from execution import manage_open_positions


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _Holding:
    def __init__(self, qty=0.0, avg=0.0):
        self.Quantity = float(qty)
        self.AveragePrice = float(avg)


def test_manage_open_positions_lazy_builds_state_with_safe_atr_default():
    sym = _Symbol("SOLUSD")
    algo = type("Algo", (), {})()
    algo.Time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    algo.Portfolio = {sym: _Holding(qty=1.0, avg=100.0)}
    algo.Securities = {sym: type("Sec", (), {"Price": 100.0})()}
    algo.feature_engine = type("Feat", (), {"current_features": staticmethod(lambda *_args, **_kwargs: {})})()
    algo.position_state = {}
    algo.Transactions = type("Tx", (), {"GetOpenOrders": staticmethod(lambda *_args, **_kwargs: [])})()
    algo._submitted_orders = {}
    algo._order_retries = {}
    algo._cancel_cooldowns = {}
    algo._failed_escalations = {}
    algo._failed_exit_counts = {}
    algo._abandoned_dust = set()
    algo.Debug = lambda *_args, **_kwargs: None
    algo.pnl_by_tag = {}
    algo.pnl_by_regime = {}

    exits = manage_open_positions(algo, data=None)
    assert exits == []
    assert sym in algo.position_state
    state = algo.position_state[sym]
    assert abs(state.entry_price - 100.0) < 1e-9
    assert abs(state.entry_atr - 5.0) < 1e-9
