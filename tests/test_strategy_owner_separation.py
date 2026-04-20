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


def test_manage_open_positions_skips_scalper_owned_state(monkeypatch):
    sym = _Symbol("ETHUSD")
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    algo = SimpleNamespace()
    algo.Time = now
    algo.config = SimpleNamespace(strategy_mode="momentum", time_stop_hours=0.0, sl_atr_multiplier=0.1, tp_atr_multiplier=0.1)
    algo.Portfolio = {sym: SimpleNamespace(Quantity=1.0, AveragePrice=100.0)}
    algo.Securities = {sym: SimpleNamespace(Price=50.0)}
    algo.feature_engine = SimpleNamespace(current_features=lambda *_args, **_kwargs: {"atr": 1.0})
    algo.position_state = {
        sym: PositionState(
            entry_price=100.0,
            highest_close=100.0,
            entry_atr=1.0,
            entry_time=now - timedelta(hours=10),
            strategy_owner="scalper",
        )
    }
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

    called = []
    monkeypatch.setattr(execution_module, "smart_liquidate", lambda *_args, **_kwargs: called.append(True) or True)

    exits = manage_open_positions(algo, data=None)
    assert exits == []
    assert called == []
