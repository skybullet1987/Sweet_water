from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from execution import place_limit_or_market


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _Holding:
    def __init__(self):
        self.Quantity = 0.0


class _Algo:
    def __init__(self, symbol):
        self.Time = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
        self.config = type("Cfg", (), {"min_qty_fallback": {}, "min_notional_fallback": {}, "max_orders_per_day": 0})()
        self.Portfolio = {symbol: _Holding()}
        self._failed_escalations = {symbol: self.Time}
        self._submitted_orders = {}
        self._order_calls = []
        self._log_once_hour = {}
        self.log_budget = 100
        self.Debug = lambda *_args, **_kwargs: None

    def MarketOrder(self, symbol, quantity, tag=""):
        self._order_calls.append((symbol, float(quantity), tag))
        return type("Ticket", (), {"OrderId": 1})()


def test_failed_escalation_cooldown_blocks_for_24h_then_allows():
    sym = _Symbol("BTCUSD")
    algo = _Algo(sym)

    blocked = place_limit_or_market(algo, sym, 0.1, force_market=True, tag="Entry")
    assert blocked is None
    assert len(algo._order_calls) == 0

    algo.Time = algo.Time + timedelta(hours=25)
    allowed = place_limit_or_market(algo, sym, 0.1, force_market=True, tag="Entry")
    assert allowed is not None
    assert len(algo._order_calls) == 1
