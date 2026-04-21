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


class _Portfolio(dict):
    def __init__(self, symbol):
        super().__init__({symbol: _Holding()})
        self.Cash = 1_000.0
        self.CashBook = {"USD": type("Cash", (), {"Amount": 1_000.0})()}


class _Algo:
    def __init__(self, symbol):
        self.Time = datetime(2025, 1, 1, 0, tzinfo=timezone.utc)
        self.config = type(
            "Cfg",
            (),
            {
                "min_qty_fallback": {},
                "min_notional_fallback": {},
                "max_orders_per_day": 0,
                "failed_esc_cooldown_hours": 6.0,
                "stale_price_minutes": 90,
            },
        )()
        self.Portfolio = _Portfolio(symbol)
        self._failed_escalations = {symbol: self.Time}
        self._submitted_orders = {}
        self._order_calls = []
        self._log_once_hour = {}
        self.log_budget = 100
        self.Debug = lambda *_args, **_kwargs: None
        self.Securities = {symbol: type("Sec", (), {"Price": 100.0, "GetLastData": lambda *_args, **_kwargs: None})()}

    def MarketOrder(self, symbol, quantity, tag=""):
        self._order_calls.append((symbol, float(quantity), tag))
        return type("Ticket", (), {"OrderId": 1})()


def test_failed_escalation_cooldown_blocks_for_6h_then_allows():
    sym = _Symbol("BTCUSD")
    algo = _Algo(sym)

    blocked = place_limit_or_market(algo, sym, 0.1, force_market=True, tag="Entry")
    assert blocked is None
    assert len(algo._order_calls) == 0

    algo.Time = algo.Time + timedelta(hours=7)
    allowed = place_limit_or_market(algo, sym, 0.1, force_market=True, tag="Entry")
    assert allowed is not None
    assert len(algo._order_calls) == 1


def test_failed_escalation_prunes_entries_older_than_24h():
    sym = _Symbol("BTCUSD")
    old = _Symbol("ETHUSD")
    algo = _Algo(sym)
    algo._failed_escalations[old] = algo.Time - timedelta(hours=25)

    _ = place_limit_or_market(algo, sym, 0.1, force_market=True, tag="Entry")

    assert old not in algo._failed_escalations
    assert sym in algo._failed_escalations


def test_force_market_price_zero_returns_none_without_submit():
    sym = _Symbol("BTCUSD")
    algo = _Algo(sym)
    algo._failed_escalations = {}
    algo.Securities[sym] = type("Sec", (), {"Price": 0.0})()

    out = place_limit_or_market(algo, sym, 0.1, force_market=True, tag="[StaleEsc]")

    assert out is None
    assert len(algo._order_calls) == 0
