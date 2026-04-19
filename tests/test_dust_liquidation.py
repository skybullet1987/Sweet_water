from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from execution import liquidate_all_positions, place_limit_or_market


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _SymbolProps:
    LotSize = 0.00001
    MinimumOrderSize = 1.0


class _Security:
    def __init__(self, price: float):
        self.Price = float(price)
        self.BidPrice = float(price)
        self.AskPrice = float(price)
        self.SymbolProperties = _SymbolProps()


class _Holding:
    def __init__(self, qty: float):
        self.Quantity = float(qty)
        self.Price = 1.0


class _Transactions:
    @staticmethod
    def GetOpenOrders(_symbol=None):
        return []


class _Algo:
    def __init__(self, symbol, qty):
        self.config = type("Cfg", (), {"min_qty_fallback": {}, "min_notional_fallback": {}, "max_orders_per_day": 0})()
        self.min_notional = 5.0
        self.Portfolio = {symbol: _Holding(qty)}
        self.Securities = {symbol: _Security(1.0)}
        self.Transactions = _Transactions()
        self._submitted_orders = {}
        self._order_calls = []
        self.log_budget = 100
        self.Debug = lambda _msg: None

    def LimitOrder(self, symbol, quantity, _limit_price, tag=""):
        self._order_calls.append(("limit", symbol, float(quantity), tag))
        return type("T", (), {"OrderId": 1})()

    def MarketOrder(self, symbol, quantity, tag=""):
        self._order_calls.append(("market", symbol, float(quantity), tag))
        return type("T", (), {"OrderId": 2})()


def test_liquidate_all_positions_marks_and_skips_dust():
    symbol = _Symbol("PEPEUSD")
    algo = _Algo(symbol, qty=0.00001)
    closed = liquidate_all_positions(algo, tag="Breaker")
    assert closed == []
    assert symbol in algo._abandoned_dust
    assert algo._order_calls == []

    closed2 = liquidate_all_positions(algo, tag="Breaker")
    assert closed2 == []
    assert algo._order_calls == []


def test_place_limit_or_market_ignores_negative_on_abandoned_dust():
    symbol = _Symbol("PEPEUSD")
    algo = _Algo(symbol, qty=1.0)
    algo._abandoned_dust = {symbol}
    ticket = place_limit_or_market(algo, symbol, -0.5, tag="Breaker")
    assert ticket is None
    assert algo._order_calls == []
