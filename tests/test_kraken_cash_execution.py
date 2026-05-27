from __future__ import annotations

import sys
from pathlib import Path

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from execution import (  # noqa: E402
    available_sell_qty,
    cancel_open_orders,
    liquidate_symbol,
    place_limit_or_market,
    reserved_sell_qty,
)


class _Symbol:
    def __init__(self, value: str):
        self.Value = value


class _Order:
    def __init__(self, qty: float, *, order_id: int = 1, tag: str = ""):
        self.Quantity = float(qty)
        self.Id = order_id
        self.OrderId = order_id
        self.Tag = tag


def _algo(*, hold: float = 0.10418, open_orders=None):
    symbol = _Symbol("ETHUSD")
    open_orders = list(open_orders or [])
    submitted = []

    sec = type(
        "Sec",
        (),
        {
            "Price": 2500.0,
            "BidPrice": 2499.0,
            "AskPrice": 2501.0,
            "Volume": 100.0,
            "SymbolProperties": type("SP", (), {"LotSize": 0.00001, "MinimumOrderSize": 0.00001})(),
            "GetLastData": lambda _self: type("D", (), {"EndTime": None})(),
        },
    )()
    portfolio = {symbol: type("H", (), {"Quantity": hold})()}
    algo = type(
        "Algo",
        (),
        {
            "long_only": True,
            "Time": None,
            "Portfolio": type(
                "P",
                (),
                {
                    "__getitem__": lambda _self, s: portfolio[s],
                },
            )(),
            "Securities": {symbol: sec},
            "Transactions": type(
                "T",
                (),
                {
                    "GetOpenOrders": lambda _self, _sym: list(open_orders),
                    "CancelOrder": lambda _self, oid: open_orders.clear(),
                },
            )(),
            "LimitOrder": lambda *_a, **_k: None,
            "Debug": lambda *_a, **_k: None,
            "_pending_limits": {},
            "_submitted": submitted,
        },
    )()
    algo.MarketOrder = lambda sym, qty, tag="": submitted.append((float(qty), tag))
    return algo, symbol, submitted, open_orders


def test_reserved_sell_qty_reduces_available():
    algo, symbol, _, _ = _algo(hold=1.0, open_orders=[_Order(-0.4, tag="SL")])
    assert reserved_sell_qty(algo, symbol) == 0.4
    assert available_sell_qty(algo, symbol) == 0.6


def test_liquidate_cancels_open_orders_before_sell():
    algo, symbol, submitted, open_orders = _algo(
        hold=0.10418,
        open_orders=[_Order(-0.10418, order_id=9, tag="SL")],
    )
    liquidate_symbol(algo, symbol, force_market=True)
    assert len(open_orders) == 0
    assert len(submitted) == 1
    assert submitted[0][0] < 0
    assert abs(submitted[0][0]) <= 0.10418 + 1e-9


def test_place_limit_or_market_will_not_oversell_with_reserved():
    algo, symbol, submitted, _ = _algo(
        hold=0.10418,
        open_orders=[_Order(-0.10, tag="SL")],
    )
    ok = place_limit_or_market(algo, symbol, -0.10418, tag="Exit", force_market=True)
    assert ok is True
    assert len(submitted) == 1
    assert abs(submitted[0][0]) <= 0.00418 + 1e-9
