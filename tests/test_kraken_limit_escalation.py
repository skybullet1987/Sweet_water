from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from execution import escalate_stale_limits, place_limit_or_market  # noqa: E402


class _Symbol:
    def __init__(self, value: str):
        self.Value = value


def test_escalate_skips_market_when_limit_buy_already_filled():
    symbol = _Symbol("ETHUSD")
    hold = 0.04834732
    submitted = []
    t0 = datetime(2024, 1, 18, 12, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)

    pending = {
        symbol: {
            "order_id": 101,
            "submitted": t0,
            "qty": hold,
            "tag": "KM:Scalper",
        }
    }

    algo = type(
        "Algo",
        (),
        {
            "long_only": True,
            "Time": t1,
            "Portfolio": type("P", (), {"__getitem__": lambda _s, _sym: type("H", (), {"Quantity": hold})()})(),
            "Securities": {
                symbol: type(
                    "Sec",
                    (),
                    {
                        "Price": 2480.0,
                        "BidPrice": 2479.0,
                        "AskPrice": 2481.0,
                        "Volume": 10.0,
                        "SymbolProperties": type("SP", (), {"LotSize": 0.00001})(),
                        "GetLastData": lambda _self: type("D", (), {"EndTime": t1})(),
                    },
                )()
            },
            "Transactions": type(
                "T",
                (),
                {
                    "GetOpenOrders": lambda _s, _sym: [],
                    "GetOrderById": lambda _s, oid: type(
                        "O", (), {"Status": "Filled", "Id": oid}
                    )(),
                },
            )(),
            "_pending_limits": pending,
            "Debug": lambda *_a, **_k: None,
        },
    )()
    algo.MarketOrder = lambda sym, qty, tag="": submitted.append((qty, tag))

    escalate_stale_limits(algo)
    assert submitted == []
    assert symbol not in pending


def test_escalate_submits_market_only_when_still_unfilled():
    symbol = _Symbol("ETHUSD")
    submitted = []
    t0 = datetime(2024, 1, 18, 12, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=1)
    qty = 0.05
    pending = {
        symbol: {
            "order_id": 102,
            "submitted": t0,
            "qty": qty,
            "tag": "KM:Scalper",
        }
    }

    algo = type(
        "Algo",
        (),
        {
            "long_only": True,
            "Time": t1,
            "Portfolio": type(
                "P",
                (),
                {
                    "Cash": 100_000.0,
                    "__getitem__": lambda _s, _sym: type("H", (), {"Quantity": 0.0})(),
                },
            )(),
            "Securities": {
                symbol: type(
                    "Sec",
                    (),
                    {
                        "Price": 2480.0,
                        "BidPrice": 2479.0,
                        "AskPrice": 2481.0,
                        "Volume": 10.0,
                        "SymbolProperties": type("SP", (), {"LotSize": 0.00001})(),
                        "GetLastData": lambda _self: type("D", (), {"EndTime": t1})(),
                    },
                )()
            },
            "Transactions": type(
                "T",
                (),
                {
                    "GetOpenOrders": lambda _s, _sym: [],
                    "GetOrderById": lambda _s, oid: type(
                        "O", (), {"Status": "Canceled", "Id": oid}
                    )(),
                },
            )(),
            "_pending_limits": pending,
            "Debug": lambda *_a, **_k: None,
        },
    )()
    algo.MarketOrder = lambda sym, qty, tag="": submitted.append((qty, tag))

    escalate_stale_limits(algo)
    assert len(submitted) == 1
    assert submitted[0][0] == qty
