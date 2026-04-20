from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from execution import can_afford, is_price_stale, mark_rebalance_failure, rebalance_symbol_blocked


class _Order:
    def __init__(self, qty: float, limit_price: float):
        self.Quantity = float(qty)
        self.LimitPrice = float(limit_price)


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


def _algo(cash=100.0, open_orders=None):
    symbol = _Symbol("ETHUSD")
    sec = type("Sec", (), {"Price": 100.0, "GetLastData": lambda _self: type("D", (), {"EndTime": datetime(2025, 1, 1, 10, tzinfo=timezone.utc)})()})()
    algo = type(
        "Algo",
        (),
        {
            "Time": datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
            "config": StrategyConfig(cash_safety_factor=0.97, rebalance_invalid_retry_cap=3),
            "Portfolio": type("P", (), {"CashBook": {"USD": type("C", (), {"Amount": cash})()}, "Cash": cash})(),
            "Transactions": type("T", (), {"GetOpenOrders": lambda _self, *_args, **_kwargs: list(open_orders or [])})(),
            "Securities": {symbol: sec},
            "Debug": lambda *_args, **_kwargs: None,
            "log_budget": 0,
        },
    )()
    return algo, symbol


def test_can_afford_accounts_for_reserved_open_buy_orders():
    algo, symbol = _algo(cash=100.0, open_orders=[_Order(0.5, 100.0)])  # reserves $50
    ok, required, available = can_afford(algo, symbol, qty=0.6, price=100.0)
    assert ok is False
    assert required == 60.0
    assert available < 50.0


def test_is_price_stale_rejects_invalid_and_old_prices():
    algo, symbol = _algo()
    assert is_price_stale(algo, symbol, float("nan")) is True
    stale_algo, stale_symbol = _algo()
    stale_algo.Time = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    stale_algo.Securities[stale_symbol] = type(
        "Sec",
        (),
        {
            "Price": 100.0,
            "GetLastData": lambda _self: type(
                "D", (), {"EndTime": stale_algo.Time - timedelta(minutes=240)}
            )(),
        },
    )()
    assert is_price_stale(stale_algo, stale_symbol, 100.0, max_age_minutes=30) is True


def test_rebalance_failure_cap_blocks_for_current_day():
    algo, symbol = _algo()
    for _ in range(3):
        mark_rebalance_failure(algo, symbol, "insuff_funds")
    assert rebalance_symbol_blocked(algo, symbol) is True
    algo.Time = algo.Time + timedelta(days=1)
    assert rebalance_symbol_blocked(algo, symbol) is False
