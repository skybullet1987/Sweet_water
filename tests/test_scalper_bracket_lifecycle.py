from __future__ import annotations

import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import main as main_module
import scalper as scalper_module
from main import SweetWaterPhase1


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _Portfolio:
    def __init__(self, symbol, quantity: float, avg_price: float = 100.0, equity: float = 500.0):
        self._symbol = symbol
        self._holding = SimpleNamespace(Quantity=quantity, AveragePrice=avg_price)
        self.TotalPortfolioValue = equity
        self.Cash = equity
        self.CashBook = {"USD": SimpleNamespace(Amount=equity)}

    def __getitem__(self, symbol):
        if symbol == self._symbol:
            return self._holding
        raise KeyError(symbol)


class _Order:
    def __init__(self, order_id: int, symbol, quantity: float, tag: str, *, stop_price: float = 0.0, limit_price: float = 0.0):
        self.Id = order_id
        self.OrderId = order_id
        self.Symbol = symbol
        self.Quantity = quantity
        self.Tag = tag
        self.StopPrice = stop_price
        self.LimitPrice = limit_price
        self.Status = "Submitted"


def test_scalper_entry_fill_submits_single_bracket_and_ondata_does_not_duplicate(monkeypatch):
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    sym = _Symbol("SOLUSD")
    now = datetime(2025, 1, 10, 12, tzinfo=timezone.utc)
    logs: list[str] = []
    bracket_submissions: list[_Order] = []
    open_orders: dict[int, _Order] = {}
    order_seq = [10]

    portfolio = _Portfolio(sym, quantity=1.0, avg_price=100.0, equity=500.0)
    entry_ticket = SimpleNamespace(Tag="ScalperMom:entry")

    class _Transactions:
        def GetOpenOrders(self, symbol=None):
            orders = list(open_orders.values())
            if symbol is None:
                return orders
            return [o for o in orders if o.Symbol == symbol]

        def GetOrderTicket(self, order_id):
            if int(order_id) == 1:
                return entry_ticket
            return None

        def CancelOrder(self, _order_id):
            return None

    def _next_id():
        order_seq[0] += 1
        return order_seq[0]

    def _stop_market_order(symbol, qty, stop, tag=""):
        oid = _next_id()
        order = _Order(oid, symbol, qty, tag, stop_price=float(stop))
        open_orders[oid] = order
        bracket_submissions.append(order)
        return SimpleNamespace(OrderId=oid, Tag=tag)

    def _limit_order(symbol, qty, limit, tag=""):
        oid = _next_id()
        order = _Order(oid, symbol, qty, tag, limit_price=float(limit))
        open_orders[oid] = order
        bracket_submissions.append(order)
        return SimpleNamespace(OrderId=oid, Tag=tag)

    algo.Time = now
    algo._bar_count = 0
    algo.Debug = lambda msg: logs.append(str(msg))
    algo.reporter = SimpleNamespace(on_order_event=lambda *_args, **_kwargs: None)
    algo.Transactions = _Transactions()
    algo.StopMarketOrder = _stop_market_order
    algo.LimitOrder = _limit_order
    algo.Portfolio = portfolio
    algo.Securities = {sym: SimpleNamespace(Price=100.0)}
    algo.feature_engine = SimpleNamespace(current_features=lambda _key: {"atr": 1.0, "z_20h": -1.0, "ret_1h": 0.0, "ret_6h": 0.0})
    algo.position_state = {}
    algo._submitted_orders = {}
    algo._failed_escalations = {}
    algo._scalper_last_trade_time = {}
    algo._abandoned_dust = set()
    algo._scalper_consec_losses = {}
    algo._scalper_daily_pnl = 0.0
    algo._scalper_daily_anchor_equity = 500.0
    algo._scalper_daily_anchor_date = now.date()
    algo._scalper_daily_breaker_until = None
    algo._scalper_session_brake_until = None
    algo._scalper_recent_pnls = deque(maxlen=6)
    algo._scalper_entry_sleeve = {}
    algo._scalper_last_exit_by_sleeve = {}
    algo._scalper_symbol_cooldown_until = {}
    algo._scalper_latch_reset_day = now.date()
    algo._scalper_equity_peak = 500.0
    algo._breaker_liquidated = False
    algo._current_holdings = lambda: [sym] if portfolio._holding.Quantity > 0 else []
    algo.symbol_by_ticker = {"BTCUSD": _Symbol("BTCUSD")}
    algo.symbols = []
    algo._ingest_data = lambda _data: None
    algo.config = SimpleNamespace(
        strategy_mode="scalper",
        scalper_stop_atr_mult=1.5,
        scalper_tp_atr_mult=2.0,
        scalper_partial_tp_atr_mult=1.0,
        scalper_daily_loss_brake=-0.03,
        scalper_max_concurrent=4,
        scalper_universe_size=20,
        scalper_breaker_symbol_loss_pct=0.005,
        scalper_mr_max_hold_bars=12,
        scalper_mom_max_hold_bars=24,
        scalper_max_bars_held=24,
        scalper_stuck_hold_bars=24,
    )

    monkeypatch.setattr(scalper_module, "evaluate_entry", lambda **_kwargs: (False, "skip"))
    monkeypatch.setattr(scalper_module, "evaluate_exit", lambda **_kwargs: (False, ""))
    monkeypatch.setattr(main_module, "smart_liquidate", lambda *_args, **_kwargs: False)

    event = SimpleNamespace(Status="Filled", Symbol=sym, OrderId=1, FillPrice=100.0, Tag="ScalperMom:entry")
    algo.OnOrderEvent(event)
    assert len(bracket_submissions) == 2
    assert sum(1 for o in bracket_submissions if o.Tag == "SL") == 1
    assert sum(1 for o in bracket_submissions if o.Tag == "TP") == 1
    assert all((o.StopPrice > 0 if o.Tag == "SL" else o.LimitPrice > 0) for o in bracket_submissions)

    first_count = len(bracket_submissions)
    algo._scalper_on_data(SimpleNamespace())
    algo.Time = now.replace(hour=13)
    algo._scalper_on_data(SimpleNamespace())

    assert len(bracket_submissions) == first_count
    assert sum(1 for o in open_orders.values() if o.Tag == "SL") == 1
    assert sum(1 for o in open_orders.values() if o.Tag == "TP") == 1
    assert any("BRACKET_AUDIT sym=SOLUSD" in line for line in logs)
