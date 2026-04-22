from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import main as main_module
import scalper as scalper_module
from execution import PositionState
from main import SweetWaterPhase1
from scalper_runtime import _ensure_scalper_brackets


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


def test_scalper_time_stop_flattens_when_brackets_never_stick(monkeypatch):
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    sym = _Symbol("XRPUSD")
    now = datetime(2025, 1, 10, 12, tzinfo=timezone.utc)
    logs: list[str] = []
    submitted: list[tuple[str, float]] = []

    portfolio = _Portfolio(sym, quantity=1.0, avg_price=100.0, equity=500.0)

    algo.Time = now
    algo.Debug = lambda msg: logs.append(str(msg))
    algo._ingest_data = lambda _data: None
    algo.symbols = []
    algo.Securities = {sym: SimpleNamespace(Price=99.5)}
    algo.Portfolio = portfolio
    algo.symbol_by_ticker = {"BTCUSD": _Symbol("BTCUSD")}
    algo.feature_engine = SimpleNamespace(
        current_features=lambda key: {"ret_1h": 0.0, "ret_6h": 0.0, "atr": 1.0, "z_20h": -1.0} if key == "BTCUSD" else {"atr": 1.0, "z_20h": -1.0}
    )
    algo.position_state = {
        sym: PositionState(
            entry_price=100.0,
            highest_close=100.0,
            entry_atr=1.0,
            entry_time=now - timedelta(hours=23),
            strategy_owner="scalper",
            initial_risk_distance=1.5,
            stop_price=98.5,
            take_profit_price=102.5,
            partial_tp_price=101.0,
            trail_anchor_price=100.0,
        )
    }
    algo._current_holdings = lambda: [sym] if portfolio._holding.Quantity > 0 else []
    algo._scalper_last_trade_time = {}
    algo._scalper_consec_losses = {}
    algo._scalper_daily_pnl = 0.0
    algo._scalper_daily_anchor_equity = 500.0
    algo._scalper_daily_anchor_date = now.date()
    algo._scalper_session_brake_until = None
    algo._scalper_recent_pnls = main_module.deque(maxlen=6)
    algo._scalper_entry_sleeve = {}
    algo._scalper_last_exit_by_sleeve = {}
    algo._failed_escalations = {}
    algo._scalper_symbol_cooldown_until = {}
    algo._scalper_latch_reset_day = now.date()
    algo._scalper_equity_peak = 500.0
    algo._breaker_liquidated = False
    algo.Transactions = SimpleNamespace(GetOpenOrders=lambda *_args, **_kwargs: [], CancelOrder=lambda *_args, **_kwargs: None)
    algo.StopMarketOrder = lambda _symbol, qty, _stop, tag="": submitted.append((tag, qty)) or object()
    algo.LimitOrder = lambda _symbol, qty, _limit, tag="": submitted.append((tag, qty)) or object()
    algo.config = SimpleNamespace(
        scalper_daily_loss_brake=-0.03,
        scalper_max_concurrent=4,
        scalper_universe_size=20,
        scalper_position_size_pct=0.15,
        min_position_floor_usd=5.0,
        scalper_mr_max_hold_bars=12,
        scalper_mom_max_hold_bars=24,
        scalper_max_bars_held=24,
        scalper_stuck_hold_bars=24,
        scalper_breaker_symbol_loss_pct=0.005,
    )

    monkeypatch.setattr(scalper_module, "evaluate_entry", lambda **_kwargs: (False, "skip"))
    monkeypatch.setattr(scalper_module, "evaluate_exit", lambda **_kwargs: (False, ""))

    def _smart_liquidate(_algo, _symbol, tag=""):
        if tag == "TimeStop":
            portfolio._holding.Quantity = 0.0
            return True
        return False

    monkeypatch.setattr(main_module, "smart_liquidate", _smart_liquidate)

    algo._scalper_on_data(SimpleNamespace())
    assert portfolio._holding.Quantity == 1.0

    algo.Time = now + timedelta(hours=2)
    algo._scalper_on_data(SimpleNamespace())

    assert portfolio._holding.Quantity == 0.0
    assert sym not in algo.position_state
    assert any("BRACKET_AUDIT sym=XRPUSD" in line for line in logs)
    assert any("SCALPER_EXIT sym=XRPUSD tag=TimeStop" in line for line in logs)
    assert submitted == []


def test_bracket_checker_does_not_submit_when_called_from_on_data():
    sym = _Symbol("SOLUSD")
    submitted: list[tuple[str, float, float]] = []
    state = PositionState(
        entry_price=100.0,
        highest_close=100.0,
        entry_atr=1.0,
        entry_time=datetime(2025, 1, 10, tzinfo=timezone.utc),
        stop_price=98.5,
        take_profit_price=102.5,
    )
    algo = SimpleNamespace(
        Transactions=SimpleNamespace(GetOpenOrders=lambda *_args, **_kwargs: []),
        StopMarketOrder=lambda _symbol, qty, stop, tag="": submitted.append((tag, qty, stop)) or object(),
        LimitOrder=lambda _symbol, qty, limit, tag="": submitted.append((tag, qty, limit)) or object(),
        Debug=lambda *_args, **_kwargs: None,
    )

    assert hasattr(state, "bracket_attempted_qty")
    delattr(state, "bracket_attempted_qty")
    has_sl, has_tp = _ensure_scalper_brackets(algo, sym, qty_now=1.0, side=1, state=state)
    _ensure_scalper_brackets(algo, sym, qty_now=1.0, side=1, state=state)

    assert not has_sl
    assert not has_tp
    assert submitted == []
    assert state.bracket_attempted_qty == 0.0
