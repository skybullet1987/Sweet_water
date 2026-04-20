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


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


def test_scalper_lazy_seed_is_idempotent_and_keeps_existing_entry_time(monkeypatch):
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    now = datetime(2025, 1, 2, 8, tzinfo=timezone.utc)
    t0 = now - timedelta(hours=5)
    sym = _Symbol("ETHUSD")
    algo.Time = now
    algo.Debug = lambda *_args, **_kwargs: None
    algo.config = SimpleNamespace(
        scalper_daily_loss_brake=-0.03,
        scalper_max_concurrent=4,
        scalper_universe_size=20,
        scalper_position_size_pct=0.15,
        min_position_floor_usd=5.0,
        scalper_use_btc_ema_gate=False,
    )
    algo._current_holdings = lambda: [sym]
    algo.symbols = []
    algo.Portfolio = type(
        "_P",
        (),
        {
            "TotalPortfolioValue": 500.0,
            "Cash": 500.0,
            "CashBook": {"USD": SimpleNamespace(Amount=500.0)},
            "__getitem__": lambda _self, _symbol: SimpleNamespace(AveragePrice=100.0),
        },
    )()
    algo.Securities = {sym: SimpleNamespace(Price=101.0)}
    algo.symbol_by_ticker = {"BTCUSD": _Symbol("BTCUSD")}
    algo.feature_engine = SimpleNamespace(
        current_features=lambda key: {"ret_1h": 0.0, "ret_6h": 0.0}
        if key == "BTCUSD"
        else {"z_20h": -1.0, "atr": 3.0}
    )
    algo.regime_engine = SimpleNamespace(btc_above_ema30d=lambda: True)
    algo.position_state = {
        sym: PositionState(entry_price=100.0, highest_close=101.0, entry_atr=3.0, entry_time=t0, strategy_owner="scalper")
    }
    algo._scalper_last_trade_time = {}
    algo._scalper_consec_losses = {}
    algo._scalper_daily_pnl = 0.0
    algo._scalper_daily_anchor_equity = 500.0
    algo._scalper_daily_anchor_date = now.date()
    algo._scalper_session_brake_until = None
    algo._scalper_recent_pnls = main_module.deque(maxlen=6)
    algo._failed_escalations = {}
    algo._breaker_liquidated = False

    monkeypatch.setattr(scalper_module, "evaluate_exit", lambda **kwargs: (False, ""))
    monkeypatch.setattr(scalper_module, "evaluate_entry", lambda **kwargs: (False, "skip"))

    algo._scalper_on_data(SimpleNamespace())
    algo._scalper_on_data(SimpleNamespace())

    assert algo.position_state[sym].entry_time == t0
