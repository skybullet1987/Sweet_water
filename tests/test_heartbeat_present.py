from __future__ import annotations

import sys
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


def test_scalper_heartbeat_log_present(monkeypatch):
    logs = []
    now = datetime(2025, 1, 3, 1, tzinfo=timezone.utc)
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = now
    algo.Debug = lambda msg: logs.append(str(msg))
    algo.config = SimpleNamespace(
        scalper_daily_loss_brake=-0.03,
        scalper_max_concurrent=4,
        scalper_universe_size=20,
        scalper_position_size_pct=0.15,
        min_position_floor_usd=5.0,
        scalper_use_btc_ema_gate=False,
    )
    algo.Portfolio = SimpleNamespace(
        TotalPortfolioValue=500.0,
        Cash=500.0,
        CashBook={"USD": SimpleNamespace(Amount=500.0)},
    )
    algo._current_holdings = lambda: []
    algo.symbol_by_ticker = {"BTCUSD": _Symbol("BTCUSD")}
    algo.feature_engine = SimpleNamespace(current_features=lambda key: {"ret_1h": 0.0, "ret_6h": 0.0} if key == "BTCUSD" else {})
    algo.regime_engine = SimpleNamespace(btc_above_ema30d=lambda: True)
    algo.symbols = []
    algo.Securities = {}
    algo.position_state = {}
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
    assert any(line.startswith("SCALPER_HB") for line in logs)
