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
from main import SweetWaterPhase1


class _Breaker:
    def update(self, _algo):
        return None

    def is_triggered(self):
        return False


def test_breaker_disengage_clears_scalper_blocking_state(monkeypatch):
    now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    sym = type("_Sym", (), {"Value": "ETHUSD"})()
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = now
    algo.Debug = lambda *_args, **_kwargs: None
    algo.config = SimpleNamespace(strategy_mode="scalper")
    algo.reporter = SimpleNamespace(daily_report=lambda: {}, tick=lambda _state: None)
    algo._ensure_monthly_universe = lambda: None
    algo._bar_count = 0
    algo._last_daily_summary_date = None
    algo.IsWarmingUp = False
    algo._drawdown_breaker = _Breaker()
    algo._breaker_liquidated = True
    algo._breaker_disengaged_at = None
    algo.Portfolio = SimpleNamespace(TotalPortfolioValue=500.0)
    algo._scalper_session_brake_until = now + timedelta(hours=4)
    algo._scalper_consec_losses = {sym: 5}
    algo._scalper_recent_pnls = main_module.deque([-0.01, -0.02], maxlen=6)
    algo._failed_escalations = {sym: now}
    algo._pending_rotation_entries = [{"symbol": sym}]
    algo._pending_rotation_entry_time = now
    algo._scalper_on_data = lambda _data: None
    algo._momentum_on_data = lambda _data, btc_ret=None, breadth=None: None
    algo._ingest_data = lambda _data: (0.0, 0.5)

    monkeypatch.setattr(main_module, "manage_open_positions", lambda _algo, _data: [])

    algo.OnData(type("Slice", (), {})())

    assert algo._scalper_session_brake_until is None
    assert algo._scalper_consec_losses == {}
    assert len(algo._scalper_recent_pnls) == 0
    assert algo._failed_escalations == {}
    assert algo._pending_rotation_entries == []
