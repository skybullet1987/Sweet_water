from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

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


class _Reporter:
    def daily_report(self):
        return {}

    def tick(self, _state):
        return None


class _Portfolio:
    TotalPortfolioValue = 500.0


def _build_algo(strategy_mode: str):
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.config = type("Cfg", (), {"strategy_mode": strategy_mode})()
    algo.Time = datetime(2025, 1, 2, tzinfo=timezone.utc)
    algo.reporter = _Reporter()
    algo.Debug = lambda *_args, **_kwargs: None
    algo._ensure_monthly_universe = lambda: None
    algo._bar_count = 0
    algo._last_daily_summary_date = None
    algo.IsWarmingUp = False
    algo._drawdown_breaker = _Breaker()
    algo._breaker_liquidated = False
    algo.Portfolio = _Portfolio()
    algo._breaker_disengaged_at = None
    algo._scalper_calls = 0
    algo._momentum_calls = 0
    algo._scalper_on_data = lambda _data: setattr(algo, "_scalper_calls", algo._scalper_calls + 1)
    algo._momentum_on_data = lambda _data: setattr(algo, "_momentum_calls", algo._momentum_calls + 1)
    return algo


def test_strategy_mode_scalper_dispatch(monkeypatch):
    monkeypatch.setattr(main_module, "manage_open_positions", lambda _algo, _data: [])
    algo = _build_algo("scalper")
    algo.OnData(type("Slice", (), {})())
    assert algo._scalper_calls == 1
    assert algo._momentum_calls == 0


def test_strategy_mode_momentum_dispatch(monkeypatch):
    monkeypatch.setattr(main_module, "manage_open_positions", lambda _algo, _data: [])
    algo = _build_algo("momentum")
    algo.OnData(type("Slice", (), {})())
    assert algo._scalper_calls == 0
    assert algo._momentum_calls == 1
