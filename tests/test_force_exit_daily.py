from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import main as main_module
from config import StrategyConfig
from main import SweetWaterPhase1


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _Slice:
    Bars = {}
    Ticks = {}


def test_force_exit_losers_runs_once_daily_at_hour_8(monkeypatch):
    sym = _Symbol("ADAUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 8, tzinfo=timezone.utc)
    algo._bar_count = 0
    algo._last_trade_bar = 0
    algo._last_no_trade_log_bar = 0
    algo._last_daily_summary_date = algo.Time.date()
    algo._last_dispersion_log_date = None
    algo._last_force_exit_date = None
    algo._dispersion_history = []
    algo._last_scored = [(sym, -0.7, {})]
    algo.IsWarmingUp = False
    algo._breaker_liquidated = False
    algo._ensure_monthly_universe = lambda: None
    algo._process_pending_entries = lambda scored_lookup=None: None
    algo._score_candidates = lambda _data: ("risk_on", [])
    algo._dispersion_regime = lambda: "full"
    algo._rebalance_due = lambda: False
    algo.reporter = type(
        "Reporter",
        (),
        {
            "daily_report": lambda *_args, **_kwargs: {"daily_trade_count": 0},
            "tick": lambda *_args, **_kwargs: None,
            "on_order_event": lambda *_args, **_kwargs: None,
        },
    )()
    algo._drawdown_breaker = type("Breaker", (), {"update": lambda *_args, **_kwargs: None, "is_triggered": lambda *_args, **_kwargs: False})()
    algo.config = StrategyConfig(strategy_mode="momentum")
    algo.Debug = lambda *_args, **_kwargs: None

    calls = []
    monkeypatch.setattr(main_module, "manage_open_positions", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(main_module, "escalate_stale_orders", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(algo, "_force_exit_losers", lambda scored: calls.append(list(scored)) or {sym})

    algo.OnData(_Slice())
    assert len(calls) == 1

    algo.Time = datetime(2025, 1, 1, 9, tzinfo=timezone.utc)
    algo.OnData(_Slice())
    assert len(calls) == 1
