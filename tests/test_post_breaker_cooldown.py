from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
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


class _Holding:
    def __init__(self, qty=0.0):
        self.Quantity = float(qty)


def _build_algo(now: datetime):
    sym = _Symbol("ETHUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = now
    algo.config = StrategyConfig(
        top_k=1,
        min_rebalance_weight_delta=0.0,
        post_breaker_cooldown_hours=48,
        max_position_pct=1.0,
        total_deployment_cap=1.0,
    )
    algo._pending_rotation_entries = []
    algo._pending_rotation_entry_time = None
    algo._last_rebalance_time = None
    algo.position_state = {}
    algo._current_holdings = lambda: []
    algo._force_exit_losers = lambda _scored: set()
    algo.Debug = lambda *_args, **_kwargs: None
    algo.sizer = type("Sizer", (), {"passes_cost_gate": lambda *_args, **_kwargs: True})()
    algo.Securities = {
        sym: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0, "FeeModel": None, "SymbolProperties": type("Props", (), {"LotSize": 0.0001, "MinimumOrderSize": 0.0001})()})()
    }
    holdings = {sym: _Holding(0.0)}
    algo.Portfolio = type(
        "Portfolio",
        (),
        {
            "TotalPortfolioValue": 500.0,
            "Cash": 500.0,
            "CashBook": {"USD": type("Cash", (), {"Amount": 500.0})()},
            "__getitem__": lambda _self, s: holdings[s],
        },
    )()
    return algo, sym


def test_rebalance_respects_post_breaker_cooldown(monkeypatch):
    now = datetime(2025, 1, 3, 16, tzinfo=timezone.utc)
    algo, sym = _build_algo(now)
    submitted = []
    monkeypatch.setattr(
        main_module,
        "place_entry",
        lambda _algo, symbol, qty, tag="", signal_score=None, force_market=False: submitted.append((symbol, qty, tag)) or object(),
    )
    scored = [(sym, 1.0, {"rv_21d": 0.2})]

    algo._breaker_disengaged_at = now - timedelta(hours=24)
    algo._rebalance_portfolio(scored, risk_scale=1.0)
    assert submitted == []

    algo._breaker_disengaged_at = now - timedelta(hours=50)
    algo._rebalance_portfolio(scored, risk_scale=1.0)
    assert submitted
    assert algo._breaker_disengaged_at is None
