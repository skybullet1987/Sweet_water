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
from execution import PositionState
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


def test_rotation_defers_entries_until_next_bar(monkeypatch):
    old = _Symbol("OLDUSD")
    new = _Symbol("NEWUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    algo.config = StrategyConfig(top_k=1, min_hold_hours=0, min_rebalance_weight_delta=0.0, total_deployment_cap=1.0, max_position_pct=1.0)
    algo._breaker_disengaged_at = None
    algo._pending_rotation_entries = []
    algo._pending_rotation_entry_time = None
    algo._last_rebalance_time = None
    algo.Debug = lambda *_args, **_kwargs: None
    algo.sizer = type("Sizer", (), {"passes_cost_gate": lambda *_args, **_kwargs: True})()
    algo.Securities = {
        old: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0, "FeeModel": None, "SymbolProperties": type("Props", (), {"LotSize": 0.0001, "MinimumOrderSize": 0.0001})()})(),
        new: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0, "FeeModel": None, "SymbolProperties": type("Props", (), {"LotSize": 0.0001, "MinimumOrderSize": 0.0001})()})(),
    }
    holdings = {old: _Holding(1.0), new: _Holding(0.0)}
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
    algo.position_state = {old: PositionState(100.0, 100.0, 1.0, algo.Time - timedelta(hours=24))}
    algo._current_holdings = lambda: [old]
    algo._force_exit_losers = lambda _scored: set()

    exits = []
    entries = []

    def _liq(_algo, symbol, tag=""):
        exits.append((symbol, tag))
        return True

    def _entry(_algo, symbol, qty, tag="", signal_score=None, force_market=False):
        _ = signal_score, force_market
        entries.append((symbol, float(qty), tag))
        return object()

    monkeypatch.setattr(main_module, "smart_liquidate", _liq)
    monkeypatch.setattr(main_module, "place_entry", _entry)

    scored = [(new, 1.0, {"rv_21d": 0.2}), (old, 0.1, {"rv_21d": 0.2})]
    algo._rebalance_portfolio(scored, risk_scale=1.0)

    assert exits
    assert entries == []
    assert len(algo._pending_rotation_entries) == 1

    algo.Time = algo.Time + timedelta(hours=1)
    algo._process_pending_entries(scored_lookup=None)
    assert entries
    assert algo._pending_rotation_entries == []
