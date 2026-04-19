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


class _Holding:
    def __init__(self, qty=0.0):
        self.Quantity = float(qty)
        self.AveragePrice = 0.0


class _Portfolio(dict):
    pass


def test_rebalance_skips_entry_on_no_chase(monkeypatch):
    sym = _Symbol("SOLUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    algo.config = StrategyConfig(top_k=1, min_rebalance_weight_delta=0.0, total_deployment_cap=1.0, max_position_pct=0.20)
    algo._breaker_disengaged_at = None
    algo._pending_rotation_entries = []
    algo._pending_rotation_entry_time = None
    algo._dispersion_history = []
    algo.Debug = lambda *_args, **_kwargs: None
    algo._force_exit_losers = lambda _scored: set()
    algo.Securities = {
        sym: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0, "FeeModel": None, "SymbolProperties": type("Props", (), {"LotSize": 0.0001, "MinimumOrderSize": 0.0001})()})()
    }
    algo.Portfolio = _Portfolio({sym: _Holding(0.0)})
    algo.Portfolio.TotalPortfolioValue = 500.0
    algo.Portfolio.Cash = 500.0
    algo.Portfolio.CashBook = {"USD": type("Cash", (), {"Amount": 500.0})()}
    algo.position_state = {}
    algo.sizer = type("Sizer", (), {"passes_cost_gate": staticmethod(lambda *_args, **_kwargs: True)})()
    algo.feature_engine = type("Feat", (), {"current_features": staticmethod(lambda *_args, **_kwargs: {"mom_24": 0.16, "vol_ratio_24h_7d": 2.0})})()
    placed = []
    monkeypatch.setattr(main_module, "place_entry", lambda *_args, **_kwargs: placed.append(True) or object())

    algo._rebalance_portfolio([(sym, 1.0, {"rv_21d": 0.2})], risk_scale=1.0)
    assert placed == []
