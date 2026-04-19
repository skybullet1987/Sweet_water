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


def _build_algo():
    sym = _Symbol("SOLUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    algo.config = StrategyConfig(top_k=1, min_rebalance_weight_delta=0.0, total_deployment_cap=1.0, max_position_pct=0.20)
    algo._breaker_disengaged_at = None
    algo._pending_rotation_entries = []
    algo._pending_rotation_entry_time = None
    algo.Debug = lambda *_args, **_kwargs: None
    algo._force_exit_losers = lambda _scored: set()
    algo.Securities = {
        sym: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0, "FeeModel": None, "SymbolProperties": type("Props", (), {"LotSize": 0.0001, "MinimumOrderSize": 0.0001})()})()
    }
    algo.Portfolio = _Portfolio({sym: _Holding(0.0)})
    algo.Portfolio.TotalPortfolioValue = 1000.0
    algo.Portfolio.Cash = 1000.0
    algo.Portfolio.CashBook = {"USD": type("Cash", (), {"Amount": 1000.0})()}
    algo.position_state = {}
    algo.sizer = type("Sizer", (), {"passes_cost_gate": staticmethod(lambda *_args, **_kwargs: True)})()
    algo.feature_engine = type("Feat", (), {"current_features": staticmethod(lambda *_args, **_kwargs: {"mom_24": 0.02, "vol_ratio_24h_7d": 2.0})})()
    return algo, sym


def test_conviction_cap_full_vs_half_regime(monkeypatch):
    algo, sym = _build_algo()
    submitted_qty = []
    monkeypatch.setattr(main_module, "place_entry", lambda _algo, _symbol, qty, **_kwargs: submitted_qty.append(float(qty)) or object())

    algo._dispersion_regime = lambda: "full"
    algo._rebalance_portfolio([(sym, 1.0, {"rv_21d": 0.1})], risk_scale=1.0)
    full_weight = submitted_qty[-1] * 100.0 / 1000.0
    assert abs(full_weight - 0.35) < 1e-6

    submitted_qty.clear()
    algo._dispersion_regime = lambda: "half"
    algo._rebalance_portfolio([(sym, 1.0, {"rv_21d": 0.1})], risk_scale=1.0)
    half_weight = submitted_qty[-1] * 100.0 / 1000.0
    assert abs(half_weight - 0.20) < 1e-6
