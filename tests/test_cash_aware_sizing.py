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


def test_rebalance_entry_notional_clamped_by_available_cash(monkeypatch):
    sym = _Symbol("ETHUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    algo.config = StrategyConfig(top_k=1, min_rebalance_weight_delta=0.0, max_position_pct=0.50, total_deployment_cap=1.0)
    algo._breaker_disengaged_at = None
    algo._pending_rotation_entries = []
    algo._pending_rotation_entry_time = None
    algo._last_rebalance_time = None
    algo.position_state = {}
    algo._current_holdings = lambda: []
    algo._force_exit_losers = lambda _scored: set()
    algo.Debug = lambda *_args, **_kwargs: None
    algo.sizer = type("Sizer", (), {"passes_cost_gate": lambda *_args, **_kwargs: True})()
    algo.feature_engine = type("Feat", (), {"current_features": staticmethod(lambda *_args, **_kwargs: {"mom_24": 0.0, "vol_ratio_24h_7d": 2.0})})()
    algo.Securities = {
        sym: type(
            "Sec",
            (),
            {
                "Price": 100.0,
                "BidPrice": 99.0,
                "AskPrice": 101.0,
                "SymbolProperties": type("Props", (), {"LotSize": 0.0001, "MinimumOrderSize": 0.0001})(),
                "FeeModel": None,
            },
        )()
    }
    holdings = {sym: _Holding(0.0)}
    algo.Portfolio = type(
        "Portfolio",
        (),
        {
            "TotalPortfolioValue": 500.0,
            "Cash": 30.0,
            "CashBook": {"USD": type("Cash", (), {"Amount": 30.0})()},
            "__getitem__": lambda _self, s: holdings[s],
        },
    )()
    submitted = []

    def _place_entry(_algo, symbol, qty, tag="", signal_score=None, force_market=False):
        _ = tag, signal_score, force_market
        submitted.append((symbol, float(qty)))
        return object()

    monkeypatch.setattr(main_module, "place_entry", _place_entry)
    scored = [(sym, 1.0, {"rv_21d": 0.2})]
    algo._rebalance_portfolio(scored, risk_scale=0.3)

    assert len(submitted) == 1
    _, qty = submitted[0]
    notional = qty * 100.0
    assert notional <= 29.1 + 1e-9
    assert notional > 28.5
