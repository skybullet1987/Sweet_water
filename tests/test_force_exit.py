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


def test_force_exit_zscore_loser_only(monkeypatch):
    a = _Symbol("A")
    b = _Symbol("B")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 16, tzinfo=timezone.utc)
    algo.config = StrategyConfig(top_k=2, min_rebalance_weight_delta=1.0)
    algo._last_rebalance_time = None
    algo.position_state = {}
    algo.Debug = lambda *_args, **_kwargs: None
    algo.symbols = [a, b]
    algo._current_holdings = lambda: [a, b]
    algo.Securities = {
        a: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0})(),
        b: type("Sec", (), {"Price": 100.0, "BidPrice": 99.0, "AskPrice": 101.0})(),
    }
    algo.Portfolio = type(
        "P",
        (),
        {
            "TotalPortfolioValue": 500.0,
            "__getitem__": lambda _self, _sym: type("H", (), {"Quantity": 0.0})(),
        },
    )()
    algo.sizer = type("S", (), {"passes_cost_gate": lambda *_args, **_kwargs: True})()
    calls = []

    def _liq(_algo, sym, tag=""):
        calls.append((sym.Value, tag))
        return True

    monkeypatch.setattr(main_module, "smart_liquidate", _liq)
    scored = [(a, -0.7, {"rv_21d": 0.2}), (b, -0.3, {"rv_21d": 0.2})]
    algo._rebalance_portfolio(scored, risk_scale=1.0)
    assert ("A", "ZScoreLoser") in calls
    assert ("B", "ZScoreLoser") not in calls
