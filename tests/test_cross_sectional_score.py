from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from main import SweetWaterPhase1
from scoring import Scorer


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


def test_collect_scores_uses_cross_sectional_zscores():
    symbols = [_Symbol(v) for v in ("A", "B", "C", "D", "E")]
    signal_values = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
    feature_map = {k: {"mom_21d": v, "rv_21d": 1.0} for k, v in signal_values.items()}

    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.symbols = symbols
    algo.feature_engine = type("FE", (), {"current_features": lambda _self, ticker: feature_map.get(ticker, {})})()
    algo.signal_features = None
    algo.regime_engine = None
    algo.scorer = Scorer(StrategyConfig(signal_mode="cross_sectional_momentum", score_clip_value=10.0))

    scored = algo._collect_scores("risk_on", btc_ret=0.0)
    finals = [score for _, score, _ in scored]
    assert abs(statistics.fmean(finals)) < 1e-9
    assert abs(statistics.pstdev(finals) - 1.0) < 1e-9
