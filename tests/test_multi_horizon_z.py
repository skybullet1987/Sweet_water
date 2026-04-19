from __future__ import annotations

import statistics
import sys
from datetime import datetime, timezone
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


def test_multi_horizon_zscores_are_standardized():
    symbols = [_Symbol(v) for v in ("A", "B", "C", "D", "E")]
    vals = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}
    feature_map = {
        k: {"mom_21d_skip": v, "mom_63d_skip": v * 1.2, "mom_90d_skip": v * 1.5, "rv_21d": 1.0}
        for k, v in vals.items()
    }
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.symbols = symbols
    algo.Time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    algo._bar_count = 1
    algo._dispersion_history = []
    algo._last_dispersion_date = None
    algo.feature_engine = type("FE", (), {"current_features": lambda _self, ticker: dict(feature_map.get(ticker, {}))})()
    algo.signal_features = None
    algo.scorer = Scorer(StrategyConfig(signal_mode="cross_sectional_momentum", score_clip_value=10.0))
    algo.Debug = lambda *_args, **_kwargs: None
    algo.regime_engine = type(
        "RE",
        (),
        {
            "hurst": type("H", (), {"hurst": lambda *_args, **_kwargs: 0.5, "hurst_change_30d": lambda *_args, **_kwargs: 0.0})(),
            "vr": type("V", (), {"regime": lambda *_args, **_kwargs: "random"})(),
        },
    )()
    scored = algo._collect_scores("risk_on", btc_ret=0.0)
    finals = [score for _, score, _ in scored]
    assert abs(statistics.fmean(finals)) < 1e-9
    assert abs(statistics.pstdev(finals) - 1.0) < 1e-9
