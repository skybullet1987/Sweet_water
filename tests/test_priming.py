from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from features import FeatureEngine
from main import SweetWaterPhase1
from regime import RegimeEngine


class _Symbol:
    def __init__(self, value: str):
        self.Value = value


def test_prime_features_history_populates_long_momentum():
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.config = StrategyConfig()
    algo.Time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    algo.Debug = lambda *_args, **_kwargs: None
    algo.feature_engine = FeatureEngine(signal_mode="cross_sectional_momentum")
    algo.regime_engine = RegimeEngine(algo.config)
    sym = _Symbol("SOLUSD")
    algo.symbol_by_ticker = {"SOLUSD": sym}

    def _history(_symbol, _start, _end, _resolution):
        n = 140 * 24
        ts = pd.date_range("2024-08-14", periods=n, freq="h", tz="UTC")
        close = np.linspace(100.0, 180.0, n)
        df = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.001, "low": close * 0.998, "close": close, "volume": 1000.0},
            index=pd.MultiIndex.from_arrays([["SOLUSD"] * n, ts]),
        )
        return df

    algo.History = _history
    algo._prime_features_from_history()
    feats = algo.feature_engine.current_features("SOLUSD")
    assert abs(float(feats.get("mom_90d", 0.0) or 0.0)) > 1e-6
