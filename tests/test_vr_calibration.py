from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from regime import VarianceRatioRegimeModel


class _Symbol:
    def __init__(self, value: str):
        self.Value = value


class _Bar:
    def __init__(self, t, close):
        self.EndTime = t
        self.Close = close


def test_vr_calibration_populates_threshold_cache_and_is_used():
    rng = np.random.default_rng(11)
    model = VarianceRatioRegimeModel(window=5000, min_samples=500)
    sym = _Symbol("BTCUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    px = 100.0
    prev_ret = 0.0
    for i in range(1300):
        ret = float(0.3 * prev_ret + rng.normal(0.0, 0.01))
        prev_ret = ret
        px *= float(np.exp(ret))
        model.update(sym, _Bar(t + timedelta(hours=i), px))
    key = sym.Value
    assert key in model._threshold_cache
    low, high = model._threshold_cache[key]
    model._vr[key] = high + 0.02
    assert model.regime(sym) == "trend"
    model._vr[key] = low - 0.02
    assert model.regime(sym) == "meanrev"
