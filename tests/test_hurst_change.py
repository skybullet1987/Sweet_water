from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from regime import HurstRegimeModel


class _Symbol:
    def __init__(self, value: str):
        self.Value = value


class _Bar:
    def __init__(self, t, close):
        self.EndTime = t
        self.Close = close


def test_hurst_change_30d_positive_for_trending_transition():
    rng = np.random.default_rng(7)
    model = HurstRegimeModel(window=80)
    sym = _Symbol("BTCUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    px = 100.0
    for i in range(70 * 24):
        if i < 35 * 24:
            ret = float(rng.normal(0.0, 0.002))
        else:
            ret = float(0.0008 + rng.normal(0.0, 0.0007))
        px *= float(np.exp(ret))
        model.update(sym, _Bar(t + timedelta(hours=i), px))
    assert model.hurst_change_30d(sym) > 0.05
