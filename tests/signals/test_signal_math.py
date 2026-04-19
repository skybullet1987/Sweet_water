from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from signals.cvd_divergence import CvdDivergenceSignal
from signals.hurst_regime import HurstRegimeSignal
from signals.order_flow_imbalance import OrderFlowImbalanceSignal
from signals.vol_cone_breakout import VolConeBreakoutSignal


class _Symbol:
    def __init__(self, value: str):
        self.Value = value


class _Bar:
    def __init__(self, t, open_, high, low, close, volume):
        self.EndTime = t
        self.Open = float(open_)
        self.High = float(high)
        self.Low = float(low)
        self.Close = float(close)
        self.Volume = float(volume)


class _Quote:
    def __init__(self, t, bid_p, bid_q, ask_p, ask_q):
        self.Time = t
        self.BidPrice = float(bid_p)
        self.BidSize = float(bid_q)
        self.AskPrice = float(ask_p)
        self.AskSize = float(ask_q)


def test_cvd_bearish_divergence_detected():
    sig = CvdDivergenceSignal()
    sym = _Symbol("ADAUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(25):
        sig.update(sym, _Bar(t + timedelta(hours=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 10))
    sig.update(sym, _Bar(t + timedelta(hours=25), 130, 150, 129, 129.2, 1))
    assert sig.score(sym) <= -0.9


def test_garman_klass_value_matches_formula():
    gk = VolConeBreakoutSignal.gk_value(100.0, 105.0, 95.0, 102.0)
    expected = 0.5 * (math.log(105.0 / 95.0) ** 2) - (2.0 * math.log(2.0) - 1.0) * (math.log(102.0 / 100.0) ** 2)
    assert abs(gk - expected) < 1e-12


def test_hurst_distinguishes_trend_vs_mean_reversion():
    rng = np.random.default_rng(7)
    n = 1200
    trend = np.zeros(n)
    meanrev = np.zeros(n)
    for i in range(1, n):
        trend[i] = 0.8 * trend[i - 1] + rng.normal(0, 0.01)
        meanrev[i] = -0.8 * meanrev[i - 1] + rng.normal(0, 0.01)
    h_trend = HurstRegimeSignal.hurst_rs(trend[200:])
    h_mean = HurstRegimeSignal.hurst_rs(meanrev[200:])
    assert h_trend > h_mean


def test_ofi_sign_convention_positive_bid_improves():
    sig = OrderFlowImbalanceSignal()
    sym = _Symbol("ETHUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    sig.update(sym, _Quote(t, 100.0, 10.0, 101.0, 11.0))
    sig.update(sym, _Quote(t + timedelta(minutes=1), 100.5, 12.0, 100.9, 9.0))
    event = sig._event_ofi("ETHUSD", 100.6, 15.0, 100.8, 8.0)
    assert event > 0
