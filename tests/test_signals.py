from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from features import CvdDivergenceFeature, FeatureEngine, OrderFlowImbalanceFeature, VolConeBreakoutFeature
from regime import HurstRegimeModel, VarianceRatioRegimeModel

MAX_FILE_BYTES = 60_000
MAX_TOTAL_LOC = 6_000


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
    sig = CvdDivergenceFeature()
    sym = _Symbol("ADAUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(25):
        sig.update(sym, _Bar(t + timedelta(hours=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 10))
    sig.update(sym, _Bar(t + timedelta(hours=25), 130, 150, 129, 129.2, 1))
    assert sig.score(sym) <= -0.9


def test_garman_klass_value_matches_formula():
    gk = VolConeBreakoutFeature.gk_value(100.0, 105.0, 95.0, 102.0)
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
    h_trend = HurstRegimeModel.hurst_rs(trend[200:])
    h_mean = HurstRegimeModel.hurst_rs(meanrev[200:])
    assert h_trend > h_mean


def test_variance_ratio_trend_ar1_gt_one():
    rng = np.random.default_rng(11)
    n = 1400
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = 0.8 * rets[i - 1] + rng.normal(0, 0.01)
    model = VarianceRatioRegimeModel(window=500, min_samples=120)
    sym = _Symbol("BTCUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    px = 100.0
    for i in range(n):
        px *= float(np.exp(rets[i]))
        model.update(sym, _Bar(t + timedelta(hours=i), px, px, px, px, 1.0))
    assert model.variance_ratio(sym) > 1.0


def test_variance_ratio_meanrev_ar1_lt_one():
    rng = np.random.default_rng(12)
    n = 1400
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = -0.8 * rets[i - 1] + rng.normal(0, 0.01)
    model = VarianceRatioRegimeModel(window=500, min_samples=120)
    sym = _Symbol("BTCUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    px = 100.0
    for i in range(n):
        px *= float(np.exp(rets[i]))
        model.update(sym, _Bar(t + timedelta(hours=i), px, px, px, px, 1.0))
    assert model.variance_ratio(sym) < 1.0


def test_ofi_sign_convention_positive_bid_improves():
    sig = OrderFlowImbalanceFeature()
    sym = _Symbol("ETHUSD")
    t = datetime(2025, 1, 1, tzinfo=timezone.utc)
    sig.update(sym, _Quote(t, 100.0, 10.0, 101.0, 11.0))
    sig.update(sym, _Quote(t + timedelta(minutes=1), 100.5, 12.0, 100.9, 9.0))
    event = sig._event_ofi("ETHUSD", 100.6, 15.0, 100.8, 8.0)
    assert event > 0


def test_feature_engine_incremental_outputs_required_keys():
    engine = FeatureEngine(signal_mode="microstructure")
    for i in range(240):
        px = 100.0 + 0.1 * i
        engine.update(
            {
                "symbol": "BTCUSD",
                "open": px * 0.999,
                "high": px * 1.002,
                "low": px * 0.998,
                "close": px,
                "volume": 1000.0 + i,
            }
        )
    feats = engine.current_features("BTCUSD")
    required = {"atr", "realized_vol_30", "mom_24", "mom_168", "ema20", "ema50", "ema200"}
    assert required.issubset(feats.keys())


def test_module_count_and_size():
    files = sorted([x for x in os.listdir(QC_RUNTIME) if x.endswith(".py")])
    expected = {
        "main.py",
        "config.py",
        "features.py",
        "regime.py",
        "scoring.py",
        "sizing.py",
        "execution.py",
        "risk.py",
        "reporting.py",
        "universe.py",
        "scalper.py",
        "scalper_runtime.py",
        "scalper_signals.py",
        "signals.py",
    }
    assert set(files) == expected
    assert len(files) == 14
    assert not any(p.is_dir() and p.name != "__pycache__" for p in QC_RUNTIME.iterdir())
    total_loc = 0
    for f in files:
        p = QC_RUNTIME / f
        assert p.stat().st_size < MAX_FILE_BYTES
        with p.open("r", encoding="utf-8") as fh:
            total_loc += sum(1 for _ in fh)
    assert total_loc < MAX_TOTAL_LOC
