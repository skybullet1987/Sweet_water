from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from kraken_ops import DriftMonitor  # noqa: E402
from kraken_ops import FillTracker  # noqa: E402
from risk import allocate_erc_notionals, erc_weights, shrink_covariance  # noqa: E402
from regime import UnifiedRegimeEngine  # noqa: E402
from core import FeatureCache, compute_bar_features  # noqa: E402


def _hourly_config():
    return replace(CONFIG, resolution_minutes=60, use_sub_hour_bars=False, feature_min_bars=48)


def test_v5_sub_hour_config():
    assert CONFIG.resolution_minutes == 15
    assert CONFIG.bph() == 4
    assert CONFIG.erc_shrinkage > 0


def test_shrink_covariance():
    cov = pd.DataFrame([[0.04, 0.02], [0.02, 0.05]], index=["A", "B"], columns=["A", "B"])
    shrunk = shrink_covariance(cov, 0.5)
    assert shrunk.loc["A", "A"] > cov.loc["A", "A"] * 0.5


def test_erc_turnover_blend():
    cov = pd.DataFrame([[0.05, 0.01], [0.01, 0.06]], index=["A", "B"], columns=["A", "B"])
    w = erc_weights(shrink_covariance(cov, 0.2))
    assert abs(sum(w.values()) - 1.0) < 0.01


def test_fill_tracker_fill_rate():
    ft = FillTracker()
    ft.on_submit(1, is_limit=True, expected_price=100.0, qty=1.0)
    ft.on_fill(1, 100.5, is_limit=True)
    ft.on_submit(2, is_limit=True, expected_price=50.0, qty=2.0)
    ft.on_cancel(2)
    assert ft.stats.fill_rate == 1.0
    assert ft.stats.avg_slippage_bps > 0


def test_drift_monitor_alert():
    dm = DriftMonitor(replace(CONFIG, baseline_sharpe=1.0, drift_sharpe_ratio_threshold=0.5))
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(100):
        dm.record_equity(now, 1000.0 * (0.999 ** i))
    alert, msg = dm.should_alert()
    assert alert
    assert "drift" in msg


def test_unified_regime_engine():
    engine = UnifiedRegimeEngine(_hourly_config())
    for i in range(600):
        engine.update_market(
            btc_close=100.0 + i * 0.01,
            btc_return=0.001,
            btc_vol=0.3,
            breadth=0.6,
            btc_above_ema200=True,
            ema200=99.0,
        )
    reg = engine.classify_unified(
        btc_features={"trend_quality": 0.04, "mom_21d": 0.1, "close": 100, "ema200": 99},
        breadth=0.65,
        median_rv=0.4,
        btc_return=0.001,
        btc_vol=0.4,
        btc_above_ema200=True,
    )
    assert reg.deployment_cap >= 0


def test_features_hourly_compat():
    cfg = _hourly_config()
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    close = np.linspace(100, 110, 200)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(200, 1000.0),
        },
        index=idx,
    )
    feats = compute_bar_features(frame, cfg)
    assert "mom_21d" in feats
