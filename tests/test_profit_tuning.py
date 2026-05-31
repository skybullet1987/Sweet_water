from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from core import FeatureCache, filter_uncorrelated_picks, return_correlation  # noqa: E402
from data import SentimentSnapshot, adjust_deployment_cap  # noqa: E402


def test_fear_boosts_neutral_cap():
    snap = SentimentSnapshot(fear_greed=0.1, btc_dominance=0.5, funding_proxy=0.5)
    boosted = adjust_deployment_cap(0.80, snap, "neutral", CONFIG)
    assert boosted > 0.80


def test_correlation_uses_pairwise_overlap():
    cache = FeatureCache()
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    for name, scale in [("AAAUSD", 1.0), ("BBBUSD", 1.0)]:
        close = np.cumprod(1 + rng.normal(0, 0.01, 80)) * 100 * scale
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
            index=idx,
        )
        for _, row in df.iterrows():
            cache._bars[name].append(row.to_dict())
    corr = return_correlation(cache, ["AAAUSD", "BBBUSD"], min_samples=20)
    assert not corr.empty


def test_decorrelation_relaxes_when_too_few_picks():
    cache = FeatureCache()
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    rng = np.random.default_rng(2)
    for name in ["AAAUSD", "BBBUSD", "CCCUSD"]:
        close = np.cumprod(1 + rng.normal(0, 0.01, 80)) * 100
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
            index=idx,
        )
        for _, row in df.iterrows():
            cache._bars[name].append(row.to_dict())
    ranked = [("AAAUSD", 1.0), ("BBBUSD", 0.9), ("CCCUSD", 0.8)]
    picks = filter_uncorrelated_picks(ranked, cache, top_k=3, max_corr=0.01, config=CONFIG)
    assert len(picks) >= 2
