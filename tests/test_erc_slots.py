from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from core import FeatureCache  # noqa: E402
from risk import allocate_erc_notionals  # noqa: E402


def _seed(cache: FeatureCache, ticker: str, n: int = 80) -> None:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(hash(ticker) % 2**32)
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100.0
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=idx,
    )
    for _, row in df.iterrows():
        cache._bars[ticker].append(row.to_dict())


def test_allocate_erc_gives_every_target_a_slot():
    cache = FeatureCache()
    for t in ("BTCUSD", "ETHUSD", "SOLUSD"):
        _seed(cache, t)
    targets = ["BTCUSD", "ETHUSD", "SOLUSD"]
    slots = allocate_erc_notionals(targets, cache, 1000.0, 0.9, config=CONFIG)
    for t in targets:
        assert slots.get(t, 0) >= CONFIG.min_position_floor_usd
