from __future__ import annotations

from datetime import timedelta
from typing import Callable

import pandas as pd

REFERENCE_SYMBOLS = ("BTCUSD",)

KRAKEN_SAFE_LIST = (
    "ETHUSD",
    "SOLUSD",
    "XRPUSD",
    "ADAUSD",
    "LTCUSD",
    "LINKUSD",
    "DOTUSD",
    "AVAXUSD",
    "BCHUSD",
    "DOGEUSD",
)


def _median_dollar_volume(frame: pd.DataFrame) -> float:
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    return float((close * volume).median())


def select_universe(history_provider: Callable[[str, object, object], pd.DataFrame], asof_date) -> list[str]:
    start = asof_date - timedelta(days=30)
    liquidity: list[tuple[str, float]] = []
    for symbol in KRAKEN_SAFE_LIST:
        frame = history_provider(symbol, start, asof_date)
        if frame is None or frame.empty:
            liquidity.append((symbol, 0.0))
            continue
        liquidity.append((symbol, _median_dollar_volume(frame)))
    ranked = [s for s, _ in sorted(liquidity, key=lambda x: x[1], reverse=True)]
    return ranked[:8]
