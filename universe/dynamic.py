from __future__ import annotations

from datetime import timedelta
from typing import Callable

import pandas as pd

KRAKEN_SAFE_LIST = (
    "BTCUSD",
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


def select_universe(history_provider: Callable[[str, object, object], pd.DataFrame], asof_date) -> list[str]:
    start = asof_date - timedelta(days=30)
    liquidity: list[tuple[str, float]] = []
    for symbol in KRAKEN_SAFE_LIST:
        hist = history_provider(symbol, start, asof_date)
        if hist is None or hist.empty:
            continue
        close = hist["close"].astype(float)
        volume = hist["volume"].astype(float)
        med_dollar_volume = float((close * volume).median())
        liquidity.append((symbol, med_dollar_volume))

    ranked = [s for s, _ in sorted(liquidity, key=lambda x: x[1], reverse=True)]
    selected = ranked[:8]
    if "BTCUSD" not in selected:
        selected = ["BTCUSD"] + [s for s in selected if s != "BTCUSD"]
    return selected[:8]
