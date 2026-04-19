from __future__ import annotations

from datetime import timedelta
from typing import Callable

import pandas as pd
from config import CONFIG

REFERENCE_SYMBOLS = ("BTCUSD",)
DEFAULT_UNIVERSE_SIZE = 80

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
    "ATOMUSD",
    "ALGOUSD",
    "XLMUSD",
    "TRXUSD",
    "BCHUSD",
    "AAVEUSD",
    "MKRUSD",
    "UNIUSD",
    "SUSHIUSD",
    "COMPUSD",
    "SNXUSD",
    "CRVUSD",
    "YFIUSD",
    "1INCHUSD",
    "GRTUSD",
    "FTMUSD",
    "IMXUSD",
    "APEUSD",
    "OPUSD",
    "ARBUSD",
    "NEARUSD",
    "MANAUSD",
    "SANDUSD",
    "ENJUSD",
    "BATUSD",
    "LDOUSD",
    "LRCUSD",
    "KSMUSD",
    "ETCUSD",
    "FILUSD",
    "EOSUSD",
    "XTZUSD",
    "CHZUSD",
    "ANKRUSD",
    "OMGUSD",
    "OXTUSD",
    "ZRXUSD",
    "RENUSD",
    "KNCUSD",
    "BALUSD",
    "RUNEUSD",
    "FLOWUSD",
    "AXSUSD",
    "AUDIOUSD",
    "CELOUSD",
    "MASKUSD",
    "ENSUSD",
    "DYDXUSD",
    "PEPEUSD",
    "BONKUSD",
    "INJUSD",
    "SEIUSD",
    "JUPUSD",
    "PYTHUSD",
    "WIFUSD",
    "FETUSD",
    "RNDRUSD",
    "TIAUSD",
    "SUIUSD",
    "APTUSD",
    "STRKUSD",
    "WLDUSD",
    "ARUSD",
    "LPTUSD",
    "PHAUSD",
    "CTSIUSD",
    "ICXUSD",
    "ZECUSD",
    "DASHUSD",
    "STORJUSD",
    "SKLUSD",
    "MINAUSD",
    "JASMYUSD",
    "GMTUSD",
    "BLURUSD",
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
    configured = int(getattr(CONFIG, "universe_size", DEFAULT_UNIVERSE_SIZE) or DEFAULT_UNIVERSE_SIZE)
    limit = max(1, min(configured, len(ranked)))
    return ranked[:limit]
