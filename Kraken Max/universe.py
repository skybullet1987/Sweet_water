from __future__ import annotations

from datetime import timedelta
from typing import Callable

import pandas as pd

from config import CONFIG

# Kraken Canada: BTC, ETH, LTC, BCH have no CAD net-purchase limits.
CANADA_UNLIMITED = frozenset({"BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"})

REFERENCE_SYMBOLS = ("BTCUSD", "ETHUSD")

BLACKLIST: frozenset[str] = frozenset({
    "PEPEUSD",
    "SHIBUSD",
    "BONKUSD",
    "FLOKIUSD",
    "WIFUSD",
    "BANANAS31USD",
    "CHILLHOUSEUSD",
    "SKLUSD",
})

# Liquid Kraken pairs — unlimited coins first, then high-liquidity alts.
KRAKEN_MAX_UNIVERSE = (
    "BTCUSD",
    "ETHUSD",
    "LTCUSD",
    "BCHUSD",
    "SOLUSD",
    "XRPUSD",
    "LINKUSD",
    "ADAUSD",
    "DOTUSD",
    "AVAXUSD",
    "ATOMUSD",
    "NEARUSD",
    "ARBUSD",
    "OPUSD",
    "INJUSD",
    "SUIUSD",
    "APTUSD",
    "SEIUSD",
    "TIAUSD",
    "RNDRUSD",
    "FETUSD",
    "UNIUSD",
    "AAVEUSD",
    "MKRUSD",
    "CRVUSD",
    "SNXUSD",
    "COMPUSD",
    "LDOUSD",
    "IMXUSD",
    "FILUSD",
    "ETCUSD",
    "ALGOUSD",
    "XLMUSD",
    "TRXUSD",
    "MANAUSD",
    "SANDUSD",
    "AXSUSD",
    "APEUSD",
    "DYDXUSD",
    "ENSUSD",
    "GRTUSD",
    "FLOWUSD",
    "RUNEUSD",
    "KSMUSD",
    "MINAUSD",
    "ZECUSD",
    "DASHUSD",
)

MIN_HOURLY_DOLLAR_VOLUME = 150_000.0
DEFAULT_UNIVERSE_SIZE = 24


def _median_dollar_volume(frame: pd.DataFrame) -> float:
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    return float((close * volume).median())


def _priority_boost(symbol: str) -> float:
    return 1e12 if symbol in CANADA_UNLIMITED else 0.0


def select_universe(
    history_provider: Callable[[str, object, object], pd.DataFrame],
    asof_date,
) -> list[str]:
    start = asof_date - timedelta(days=21)
    rows: list[tuple[str, float]] = []
    for symbol in KRAKEN_MAX_UNIVERSE:
        if symbol in BLACKLIST:
            continue
        frame = history_provider(symbol, start, asof_date)
        if frame is None or frame.empty:
            rows.append((symbol, 0.0))
            continue
        rows.append((symbol, _median_dollar_volume(frame)))

    liquid = [(s, v) for s, v in rows if v >= MIN_HOURLY_DOLLAR_VOLUME]
    pool = liquid if liquid else rows
    ranked = sorted(pool, key=lambda x: (_priority_boost(x[0]), x[1]), reverse=True)
    limit = max(1, min(int(CONFIG.universe_size), len(ranked)))
    return [s for s, _ in ranked[:limit]]
