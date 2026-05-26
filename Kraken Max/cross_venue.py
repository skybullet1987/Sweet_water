from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import CONFIG, KrakenMaxConfig

_KRAKEN_MAX = Path(__file__).resolve().parent

# Map Kraken tickers to Binance spot symbols in lead CSV
_DEFAULT_MAP = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "LINKUSD": "LINKUSDT",
    "AVAXUSD": "AVAXUSDT",
    "ADAUSD": "ADAUSDT",
    "DOTUSD": "DOTUSDT",
    "XRPUSD": "XRPUSDT",
}


@dataclass
class LeadSignal:
    symbol: str
    ret_1h: float
    z_score: float
    boost: float


class CrossVenueLead:
    """
    Optional Binance/Coinbase spot lead (v6).
    Loads CSV with columns: symbol, timestamp, close (or ret_1h).
    Execution remains on Kraken — lead only nudges entry scores.
    """

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self._panel: dict[str, pd.Series] = {}
        self._load_csv()

    def _load_csv(self) -> None:
        path = Path(self.config.cross_venue_lead_csv)
        if not path.is_absolute():
            path = _KRAKEN_MAX / path
        if not path.exists():
            return
        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values(["symbol", "timestamp"])
            if "ret_1h" not in df.columns and "close" in df.columns:
                df["ret_1h"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)
            for sym, grp in df.groupby("symbol"):
                s = grp.set_index("timestamp")["ret_1h"].astype(float)
                self._panel[str(sym)] = s
        except Exception:
            self._panel = {}

    @property
    def loaded(self) -> bool:
        return bool(self._panel)

    def _kraken_to_lead_symbol(self, kraken_ticker: str) -> str:
        mapping = dict(_DEFAULT_MAP)
        return mapping.get(kraken_ticker, kraken_ticker.replace("USD", "USDT"))

    def lead_at(self, kraken_ticker: str, when) -> LeadSignal | None:
        if not self._panel or not bool(self.config.use_cross_venue_lead):
            return None
        lead_sym = self._kraken_to_lead_symbol(kraken_ticker)
        series = self._panel.get(lead_sym)
        if series is None or series.empty:
            return None
        try:
            ts = pd.Timestamp(when, tz="UTC")
            idx = series.index.get_indexer([ts], method="pad")
            if idx[0] < 0:
                return None
            ret = float(series.iloc[idx[0]])
        except Exception:
            ret = float(series.iloc[-1])
        window = series.tail(max(24, self.config.cross_venue_z_window))
        mu = float(window.mean())
        sd = float(window.std()) or 1e-6
        z = (ret - mu) / sd
        cap = float(self.config.cross_venue_max_boost)
        boost = max(-cap, min(cap, z * float(self.config.cross_venue_boost_per_z)))
        return LeadSignal(symbol=kraken_ticker, ret_1h=ret, z_score=z, boost=boost)

    def score_adjustment(self, kraken_ticker: str, when) -> float:
        sig = self.lead_at(kraken_ticker, when)
        return float(sig.boost) if sig else 0.0

    def aggregate_boost(self, tickers: list[str], when) -> float:
        if not tickers:
            return 0.0
        boosts = [self.score_adjustment(t, when) for t in tickers]
        return float(sum(boosts) / len(boosts))
