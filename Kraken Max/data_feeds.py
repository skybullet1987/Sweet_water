from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import CONFIG, KrakenMaxConfig

try:  # pragma: no cover
    from AlgorithmImports import (
        FearGreedIndex,
        PythonData,
        Resolution,
        SubscriptionTransportMedium,
    )
    HAS_QC = True
except ImportError:  # pragma: no cover
    HAS_QC = False

    class PythonData:  # type: ignore
        pass

    class Resolution:
        Daily = "Daily"

    class SubscriptionTransportMedium:
        LocalFile = "LocalFile"
        RemoteFile = "RemoteFile"


DATA_DIR = Path(__file__).resolve().parent / "data"
FEAR_GREED_CSV = DATA_DIR / "fear_greed_history.csv"
FUNDING_CSV = DATA_DIR / "funding_rates.csv"
OPEN_INTEREST_CSV = DATA_DIR / "open_interest.csv"


@dataclass
class ExternalSentiment:
    fear_greed_index: float = 50.0  # 0-100 CNN-style
    fear_greed_normalized: float = 0.5  # 0-1
    btc_dominance: float = 0.5
    funding_rate_btc: float = 0.0
    funding_rate_eth: float = 0.0
    funding_stress: float = 0.0
    open_interest_btc: float = 0.0
    open_interest_stress: float = 0.0
    source_fg: str = "proxy"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def load_fear_greed_csv(path: Path | None = None) -> pd.DataFrame:
    target = path or FEAR_GREED_CSV
    if not target.exists():
        return pd.DataFrame(columns=["date", "value"])
    df = pd.read_csv(target)
    if "value" not in df.columns and "fear_greed_score" in df.columns:
        df = df.rename(columns={"fear_greed_score": "value"})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date")


def load_open_interest_csv(path: Path | None = None) -> pd.DataFrame:
    target = path or OPEN_INTEREST_CSV
    if not target.exists():
        return pd.DataFrame(columns=["date", "symbol", "open_interest"])
    df = pd.read_csv(target)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    return df.dropna(subset=["date", "symbol", "open_interest"])


def load_funding_csv(path: Path | None = None) -> pd.DataFrame:
    target = path or FUNDING_CSV
    if not target.exists():
        return pd.DataFrame(columns=["date", "symbol", "funding_rate"])
    df = pd.read_csv(target)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df.dropna(subset=["date", "symbol", "funding_rate"])


def compute_btc_dominance(
    feature_map: dict[str, dict[str, float]],
    *,
    btc_ticker: str = "BTCUSD",
) -> float:
    """Cap-weight proxy: BTC dollar vol share in universe."""
    weights: list[float] = []
    btc_w = 0.0
    for ticker, feats in feature_map.items():
        close = float(feats.get("close", 0.0))
        vol = float(feats.get("volume_24h", feats.get("volume_surge", 0.0)))
        w = max(close * max(vol, 1.0), 1.0)
        weights.append(w)
        if ticker == btc_ticker:
            btc_w = w
    total = sum(weights)
    if total <= 0:
        return 0.5
    return _clamp01(btc_w / total)


if HAS_QC:

    class AlternativeFearGreed(PythonData):
        """Fallback F&G from bundled CSV (alternative.me format)."""

        def GetSource(self, config, date, isLiveMode):
            path = str(FEAR_GREED_CSV)
            return SubscriptionDataSource(path, SubscriptionTransportMedium.LocalFile, FileFormat.CSV)

        def Reader(self, config, line, date, isLiveMode):
            if not (line and line.strip()) or line.startswith("date"):
                return None
            parts = line.split(",")
            if len(parts) < 2:
                return None
            item = AlternativeFearGreed()
            item.Symbol = config.Symbol
            item.Time = datetime.strptime(parts[0][:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            item.EndTime = item.Time + timedelta(days=1)
            item.Value = float(parts[1])
            return item

class SentimentDataHub:
    """Unified real + proxy sentiment for Kraken Max v3."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.fg_symbol: Any = None
        self.funding_symbols: dict[str, Any] = {}
        self._fg_history = load_fear_greed_csv()
        self._funding_history = load_funding_csv()
        self._oi_history = load_open_interest_csv()
        self.last: ExternalSentiment = ExternalSentiment()

    def initialize_algorithm(self, algo) -> None:
        if HAS_QC:
            try:
                self.fg_symbol = algo.AddData(FearGreedIndex, "FG").Symbol
                algo.Debug("KRAKEN_MAX v3 FearGreedIndex subscribed")
            except Exception as exc:
                algo.Debug(f"KRAKEN_MAX FG QC dataset unavailable: {exc}")
                if FEAR_GREED_CSV.exists():
                    try:
                        self.fg_symbol = algo.AddData(AlternativeFearGreed, "FG_ALT").Symbol
                    except Exception as exc2:
                        algo.Debug(f"KRAKEN_MAX FG CSV fallback failed: {exc2}")
        self._fg_history = load_fear_greed_csv()

    def update_from_slice(self, algo, data, feature_map: dict[str, dict[str, float]]) -> ExternalSentiment:
        fg_val = None
        source = "proxy"
        if self.fg_symbol is not None and self.fg_symbol in data:
            try:
                fg_val = float(data[self.fg_symbol].Value)
                source = "qc_fg"
            except Exception:
                pass
        if fg_val is None and not self._fg_history.empty:
            ts = pd.Timestamp(getattr(algo, "Time", datetime.now(timezone.utc)), tz="UTC")
            row = self._fg_history[self._fg_history["date"] <= ts]
            if not row.empty:
                fg_val = float(row.iloc[-1]["value"])
                source = "csv_fg"

        if fg_val is None:
            breadth = sum(1 for f in feature_map.values() if float(f.get("mom_7d", 0)) > 0) / max(len(feature_map), 1)
            fg_val = 50.0 + 30.0 * (breadth - 0.5)
            source = "proxy"

        dom = compute_btc_dominance(feature_map)
        fund_btc, fund_eth = self._latest_funding(algo)
        stress = _clamp01(max(0.0, fund_btc) * 200 + max(0.0, fund_eth) * 200)
        oi_btc, oi_stress = self._latest_open_interest(algo)

        self.last = ExternalSentiment(
            fear_greed_index=float(fg_val),
            fear_greed_normalized=_clamp01(float(fg_val) / 100.0),
            btc_dominance=dom,
            funding_rate_btc=fund_btc,
            funding_rate_eth=fund_eth,
            funding_stress=max(stress, oi_stress * 0.5),
            open_interest_btc=oi_btc,
            open_interest_stress=oi_stress,
            source_fg=source,
        )
        return self.last

    def _latest_funding(self, algo) -> tuple[float, float]:
        if not self._funding_history.empty:
            ts = pd.Timestamp(getattr(algo, "Time", datetime.now(timezone.utc)), tz="UTC")
            sub = self._funding_history[self._funding_history["date"] <= ts]
            if not sub.empty:
                btc = sub[sub["symbol"].str.upper().str.contains("BTC", na=False)]
                eth = sub[sub["symbol"].str.upper().str.contains("ETH", na=False)]
                b = float(btc.iloc[-1]["funding_rate"]) if not btc.empty else 0.0
                e = float(eth.iloc[-1]["funding_rate"]) if not eth.empty else 0.0
                return b, e
        return 0.0, 0.0

    def _latest_open_interest(self, algo) -> tuple[float, float]:
        if self._oi_history.empty:
            return 0.0, 0.0
        ts = pd.Timestamp(getattr(algo, "Time", datetime.now(timezone.utc)), tz="UTC")
        sub = self._oi_history[self._oi_history["date"] <= ts]
        btc = sub[sub["symbol"].str.upper().str.contains("BTC", na=False)]
        if btc.empty:
            return 0.0, 0.0
        oi = float(btc.iloc[-1]["open_interest"])
        med = float(btc["open_interest"].median()) if len(btc) > 3 else oi
        stress = _clamp01((oi / max(med, 1e-9)) - 1.0) if med > 0 else 0.0
        return oi, stress

    def to_context(self) -> dict[str, float]:
        return {
            "fear_greed": self.last.fear_greed_normalized,
            "btc_dominance": self.last.btc_dominance,
            "funding_stress": self.last.funding_stress,
            "funding_btc": self.last.funding_rate_btc,
            "open_interest_stress": self.last.open_interest_stress,
        }
