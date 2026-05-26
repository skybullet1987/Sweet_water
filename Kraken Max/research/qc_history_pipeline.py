#!/usr/bin/env python3
"""
Multi-asset Kraken history export for walk-forward.

QuantConnect Research:
    qb = QuantBook()
    from qc_history_pipeline import pull_kraken_history_qc
    bars = pull_kraken_history_qc(qb, tickers, start, end)
    bars.to_csv('kraken_hourly.csv', index=False)

Local: use --csv from an existing export.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd


DEFAULT_TICKERS = (
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "LINKUSD",
    "AVAXUSD",
    "ADAUSD",
    "DOTUSD",
    "XRPUSD",
)


def pull_kraken_history_qc(qb, tickers: Iterable[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Pull hourly Kraken crypto history inside QuantConnect Research."""
    try:
        from AlgorithmImports import Market, Resolution
    except ImportError as exc:
        raise RuntimeError("QuantBook / AlgorithmImports required") from exc

    symbols = []
    for t in tickers:
        symbols.append(qb.AddCrypto(t, Resolution.Hour, Market.Kraken).Symbol)
    hist = qb.History(symbols, start, end, Resolution.Hour)
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    df = hist.reset_index()
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in {"time", "endtime", "date"}:
            rename[c] = "timestamp"
        elif cl == "symbol":
            rename[c] = "symbol"
    df = df.rename(columns=rename)
    if "symbol" not in df.columns and "symbol" in df.index.names:
        df = df.reset_index()
    colmap = {c.lower(): c for c in df.columns}
    out = pd.DataFrame(
        {
            "symbol": df[colmap.get("symbol", "symbol")].astype(str).str.replace("/", "", regex=False),
            "timestamp": pd.to_datetime(df[colmap.get("timestamp", "time")], utc=True),
            "open": df[colmap.get("open", "open")].astype(float),
            "high": df[colmap.get("high", "high")].astype(float),
            "low": df[colmap.get("low", "low")].astype(float),
            "close": df[colmap.get("close", "close")].astype(float),
            "volume": df[colmap.get("volume", "volume")].astype(float),
        }
    )
    return out.dropna().sort_values(["symbol", "timestamp"])


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate / merge local hourly CSV exports")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("kraken_hourly_merged.csv"))
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise SystemExit(f"missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"symbols={df['symbol'].nunique()} rows={len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
