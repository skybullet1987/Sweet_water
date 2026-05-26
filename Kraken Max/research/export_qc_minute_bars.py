#!/usr/bin/env python3
"""
Export true sub-hour Kraken bars from QuantConnect Research (v8).

In QC Research:
    from export_qc_minute_bars import export_minute_history_qc
    bars = export_minute_history_qc(qb, ["BTCUSD","ETHUSD"], start, end, bar_minutes=15)
    bars.to_csv("kraken_15m_native.csv", index=False)

Uses Resolution.Minute History then consolidates to N-minute OHLCV (not hourly upsample).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workflow import consolidate_minute_ohlcv  # noqa: E402
from workflow import resample_bars_to_minutes  # noqa: E402


def export_minute_history_qc(
    qb,
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    *,
    bar_minutes: int = 15,
) -> pd.DataFrame:
    try:
        from AlgorithmImports import Market, Resolution
    except ImportError as exc:
        raise RuntimeError("QuantBook required") from exc

    symbols = [qb.AddCrypto(t, Resolution.Minute, Market.Kraken).Symbol for t in tickers]
    hist = qb.History(symbols, start, end, Resolution.Minute)
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    df = hist.reset_index()
    colmap = {str(c).lower(): c for c in df.columns}
    sym_col = colmap.get("symbol", "symbol")
    time_col = colmap.get("time", colmap.get("endtime", "time"))
    out = pd.DataFrame(
        {
            "symbol": df[sym_col].astype(str).str.replace("/", "", regex=False),
            "timestamp": pd.to_datetime(df[time_col], utc=True),
            "open": df[colmap["open"]].astype(float),
            "high": df[colmap["high"]].astype(float),
            "low": df[colmap["low"]].astype(float),
            "close": df[colmap["close"]].astype(float),
            "volume": df[colmap["volume"]].astype(float),
        }
    )
    if int(bar_minutes) >= 60:
        return out.sort_values(["symbol", "timestamp"])
    return consolidate_minute_ohlcv(out, bar_minutes=bar_minutes)


def upsample_fallback(hourly_csv: Path, out_csv: Path, bar_minutes: int = 15) -> pd.DataFrame:
    """Fallback when minute history unavailable — labeled synthetic in filename."""
    df = pd.read_csv(hourly_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = resample_bars_to_minutes(df, bar_minutes=bar_minutes)
    out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Upsample hourly CSV to 15m (fallback only)")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("kraken_15m_upsampled.csv"))
    p.add_argument("--minutes", type=int, default=15)
    args = p.parse_args()
    upsample_fallback(args.csv, args.out, args.minutes)
    print(f"wrote {args.out} (synthetic upsample — prefer export_minute_history_qc in QC)")
