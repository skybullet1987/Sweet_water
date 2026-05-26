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

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

_KM_ROOT = Path(__file__).resolve().parents[1]
if str(_KM_ROOT) not in sys.path:
    sys.path.insert(0, str(_KM_ROOT))


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


def pull_kraken_history_qc(
    qb,
    tickers: Iterable[str],
    start: datetime,
    end: datetime,
    *,
    bar_minutes: int = 60,
) -> pd.DataFrame:
    """Pull Kraken crypto history inside QuantConnect Research (hourly or minute→consolidate)."""
    try:
        from AlgorithmImports import Market, Resolution
    except ImportError as exc:
        raise RuntimeError("QuantBook / AlgorithmImports required") from exc

    symbols = []
    res = Resolution.Hour if int(bar_minutes) >= 60 else Resolution.Minute
    for t in tickers:
        symbols.append(qb.AddCrypto(t, res, Market.Kraken).Symbol)
    hist = qb.History(symbols, start, end, res)
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
    out = out.dropna().sort_values(["symbol", "timestamp"])
    if int(bar_minutes) < 60 and not out.empty:
        from workflow import resample_bars_to_minutes

        out = resample_bars_to_minutes(out, bar_minutes=int(bar_minutes))
    return out


def export_15m_from_hourly_csv(csv_path, out_path, bar_minutes: int = 15) -> pd.DataFrame:
    """Local helper: upsample hourly export to N-minute bars for 15m walk-forward."""
    from pathlib import Path

    from workflow import resample_bars_to_minutes

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    out = resample_bars_to_minutes(df, bar_minutes=bar_minutes)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate / merge local hourly CSV exports")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("kraken_hourly_merged.csv"))
    parser.add_argument("--to-15m", action="store_true", help="Upsample hourly CSV to 15m bars")
    args = parser.parse_args()
    if args.to_15m:
        out15 = args.out if "15m" in str(args.out) else args.out.with_name(args.out.stem + "_15m.csv")
        df = export_15m_from_hourly_csv(args.csv, out15)
        print(f"15m rows={len(df)} -> {out15}")
        return
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
