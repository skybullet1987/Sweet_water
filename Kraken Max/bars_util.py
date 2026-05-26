from __future__ import annotations

import pandas as pd


def consolidate_minute_ohlcv(df: pd.DataFrame, bar_minutes: int = 15) -> pd.DataFrame:
    """Consolidate minute OHLCV to N-minute bars per symbol (v8 native sub-hour)."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    rows: list[dict] = []
    rule = f"{int(bar_minutes)}min"
    for sym, grp in data.groupby("symbol"):
        g = grp.set_index("timestamp").sort_index()
        ohlc = pd.DataFrame(
            {
                "open": g["open"].resample(rule).first(),
                "high": g["high"].resample(rule).max(),
                "low": g["low"].resample(rule).min(),
                "close": g["close"].resample(rule).last(),
                "volume": g["volume"].resample(rule).sum(),
            }
        ).dropna(subset=["close"])
        for ts, row in ohlc.iterrows():
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
    return pd.DataFrame(rows)
