#!/usr/bin/env python3
"""Download Fear & Greed + funding proxies into Kraken Max/data/*.csv for backtests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def fetch_fear_greed() -> pd.DataFrame:
    try:
        import urllib.request

        url = "https://api.alternative.me/fng/?limit=365&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "KrakenMax/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        rows = []
        for item in payload.get("data", []):
            ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
            rows.append({"date": ts.strftime("%Y-%m-%d"), "value": float(item["value"])})
        return pd.DataFrame(rows).sort_values("date")
    except Exception as exc:
        print(f"fear_greed fetch failed: {exc}")
        return pd.DataFrame(columns=["date", "value"])


def fetch_funding_placeholder() -> pd.DataFrame:
    """Placeholder funding CSV — replace with exchange API export for production."""
    dates = pd.date_range("2024-01-01", "2025-06-01", freq="MS", tz="UTC")
    rows = []
    for d in dates:
        rows.append({"date": d.strftime("%Y-%m-%d"), "symbol": "BTCUSD", "funding_rate": 0.0003})
        rows.append({"date": d.strftime("%Y-%m-%d"), "symbol": "ETHUSD", "funding_rate": 0.0002})
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fg = fetch_fear_greed()
    if not fg.empty:
        out = DATA / "fear_greed_history.csv"
        fg.to_csv(out, index=False)
        print(f"wrote {out} rows={len(fg)}")
    fund = fetch_funding_placeholder()
    fout = DATA / "funding_rates.csv"
    fund.to_csv(fout, index=False)
    print(f"wrote {fout} rows={len(fund)}")


if __name__ == "__main__":
    main()
