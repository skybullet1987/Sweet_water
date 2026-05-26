#!/usr/bin/env python3
"""Download Fear & Greed, Binance funding, and open interest into Kraken Max/data/*.csv."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

FUNDING_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def fetch_fear_greed() -> pd.DataFrame:
    try:
        import urllib.request

        url = "https://api.alternative.me/fng/?limit=365&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "KrakenMax/4.0"})
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


def fetch_binance_funding(symbols: tuple[str, ...] = FUNDING_SYMBOLS) -> pd.DataFrame:
    import urllib.request

    rows = []
    for sym in symbols:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=200"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KrakenMax/4.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            for item in data:
                ts = datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=timezone.utc)
                rate = float(item["fundingRate"])
                spot_sym = sym.replace("USDT", "USD")
                rows.append({"date": ts.strftime("%Y-%m-%d"), "symbol": spot_sym, "funding_rate": rate})
        except Exception as exc:
            print(f"funding {sym} failed: {exc}")
    return pd.DataFrame(rows).drop_duplicates(subset=["date", "symbol"]).sort_values("date")


def fetch_binance_open_interest(symbols: tuple[str, ...] = FUNDING_SYMBOLS) -> pd.DataFrame:
    import urllib.request

    rows = []
    for sym in symbols:
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "KrakenMax/4.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                item = json.loads(resp.read().decode())
            ts = datetime.now(timezone.utc)
            spot_sym = sym.replace("USDT", "USD")
            rows.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "symbol": spot_sym,
                    "open_interest": float(item.get("openInterest", 0.0)),
                }
            )
        except Exception as exc:
            print(f"oi {sym} failed: {exc}")
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    fg = fetch_fear_greed()
    if not fg.empty:
        out = DATA / "fear_greed_history.csv"
        fg.to_csv(out, index=False)
        print(f"wrote {out} rows={len(fg)}")
    fund = fetch_binance_funding()
    if fund.empty:
        fund = pd.DataFrame(
            [
                {"date": "2025-01-01", "symbol": "BTCUSD", "funding_rate": 0.0001},
                {"date": "2025-01-01", "symbol": "ETHUSD", "funding_rate": 0.0001},
            ]
        )
    fout = DATA / "funding_rates.csv"
    fund.to_csv(fout, index=False)
    print(f"wrote {fout} rows={len(fund)}")
    oi = fetch_binance_open_interest()
    oout = DATA / "open_interest.csv"
    oi.to_csv(oout, index=False)
    print(f"wrote {oout} rows={len(oi)}")


if __name__ == "__main__":
    main()
