from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from universe import BLACKLIST, MIN_HOURLY_DOLLAR_VOLUME, select_universe
import universe as universe_module


def test_blacklist_contains_required_meme_symbols():
    required = {"PEPEUSD", "SHIBUSD", "BONKUSD", "FLOKIUSD", "WIFUSD", "SKLUSD"}
    assert required.issubset(BLACKLIST)


def test_select_universe_excludes_blacklist_even_with_high_volume(monkeypatch):
    symbols = ("BTCUSD", "ETHUSD", "PEPEUSD", "SHIBUSD", "BONKUSD", "FLOKIUSD", "WIFUSD", "SKLUSD")
    monkeypatch.setattr(universe_module, "KRAKEN_SAFE_LIST", symbols)

    def history_provider(_symbol, _start, _end):
        return pd.DataFrame({"close": [100.0, 100.0], "volume": [50_000.0, 50_000.0]})

    selected = select_universe(history_provider, pd.Timestamp("2025-10-01", tz="UTC"))
    for symbol in ("PEPEUSD", "SHIBUSD", "BONKUSD", "FLOKIUSD", "WIFUSD", "SKLUSD"):
        assert symbol not in selected


def test_select_universe_filters_low_hourly_dollar_volume(monkeypatch):
    monkeypatch.setattr(universe_module, "KRAKEN_SAFE_LIST", ("GOODUSD", "LOWUSD"))
    monkeypatch.setattr(universe_module, "BLACKLIST", frozenset())

    def history_provider(symbol, _start, _end):
        if symbol == "GOODUSD":
            return pd.DataFrame({"close": [200.0, 200.0], "volume": [20_000.0, 20_000.0]})
        return pd.DataFrame({"close": [10.0, 10.0], "volume": [20_000.0, 20_000.0]})

    selected = select_universe(history_provider, pd.Timestamp("2025-10-01", tz="UTC"))
    assert "GOODUSD" in selected
    assert "LOWUSD" not in selected
    assert MIN_HOURLY_DOLLAR_VOLUME == 250_000.0
