from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from core import history_bars_from_qc, select_universe  # noqa: E402


def test_history_bars_from_qc_without_volume_column():
    hist = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )
    out = history_bars_from_qc(hist)
    assert len(out) == 2
    assert "volume" in out.columns
    assert out["volume"].iloc[0] == 1000.0


def test_select_universe_survives_missing_volume_in_history():
    def provider(symbol: str, start, end) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
            }
        )

    tickers = select_universe(provider, pd.Timestamp("2024-06-01", tz="UTC"), candidates=("BTCUSD",))
    assert tickers == ["BTCUSD"]
