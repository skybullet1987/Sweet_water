from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from data import (  # noqa: E402
    compute_btc_dominance,
    load_fear_greed_csv,
    load_funding_csv,
    SentimentDataHub,
)
from core import load_optimized_ensemble_weights  # noqa: E402
from execution import _USE_PRO  # noqa: E402
from data import compute_sentiment, merge_external_sentiment  # noqa: E402
from workflow import prepare_hourly_panel, walk_forward_optimize  # noqa: E402


def _multi_symbol_bars(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
    rows = []
    rng = np.random.default_rng(42)
    for sym, drift in [("BTCUSD", 0.0003), ("ETHUSD", 0.0002), ("SOLUSD", 0.0005)]:
        close = 100.0
        for ts in idx:
            close *= 1 + drift + rng.normal(0, 0.008)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts.isoformat(),
                    "open": close * 0.999,
                    "high": close * 1.001,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_fear_greed_csv_loads():
    df = load_fear_greed_csv()
    assert not df.empty
    assert "value" in df.columns


def test_funding_csv_loads():
    df = load_funding_csv()
    assert not df.empty


def test_btc_dominance_from_features():
    m = {
        "BTCUSD": {"close": 100, "volume_24h": 5000},
        "ETHUSD": {"close": 50, "volume_24h": 1000},
    }
    dom = compute_btc_dominance(m)
    assert dom > 0.5


def test_merge_external_sentiment():
    proxy = compute_sentiment(
        btc_features={"mom_7d": 0.1, "mom_21d": 0.1, "trend_quality": 0.02},
        eth_features={"mom_7d": 0.05},
        breadth=0.6,
        median_rv=0.4,
    )
    from data import ExternalSentiment

    ext = ExternalSentiment(fear_greed_index=25.0, fear_greed_normalized=0.25, btc_dominance=0.7, funding_stress=0.3)
    ext.source_fg = "csv_fg"
    merged = merge_external_sentiment(proxy, ext)
    assert merged.fear_greed == 0.25
    assert merged.fear_greed_index == 25.0


def test_ensemble_weights_json():
    w = load_optimized_ensemble_weights()
    assert "w_momentum" in w


def test_walk_forward_optimize_runs():
    bars = _multi_symbol_bars(600)
    result = walk_forward_optimize(bars, n_folds=2)
    assert result.oos_trades >= 0
    assert "w_momentum" in result.best_weights


def test_execution_bridge_qc_runtime():
    assert isinstance(_USE_PRO, bool)


def test_prepare_hourly_panel():
    panel = prepare_hourly_panel(_multi_symbol_bars(200))
    assert "BTCUSD" in panel
    assert "ret24" in panel["BTCUSD"].columns
