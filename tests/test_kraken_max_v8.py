from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from workflow import AutoRevalidator  # noqa: E402
from workflow import consolidate_minute_ohlcv  # noqa: E402
from config import CONFIG  # noqa: E402
from kraken_ops import DigestBundle, build_html_digest, build_text_digest  # noqa: E402
from workflow import (  # noqa: E402
    build_regime_index,
    optimize_regime_weights,
    regime_label_from_btc_row,
    save_regime_weights,
)


def _hourly_bars(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    rng = np.random.default_rng(3)
    for sym in ("BTCUSD", "ETHUSD"):
        c = 100.0
        for ts in idx:
            c *= 1 + rng.normal(0.0003, 0.006)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": c,
                    "high": c * 1.002,
                    "low": c * 0.998,
                    "close": c,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_v8_config():
    assert not CONFIG.enable_auto_revalidation  # off by default — heavy on QC backtests
    assert not CONFIG.subscribe_all_universe_on_init
    assert len(CONFIG.seed_subscribe_symbols) >= 6
    assert CONFIG.enable_dashboard_digest
    assert not CONFIG.use_regime_wf_weights  # static regime weights until WF merge is implemented


def test_regime_label():
    row = pd.Series({"ret168": 0.05, "ema20": 102, "ema50": 100, "rv": 0.3})
    assert regime_label_from_btc_row(row) == "bull"


def test_consolidate_minute():
    idx = pd.date_range("2024-01-01", periods=60, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSD"] * 60,
            "timestamp": idx,
            "open": np.linspace(100, 101, 60),
            "high": np.linspace(100.1, 101.1, 60),
            "low": np.linspace(99.9, 100.9, 60),
            "close": np.linspace(100, 101, 60),
            "volume": np.ones(60) * 10,
        }
    )
    out = consolidate_minute_ohlcv(df, 15)
    assert len(out) == 4
    assert out["symbol"].iloc[0] == "BTCUSD"


def test_optimize_regime_weights_smoke():
    bars = _hourly_bars(300)
    cfg = replace(CONFIG, regime_wf_min_bars=30)
    weights = optimize_regime_weights(bars, config=cfg)
    assert "bull" in weights
    assert abs(sum(weights["bull"].values()) - 1.0) < 0.05


def test_auto_revalidation_smoke(tmp_path):
    bars = _hourly_bars(500)
    cfg = replace(
        CONFIG,
        validation_min_sharpe=-99,
        validation_max_drawdown=-0.99,
        validation_min_trades=1,
        validation_min_win_rate=0.0,
        regime_wf_min_bars=30,
        auto_save_ensemble_weights=False,
        validation_report_path="validation_report_test.json",
        regime_weights_path="regime_weights_test.json",
        ensemble_weights_path="ensemble_weights_test.json",
    )
    rev = AutoRevalidator(cfg)
    result = rev.run(bars, algo=None, n_folds=2)
    assert "oos_sharpe" in result
    assert "validation_passed" in result


def test_dashboard_digest():
    bundle = DigestBundle(
        telemetry={"equity": 1500, "regime": "bull", "micro_regime": "x", "deployment_cap": 0.9, "fill_rate": 0.8, "avg_slippage_bps": 12, "live_sharpe": 0.5, "baseline_sharpe": 0.6, "drift_ratio": 0.83},
        scorecard={"live_sharpe": 0.4, "win_rate": 0.55, "profit_factor": 1.2, "n_trades": 10, "drawdown": -0.05},
        validation={"passed": True, "oos_sharpe": 0.3, "oos_max_drawdown": -0.2},
        revalidation={"oos_sharpe": 0.35},
    )
    text = build_text_digest(bundle)
    html = build_html_digest(bundle)
    assert "Kraken Max" in text
    assert "<html>" in html
