from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from core import filter_uncorrelated_picks, return_correlation  # noqa: E402
from core import AlphaEnsemble  # noqa: E402
from core import FeatureCache, btc_beta_vs, compute_bar_features, cross_section_ranks  # noqa: E402
from kraken_ml import MLScorer, load_ml_weights  # noqa: E402
from kraken_ml import MLTrainer  # noqa: E402
from regime import RegimeEngine  # noqa: E402
from risk import PositionRisk, should_exit  # noqa: E402
from core import build_scalper_features, evaluate_scalper_entry, evaluate_scalper_exit  # noqa: E402
from data import adjust_deployment_cap, compute_sentiment  # noqa: E402
from core import AggressiveSizer, can_afford  # noqa: E402
from core import CANADA_UNLIMITED, BLACKLIST, select_universe  # noqa: E402


def _synthetic_ohlcv(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    close = np.cumprod(1 + rng.normal(0.0005, 0.01, rows)) * 100.0
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(rows, 1000.0),
        },
        index=idx,
    )


def test_v2_config_flags():
    assert CONFIG.use_limit_orders is False
    assert CONFIG.momentum_force_market is True
    assert CONFIG.enable_scalper is False
    assert CONFIG.rank_entries_when_empty is True
    assert not CONFIG.use_calibrated_costs
    assert not CONFIG.use_advanced_regime
    assert CONFIG.bph() == 1
    assert CONFIG.entry_score_threshold <= 0.35
    assert CONFIG.rebalance_hours <= 4
    assert CONFIG.max_orders_per_day >= 48
    assert CONFIG.max_pairwise_corr >= 0.85
    assert CONFIG.ml_retrain_days == 30


def test_compute_bar_features_nonempty():
    feats = compute_bar_features(_synthetic_ohlcv())
    assert "mom_21d" in feats
    assert "ema50" in feats
    assert feats["atr"] > 0


def test_correlation_filter_reduces_cluster():
    cache = FeatureCache()
    base = _synthetic_ohlcv(180)
    for i, ticker in enumerate(["AAAUSD", "BBBUSD", "CCCUSD"]):
        df = base.copy()
        df["close"] = df["close"] * (1.0 + 0.001 * i)
        for _, row in df.iterrows():
            cache._bars[ticker].append(row.to_dict())
    ranked = [("AAAUSD", 1.0), ("BBBUSD", 0.9), ("CCCUSD", 0.8)]
    picks = filter_uncorrelated_picks(ranked, cache, top_k=2, max_corr=0.99)
    assert len(picks) >= 1


def test_sentiment_adjusts_cap():
    btc = {"mom_7d": 0.05, "mom_21d": 0.1, "trend_quality": 0.02}
    eth = {"mom_7d": 0.03}
    snap = compute_sentiment(btc_features=btc, eth_features=eth, breadth=0.7, median_rv=0.4)
    assert 0.0 <= snap.fear_greed <= 1.0
    boosted = adjust_deployment_cap(0.9, snap, "bull")
    assert boosted >= 0.9


def test_ml_trainer_retrain():
    trainer = MLTrainer()
    for i in range(100):
        row = {k: 0.01 * (i % 5) for k in (
            "mom_7d", "mom_21d", "mom_accel", "breakout_strength", "volume_surge",
            "rsi_pullback", "trend_quality", "rv_21d_inv", "breadth", "btc_beta",
        )}
        trainer.samples.append((row, 1.0 if i % 2 == 0 else 0.0))
    scorer = MLScorer(load_ml_weights())
    blob = trainer.retrain(scorer, datetime(2025, 6, 1, tzinfo=timezone.utc))
    assert "weights" in blob or scorer.weights


def test_load_ml_weights_builtin_only():
    blob = load_ml_weights()
    assert blob["bias"] == 0.0
    assert "weights" in blob
    assert MLScorer(blob).weights


def test_scalper_entry_ranging():
    frame = _synthetic_ohlcv(120)
    frame["close"] = frame["close"] * 0.92
    feats = build_scalper_features(frame)
    ok, _ = evaluate_scalper_entry(feats, btc_ret_1h=0.0, btc_ret_6h=0.0, last_trade_hours=99.0)
    assert isinstance(ok, bool)


def test_scalper_exit_overshoot():
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    st = PositionRisk(100.0, now, 1.0, 100.0, strategy_owner="scalper")
    ok, reason = evaluate_scalper_exit(
        entry_price=100.0,
        entry_time=now,
        current_price=101.0,
        now=now,
        feats={"z_20h": 1.0},
        highest_close=101.0,
        entry_atr=1.0,
    )
    assert ok and reason == "overshoot"


def test_regime_scalper_allowed_in_neutral():
    engine = RegimeEngine()
    snap = compute_sentiment(
        btc_features={"mom_7d": 0.0, "mom_21d": 0.01, "trend_quality": 0.0},
        eth_features={"mom_7d": 0.0},
        breadth=0.5,
        median_rv=0.5,
    )
    reg = engine.classify(btc_features={"mom_7d": 0.0, "mom_21d": 0.01, "trend_quality": 0.0}, breadth=0.5, median_rv=0.5, sentiment=snap)
    assert reg.name == "neutral"


def test_btc_beta_proxy():
    assert btc_beta_vs({"mom_7d": 0.2}, {"mom_7d": 0.1}) == pytest.approx(0.1)


def test_long_only_and_cash_config():
    assert CONFIG.enable_shorts is False
    assert CONFIG.starting_cash == 1000.0
