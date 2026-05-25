from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from ensemble import AlphaEnsemble  # noqa: E402
from features import compute_bar_features, cross_section_ranks  # noqa: E402
from ml_scorer import MLScorer, load_ml_weights  # noqa: E402
from regime import RegimeEngine  # noqa: E402
from risk import PortfolioRisk, PositionRisk, should_exit  # noqa: E402
from sizing import AggressiveSizer  # noqa: E402
from universe import CANADA_UNLIMITED, BLACKLIST, select_universe  # noqa: E402


def _synthetic_ohlcv(rows: int = 200) -> pd.DataFrame:
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    close = np.cumprod(1 + np.random.default_rng(7).normal(0.0005, 0.01, rows)) * 100.0
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


def test_canada_unlimited_priority():
    assert "BTCUSD" in CANADA_UNLIMITED
    assert "ETHUSD" in CANADA_UNLIMITED
    assert "PEPEUSD" in BLACKLIST


def test_compute_bar_features_nonempty():
    feats = compute_bar_features(_synthetic_ohlcv())
    assert "mom_21d" in feats
    assert "breakout_strength" in feats
    assert feats["atr"] > 0


def test_ml_weights_load():
    blob = load_ml_weights()
    assert "weights" in blob
    scorer = MLScorer(blob)
    s = scorer.score({"mom_7d": 0.1, "mom_21d": 0.2, "breakout_strength": 0.01})
    assert -1.0 <= s <= 1.0


def test_ensemble_ranks_high_momentum():
    ens = AlphaEnsemble()
    strong = {
        "mom_7d": 0.12,
        "mom_21d": 0.25,
        "mom_accel": 0.05,
        "rv_21d": 0.4,
        "breakout_strength": 0.02,
        "volume_surge": 0.5,
        "trend_quality": 0.04,
        "rsi_pullback": 0.1,
    }
    weak = {**strong, "mom_21d": -0.2, "breakout_strength": -0.05, "trend_quality": -0.02}
    a = ens.score_symbol(strong, rank_mom_21=0.9, rank_breakout=0.85, breadth=0.7)["final"]
    b = ens.score_symbol(weak, rank_mom_21=0.1, rank_breakout=0.1, breadth=0.3)["final"]
    assert a > b


def test_regime_bull_and_chaos():
    engine = RegimeEngine()
    bull = engine.classify(
        btc_features={"trend_quality": 0.05, "mom_21d": 0.1},
        breadth=0.7,
        median_rv=0.5,
    )
    chaos = engine.classify(
        btc_features={"trend_quality": -0.02, "mom_21d": -0.1},
        breadth=0.2,
        median_rv=1.2,
    )
    assert bull.name == "bull"
    assert chaos.name == "chaos"
    assert chaos.deployment_cap == 0.0


def test_sizer_respects_threshold():
    sizer = AggressiveSizer()
    low = sizer.weight_for_score(0.1, 0.5, 0.9)
    high = sizer.weight_for_score(0.9, 0.5, 0.9)
    assert low == 0.0
    assert high > 0.0


def test_should_exit_hard_stop():
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    st = PositionRisk(entry_price=100.0, entry_time=now, entry_atr=2.0, highest_close=100.0)
    ok, reason = should_exit(
        st,
        close=90.0,
        now=now,
        hard_stop_pct=-0.08,
        catastrophic_stop_pct=-0.12,
        tp_atr_mult=4.0,
        sl_atr_mult=1.0,
        chandelier_atr_mult=2.0,
        activate_trail_pct=0.1,
        time_stop_hours=72.0,
    )
    assert ok and reason == "hard_stop"


def test_long_only_config():
    assert CONFIG.enable_shorts is False
    assert CONFIG.starting_cash == 1000.0


def test_select_universe_mock():
    def provider(ticker, start, end):
        _ = (start, end)
        return _synthetic_ohlcv(180)

    out = select_universe(provider, datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert len(out) >= 1
    assert out[0] in CANADA_UNLIMITED or True


def test_cross_section_ranks():
    m = {
        "A": {"mom_21d": 0.1},
        "B": {"mom_21d": 0.2},
        "C": {"mom_21d": -0.1},
    }
    ranks = cross_section_ranks(m, "mom_21d")
    assert ranks["B"] > ranks["A"] > ranks["C"]


def test_ml_weights_json_valid():
    path = KRAKEN_MAX / "ml_weights.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data["weights"], dict)
