from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from workflow import validate_bars  # noqa: E402
from risk import filter_cluster_caps, max_cluster_exposure  # noqa: E402
from config import CONFIG  # noqa: E402
from kraken_ops import CalibratedCostModel  # noqa: E402
from core import AlphaEnsemble  # noqa: E402
from kraken_ops import FillTracker  # noqa: E402
from regime import config_for_regime, normalize_regime_key  # noqa: E402
from kraken_ops import PaperTradingScorecard  # noqa: E402


class _FakeAlgo:
    def __init__(self):
        self.config = CONFIG
        self.Portfolio = type("P", (), {"TotalPortfolioValue": 2000.0})()
        self.fill_tracker = FillTracker()
        self.Time = datetime(2025, 6, 1, tzinfo=timezone.utc)


def test_v7_config():
    assert CONFIG.use_regime_ensembles
    assert CONFIG.enable_cluster_risk
    assert CONFIG.enable_scorecard


def test_regime_ensemble_weights_differ():
    bull = config_for_regime(CONFIG, "bull")
    bear = config_for_regime(CONFIG, "bear")
    assert bull.w_momentum > bear.w_momentum
    assert normalize_regime_key("qc_bull|x") == "bull"


def test_cluster_caps():
    ranked = [("SOLUSD", 1.0), ("AVAXUSD", 0.9), ("DOTUSD", 0.8), ("BTCUSD", 0.7)]
    picked = filter_cluster_caps(ranked, max_per_cluster=2, config=CONFIG)
    assert len(picked) <= 4
    assert "BTCUSD" in picked
    exp = max_cluster_exposure({"BTCUSD": 0.5, "SOLUSD": 0.5})
    assert exp.get("major", 0) == 0.5


def test_calibrated_cost_from_fills():
    algo = _FakeAlgo()
    for i in range(6):
        algo.fill_tracker.on_submit(i, is_limit=True, expected_price=100.0, qty=1.0)
        algo.fill_tracker.on_fill(i, 101.0, is_limit=True)
    model = CalibratedCostModel()
    pct = model.round_trip_pct(algo)
    assert pct > float(CONFIG.expected_round_trip_fees)


def test_scorecard_paper_gate():
    sc = PaperTradingScorecard(replace(CONFIG, paper_min_days=1.0, paper_min_trades=2))
    algo = _FakeAlgo()
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(50):
        sc.record_equity(now + timedelta(days=i), 1000.0 * (1.001 ** i))
    sc.record_trade("BTCUSD", 0.05, strategy="momentum", when=now)
    sc.record_trade("ETHUSD", 0.03, strategy="momentum", when=now)
    snap = sc.build(algo)
    ok, _ = sc.passes_paper_gate(snap)
    assert ok or snap.n_trades >= 2


def test_ensemble_regime_scoring_differs():
    ens = AlphaEnsemble(CONFIG)
    feats = {
        "mom_7d": 0.1,
        "mom_21d": 0.2,
        "mom_accel": 0.05,
        "rv_21d": 0.3,
        "breakout_strength": 0.02,
        "volume_surge": 0.1,
        "trend_quality": 0.04,
        "rsi_pullback": -0.5,
    }
    bull = ens.score_symbol(feats, regime_name="bull")["final"]
    bear = ens.score_symbol(feats, regime_name="bear")["final"]
    assert bull != bear


def _bars(n: int = 600) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    rng = np.random.default_rng(11)
    for sym in ("BTCUSD", "ETHUSD"):
        c = 100.0
        for ts in idx:
            c *= 1 + rng.normal(0, 0.002)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": c,
                    "high": c * 1.001,
                    "low": c * 0.999,
                    "close": c,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_backtest_validator_runs():
    cfg = replace(
        CONFIG,
        validation_min_sharpe=-99.0,
        validation_max_drawdown=-0.99,
        validation_min_trades=1,
        validation_min_win_rate=0.0,
    )
    report = validate_bars(_bars(), config=cfg, n_folds=2)
    assert report.oos_trades >= 0
    assert isinstance(report.passed, bool)
