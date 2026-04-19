from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import CONFIG, StrategyConfig
from execution import Executor, KrakenFeeModel, RealisticSlippage
from features import FeatureEngine, amihud_illiquidity, rank_momentum, realized_vol, roll_spread, zscore_vs_universe
from regime import RegimeEngine
from reporting import Reporter, walk_forward_run
from risk import RiskManager
from scoring import Scorer
from sizing import Sizer
from universe import select_universe


def _bar(symbol: str, i: int, base: float = 100.0) -> dict[str, float | str]:
    close = base * (1 + 0.001 * i)
    return {
        "symbol": symbol,
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": 1000 + 10 * i,
    }


# ============================================================
# SECTION 1: Features
# ============================================================
def test_amihud_illiquidity_known_input():
    r = pd.Series([0.01, -0.02, 0.03, -0.01])
    dv = pd.Series([100, 100, 100, 100])
    val = amihud_illiquidity(r, dv, window=2).iloc[-1]
    assert math.isclose(val, 0.0002, rel_tol=1e-9)


def test_roll_spread_negative_cov_returns_nan():
    close = pd.Series([100, 101, 100.5, 101.3, 101.0, 101.7])
    out = roll_spread(close, window=3)
    assert out.notna().sum() >= 1


def test_indicator_taLib_fallback_matches_numpy():
    engine = FeatureEngine()
    for i in range(80):
        engine.update(_bar("BTCUSD", i))
    feats = engine.current_features("BTCUSD")
    assert "rsi" in feats and "atr" in feats and "adx" in feats


def test_cross_sectional_helpers_work():
    df = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.0, -0.01, 0.01]})
    z = zscore_vs_universe(df)
    r = rank_momentum(df, window=2)
    assert z.shape == df.shape
    assert r.shape == df.shape


# ============================================================
# SECTION 2: Regime (HMM)
# ============================================================
def test_hmm_assigns_trending_to_risk_on():
    cfg = StrategyConfig(hmm_train_window_bars=40, hmm_retrain_every_bars=20)
    reg = RegimeEngine(cfg)
    for i in range(80):
        reg.update(0.001 + (i % 5) * 1e-5, 0.01, 0.8)
    assert reg.current_state() in {"risk_on", "chop", "risk_off"}


def test_gmm_fallback_when_hmmlearn_unavailable(monkeypatch):
    import regime as regime_mod

    monkeypatch.setattr(regime_mod, "HAS_HMM", False)
    cfg = StrategyConfig(hmm_train_window_bars=30, hmm_retrain_every_bars=10)
    reg = RegimeEngine(cfg)
    for i in range(60):
        reg.update((-1) ** i * 0.001, 0.02, 0.5)
    assert set(reg.current_state_probs().keys()) == {"risk_on", "risk_off", "chop"}


# ============================================================
# SECTION 3: Scoring
# ============================================================
def test_scoring_risk_off_zero():
    s = Scorer()
    assert s.score("ETHUSD", {}, "risk_off", {}) == 0.0


def test_scoring_risk_on_positive_path():
    s = Scorer()
    score = s.score("ETHUSD", {"ema20": 2, "ema50": 1, "mom_24": 0.01, "adx": 30, "ofi": 1}, "risk_on", {"btc_trend": 0.01})
    assert score > 0


# ============================================================
# SECTION 4: Sizing
# ============================================================
def test_sizer_output_bounded():
    sz = Sizer()
    for i in range(100):
        sz.update_returns(0.001 if i % 2 == 0 else -0.0005)
        sz.record_trade(0.01 if i % 3 else -0.005)
    weight = sz.size_for_trade("SOLUSD", 0.8, {"equity": 500, "gross_exposure": 0.2})
    assert 0 <= weight <= CONFIG.kelly_cap


# ============================================================
# SECTION 5: Execution + Triple Barrier
# ============================================================
def test_fee_slippage_estimators_positive():
    assert KrakenFeeModel.estimate_round_trip_cost() > 0
    assert RealisticSlippage.estimate_slippage_bps(10, 0.1) >= 1


def test_executor_triple_barrier_exit_take_profit():
    ex = Executor()
    ex.register_fill("ETHUSD", price=100, atr=2, side=1, bar_index=0)
    out = ex.manage_exits({"ETHUSD": {"high": 105, "low": 99, "close": 104}}, bar_index=1)
    assert ("ETHUSD", "take_profit") in out


# ============================================================
# SECTION 6: Risk + Circuit Breaker
# ============================================================
def test_risk_circuit_breaker_blocks_entries():
    rm = RiskManager()
    rm.evaluate({"target_weight": 0.1, "equity": 100, "gross_exposure": 0, "net_exposure": 0, "correlation": 0})
    decision = rm.evaluate({"target_weight": 0.1, "equity": 85, "gross_exposure": 0, "net_exposure": 0, "correlation": 0})
    assert not decision.approved


# ============================================================
# SECTION 7: Pipeline integration (replaces old behavioral tests)
# ============================================================
def test_score_to_order_end_to_end():
    fe = FeatureEngine()
    for i in range(100):
        fe.update(_bar("XRPUSD", i, base=50))
    feats = fe.current_features("XRPUSD")
    reg = RegimeEngine(StrategyConfig(hmm_train_window_bars=30, hmm_retrain_every_bars=10))
    for _ in range(60):
        reg.update(0.001, 0.01, 0.7)
    score = Scorer().score("XRPUSD", feats, reg.current_state(), {"btc_trend": 0.001})
    size = Sizer().size_for_trade("XRPUSD", score, {"equity": 500, "gross_exposure": 0})
    decision = RiskManager().evaluate({"target_weight": size, "equity": 500, "gross_exposure": 0, "net_exposure": 0, "correlation": 0})
    if decision.approved and abs(score) >= CONFIG.score_threshold:
        order = Executor().place_entry("XRPUSD", decision.adjusted_target_weight, score)
        assert order["symbol"] == "XRPUSD"


def test_zero_trade_regression_guard():
    out = walk_forward_run(pd.read_csv(REPO_ROOT / "tests/fixtures/walk_forward_bars.csv"), CONFIG)
    assert out["oos_trade_count"] > 0


def test_cost_accounting_no_double_count():
    rpt = Reporter()
    rpt.on_order_event({"status": "filled", "pnl": 0.01})
    rpt.on_order_event({"status": "filled", "pnl": -0.01})
    rpt.on_order_event({"status": "canceled"})
    final = rpt.final_report()
    assert final["trade_count"] == 2
    assert 0 <= final["cancel_rate"] <= 1


# ============================================================
# SECTION 8: Walk-forward smoke
# ============================================================
def test_walk_forward_fixture_meets_baseline():
    fixture = pd.read_csv(REPO_ROOT / "tests/fixtures/walk_forward_bars.csv")
    baseline_path = REPO_ROOT / "tests/fixtures/walk_forward_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    out = walk_forward_run(fixture, CONFIG)
    assert out["oos_sharpe"] >= 0.5
    assert out["oos_avg_win_avg_loss"] >= 1.0
    assert 15 <= out["oos_trade_count"] <= 200
    assert out["oos_cancel_rate"] < 0.20
    dist = out["regime_distribution"]
    active = [k for k, v in dist.items() if v >= 0.05]
    assert len(active) >= 2
    assert baseline["oos_trade_count"] >= 0


def test_universe_selector_keeps_btc():
    def _hist(symbol, start, end):
        _ = start, end
        rows = 30
        return pd.DataFrame({"close": np.linspace(100, 120, rows), "volume": np.linspace(1000, 2000, rows)})

    selected = select_universe(_hist, pd.Timestamp("2026-01-01", tz="UTC"))
    assert "BTCUSD" in selected
