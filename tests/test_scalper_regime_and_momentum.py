from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from scalper import evaluate_entry, regime_for, vol_target_qty


def test_regime_for_identifies_pullback_and_breakout():
    cfg = StrategyConfig()
    pullback = regime_for({"close": 110.0, "ema50": 120.0, "ema200": 100.0, "adx": 18.0}, config=cfg)
    breakout = regime_for(
        {"close": 130.0, "ema50": 120.0, "ema200": 100.0, "adx": 35.0, "new_high_24h": 1.0}, config=cfg
    )
    assert pullback == "ranging"
    assert breakout == "uptrend_breakout"


def test_momentum_entry_needs_breakout_and_volume_confirmation():
    cfg = StrategyConfig()
    feats = {
        "z_20h": 2.5,
        "rsi_14": 55.0,
        "rv_21d": 0.6,
        "close": 130.0,
        "ema50": 120.0,
        "ema200": 100.0,
        "adx": 35.0,
        "new_high_24h": 1.0,
        "volume_rel_20h": 1.8,
    }
    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=feats,
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        sleeve="momentum",
        config=cfg,
    )
    assert (ok, reason) == (True, "OK")

    feats["volume_rel_20h"] = 1.1
    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=feats,
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        sleeve="momentum",
        config=cfg,
    )
    assert ok is False and reason.startswith("vol_confirm_fail")


def test_vol_target_qty_respects_risk_and_exposure_caps():
    qty, notional = vol_target_qty(
        equity=1_000.0,
        price=100.0,
        atr_pct=0.05,
        available_cash=800.0,
        current_gross_exposure_pct=0.1,
        risk_per_trade_pct=0.005,
        max_symbol_exposure_pct=0.25,
        max_gross_exposure_pct=0.90,
    )
    assert qty > 0
    assert notional <= 250.0 + 1e-9


def test_vol_target_qty_respects_effective_position_size_pct():
    qty, notional = vol_target_qty(
        equity=1_000.0,
        price=100.0,
        atr_pct=0.01,
        available_cash=800.0,
        current_gross_exposure_pct=0.0,
        risk_per_trade_pct=0.02,
        max_symbol_exposure_pct=0.50,
        max_gross_exposure_pct=0.95,
        position_size_pct=0.08,
    )
    assert qty > 0
    assert notional <= 80.0 + 1e-9
