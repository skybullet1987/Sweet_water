from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from scalper import evaluate_entry


def _feats(z=-2.5, rsi=30.0, rv=0.5):
    return {"z_20h": z, "rsi_14": rsi, "rv_21d": rv}


def test_entry_all_pass():
    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=StrategyConfig(),
    )
    assert (ok, reason) == (True, "OK")


def test_entry_gate_reasons():
    cfg = StrategyConfig()

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=True,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert (ok, reason) == (False, "already_long")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(z=-1.5),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("z_above_threshold")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(rsi=31.0),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("mr_rsi_confirm_fail")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(rsi=50.0),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("rsi_out_of_band")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(rv=2.0),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("rv_too_high")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=-0.02,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("btc_1h_breakdown")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=-0.05,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("btc_6h_cascade")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=1.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("anti_churn")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.1,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("insufficient_cash")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=-0.04,
        consecutive_losses_for_symbol=0,
        config=cfg,
    )
    assert ok is False and reason.startswith("daily_loss_brake")

    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.5,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=3,
        config=cfg,
    )
    assert ok is False and reason.startswith("loss_streak")


def test_entry_cash_gate_shrinks_instead_of_rejecting_when_notional_above_min():
    cfg = StrategyConfig()
    ok, reason = evaluate_entry(
        symbol="ETHUSD",
        feats=_feats(),
        btc_ret_1h=0.0,
        btc_ret_6h=0.0,
        has_position=False,
        last_trade_hours_ago=10.0,
        available_cash_pct=0.08,
        daily_pnl_pct=0.0,
        consecutive_losses_for_symbol=0,
        equity=500.0,
        config=cfg,
    )
    assert (ok, reason) == (True, "OK")
