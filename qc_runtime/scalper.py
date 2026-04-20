from __future__ import annotations

from dataclasses import dataclass

try:
    from config import CONFIG
except ModuleNotFoundError:
    from .config import CONFIG  # type: ignore


@dataclass
class ScalperDecision:
    action: str  # "enter" | "exit" | "skip"
    symbol: object
    reason: str
    score: float = 0.0
    tag: str = ""


def evaluate_entry(
    *,
    symbol,
    feats: dict,
    btc_ret_1h: float,
    btc_ret_6h: float,
    has_position: bool,
    last_trade_hours_ago: float,
    available_cash_pct: float,
    daily_pnl_pct: float,
    consecutive_losses_for_symbol: int,
    config=CONFIG,
) -> tuple[bool, str]:
    if has_position:
        return False, "already_long"
    z = float(feats.get("z_20h", 0.0) or 0.0)
    if z > config.scalper_z_entry:
        return False, f"z_above_threshold:{z:.2f}"
    rsi = float(feats.get("rsi_14", 50.0) or 50.0)
    if not (config.scalper_rsi_min <= rsi <= config.scalper_rsi_max):
        return False, f"rsi_out_of_band:{rsi:.1f}"
    rv = float(feats.get("rv_21d", 0.0) or 0.0)
    if rv > config.scalper_rv_max:
        return False, f"rv_too_high:{rv:.2f}"
    if btc_ret_1h < config.scalper_btc_1h_floor:
        return False, f"btc_1h_breakdown:{btc_ret_1h:.3f}"
    if btc_ret_6h < config.scalper_btc_6h_floor:
        return False, f"btc_6h_cascade:{btc_ret_6h:.3f}"
    if last_trade_hours_ago < config.scalper_anti_churn_hours:
        return False, f"anti_churn:{last_trade_hours_ago:.1f}h"
    if available_cash_pct < config.scalper_position_size_pct:
        return False, f"insufficient_cash:{available_cash_pct:.2f}"
    if daily_pnl_pct < config.scalper_daily_loss_brake:
        return False, f"daily_loss_brake:{daily_pnl_pct:.3f}"
    if consecutive_losses_for_symbol >= config.scalper_consecutive_loss_brake:
        return False, f"loss_streak:{consecutive_losses_for_symbol}"
    return True, "OK"


def evaluate_exit(
    *,
    symbol,
    feats: dict,
    entry_price: float,
    entry_time,
    current_time,
    current_price: float,
    btc_ret_1h: float,
    config=CONFIG,
) -> tuple[bool, str]:
    if current_price <= 0 or entry_price <= 0:
        return False, ""
    pnl_pct = current_price / entry_price - 1.0
    z = float(feats.get("z_20h", 0.0) or 0.0)
    hours_held = (current_time - entry_time).total_seconds() / 3600.0

    if btc_ret_1h < config.scalper_btc_panic_threshold:
        return True, "BTCPanic"
    if pnl_pct <= config.scalper_hard_sl_pct:
        return True, "SL"
    if hours_held >= config.scalper_time_stop_hours:
        return True, "TimeStop"
    if z >= config.scalper_overshoot_z:
        return True, "Overshoot"
    if z >= config.scalper_meanrev_z:
        return True, "MeanRev"
    return False, ""
