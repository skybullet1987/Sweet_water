from __future__ import annotations

from dataclasses import dataclass
import math

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
    equity: float = 0.0,
    sleeve: str = "meanrev",
    config=CONFIG,
) -> tuple[bool, str]:
    if has_position:
        return False, "already_long"
    regime = regime_for(feats, config=config)
    z = float(feats.get("z_20h", 0.0) or 0.0)
    if sleeve == "momentum":
        if z < float(getattr(config, "scalper_breakout_z_entry", 2.0)):
            return False, f"z_below_breakout:{z:.2f}"
        if regime not in {"uptrend", "uptrend_pullback", "uptrend_breakout"}:
            return False, f"REGIME_BLOCK:{regime}"
        vol_rel = float(feats.get("volume_rel_20h", 0.0) or 0.0)
        if vol_rel < float(getattr(config, "scalper_breakout_volume_mult", 1.5)):
            return False, f"vol_confirm_fail:{vol_rel:.2f}"
    else:
        if z > config.scalper_z_entry:
            return False, f"z_above_threshold:{z:.2f}"
        if regime not in {"uptrend_pullback", "ranging", "neutral"}:
            return False, f"REGIME_BLOCK:{regime}"
    rsi = float(feats.get("rsi_14", 50.0) or 50.0)
    if sleeve != "momentum" and not (config.scalper_rsi_min <= rsi <= config.scalper_rsi_max):
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
    effective_size_pct = effective_position_size_pct(available_cash_pct, config=config)
    if effective_size_pct <= 0:
        return False, f"insufficient_cash:{available_cash_pct:.2f}"
    if available_cash_pct < float(getattr(config, "scalper_position_size_pct", 0.15) or 0.15):
        min_notional = float(getattr(config, "min_notional_usd", 5.0) or 5.0)
        if effective_size_pct * max(float(equity or 0.0), 0.0) < min_notional:
            return False, f"insufficient_cash:{available_cash_pct:.2f}"
    if daily_pnl_pct < config.scalper_daily_loss_brake:
        return False, f"daily_loss_brake:{daily_pnl_pct:.3f}"
    if consecutive_losses_for_symbol >= config.scalper_consecutive_loss_brake:
        return False, f"loss_streak:{consecutive_losses_for_symbol}"
    return True, "OK"


def effective_position_size_pct(available_cash_pct: float, config=CONFIG) -> float:
    size_pct = float(getattr(config, "scalper_position_size_pct", 0.15) or 0.15)
    avail = max(float(available_cash_pct or 0.0), 0.0)
    if avail < size_pct:
        return max(avail * 0.95, 0.0)
    return size_pct


def evaluate_exit(
    *,
    symbol,
    feats: dict,
    entry_price: float,
    entry_time,
    current_time,
    current_price: float,
    btc_ret_1h: float,
    highest_close: float | None = None,
    entry_atr: float | None = None,
    sleeve: str = "meanrev",
    config=CONFIG,
) -> tuple[bool, str]:
    if current_price <= 0 or entry_price <= 0:
        return False, ""
    if entry_time is None:
        return False, ""
    pnl_pct = current_price / entry_price - 1.0
    z = float(feats.get("z_20h", 0.0) or 0.0)
    hours_held = (current_time - entry_time).total_seconds() / 3600.0

    if btc_ret_1h < config.scalper_btc_panic_threshold:
        return True, "BTCPanic"
    if pnl_pct <= config.scalper_hard_sl_pct:
        return True, "SL"
    atr = float(entry_atr or 0.0)
    if not math.isfinite(atr) or atr <= 0:
        atr = entry_price * 0.01
    high_ref = float(highest_close if highest_close is not None else current_price)
    trail_mult = (
        float(getattr(config, "scalper_trail_atr_mult_mom", 3.0))
        if sleeve == "momentum"
        else float(getattr(config, "scalper_trail_atr_mult_mr", 2.0))
    )
    tp_mult = (
        float(getattr(config, "scalper_tp_atr_mult_mom", 5.0))
        if sleeve == "momentum"
        else float(getattr(config, "scalper_tp_atr_mult_mr", 2.0))
    )
    base_stop = entry_price * (1.0 + float(getattr(config, "scalper_hard_sl_pct", -0.015)))
    trailing_stop = max(base_stop, high_ref - trail_mult * atr)
    risk_r = max(entry_price - trailing_stop, atr)
    tp_price = entry_price + max(tp_mult * atr, 1.5 * risk_r)
    if current_price <= trailing_stop:
        return True, "ATRStop"
    if current_price >= tp_price:
        return True, "TP"
    stop_hours = float(config.scalper_time_stop_hours) * (2.0 if sleeve == "momentum" else 1.0)
    if hours_held >= stop_hours:
        return True, "TimeStop"
    if sleeve != "momentum":
        if z >= config.scalper_overshoot_z:
            return True, "Overshoot"
        if z >= config.scalper_meanrev_z:
            return True, "MeanRev"
    return False, ""


def regime_for(feats: dict, config=CONFIG) -> str:
    close = float(feats.get("close", feats.get("price", 0.0)) or 0.0)
    ema50 = float(feats.get("ema50", 0.0) or 0.0)
    ema200 = float(feats.get("ema200", 0.0) or 0.0)
    adx = float(feats.get("adx", 0.0) or 0.0)
    new_high = bool(feats.get("new_high_24h", False))
    new_low = bool(feats.get("new_low_24h", False))
    if close <= 0 or ema50 <= 0 or ema200 <= 0:
        return "neutral"
    is_up = close > ema50 > ema200
    is_down = close < ema50 < ema200
    pullback = close > ema200 and close < ema50
    if adx < float(getattr(config, "scalper_adx_range_max", 20.0)):
        return "ranging"
    if is_up and adx > float(getattr(config, "scalper_adx_trend_min", 30.0)) and new_high:
        return "uptrend_breakout"
    if is_down and adx > float(getattr(config, "scalper_adx_trend_min", 30.0)) and new_low:
        return "downtrend_breakdown"
    if pullback:
        return "uptrend_pullback"
    if is_up:
        return "uptrend"
    if is_down:
        return "downtrend"
    return "neutral"


def vol_target_qty(
    *,
    equity: float,
    price: float,
    atr_pct: float,
    available_cash: float,
    current_gross_exposure_pct: float,
    risk_per_trade_pct: float,
    max_symbol_exposure_pct: float,
    max_gross_exposure_pct: float,
    position_size_pct: float = 1.0,
) -> tuple[float, float]:
    if equity <= 0 or price <= 0:
        return 0.0, 0.0
    atr_pct = max(float(atr_pct or 0.0), 0.0025)
    risk_budget = max(0.0, float(equity) * float(risk_per_trade_pct))
    raw_notional = risk_budget / atr_pct
    max_symbol_notional = float(equity) * float(max_symbol_exposure_pct)
    max_gross_notional = max(0.0, float(equity) * float(max_gross_exposure_pct) - float(equity) * float(current_gross_exposure_pct))
    size_cap_notional = max(0.0, float(equity) * max(float(position_size_pct or 0.0), 0.0))
    notional = max(0.0, min(raw_notional, max_symbol_notional, max_gross_notional, float(available_cash) * 0.97, size_cap_notional))
    qty = notional / float(price)
    return qty, notional
