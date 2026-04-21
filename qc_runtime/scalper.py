from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd

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
        if z < float(getattr(config, "scalper_breakout_z_entry", 2.3)):
            return False, f"z_below_breakout:{z:.2f}"
        ret_6h = float(feats.get("ret_6h", 0.0) or 0.0)
        if ret_6h <= 0:
            return False, f"ret6h_confirm_fail:{ret_6h:.4f}"
        if regime not in {"uptrend", "uptrend_pullback", "uptrend_breakout"}:
            return False, f"REGIME_BLOCK:{regime}"
        vol_rel = float(feats.get("volume_rel_20h", 0.0) or 0.0)
        if vol_rel < float(getattr(config, "scalper_breakout_volume_mult", 1.2)):
            return False, f"vol_confirm_fail:{vol_rel:.2f}"
    else:
        if z > config.scalper_z_entry:
            return False, f"z_above_threshold:{z:.2f}"
        if regime not in {"uptrend_pullback", "ranging", "neutral"}:
            return False, f"REGIME_BLOCK:{regime}"
        rsi = float(feats.get("rsi_14", 50.0) or 50.0)
        if not (config.scalper_rsi_min <= rsi <= config.scalper_rsi_max):
            return False, f"rsi_out_of_band:{rsi:.1f}"
        meanrev_rsi_cap = float(getattr(config, "scalper_meanrev_rsi_long_max", 30.0) or 30.0)
        if rsi > meanrev_rsi_cap:
            return False, f"mr_rsi_confirm_fail:{rsi:.1f}"
        close = float(feats.get("close", feats.get("price", 0.0)) or 0.0)
        atr = float(feats.get("atr", 0.0) or 0.0)
        low_50 = float(feats.get("low_50", close) or close)
        if close > 0 and atr > 0 and low_50 > 0 and close > (low_50 + 2.0 * atr):
            return False, f"mr_low_confirm_fail:{close:.4f}"
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
    initial_risk_distance: float | None = None,
    partial_tp_done: bool = False,
    tight_trail_armed: bool = False,
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
    atr = float(entry_atr or 0.0)
    if not math.isfinite(atr) or atr <= 0:
        atr = entry_price * 0.01
    stop_atr_mult = float(getattr(config, "scalper_stop_atr_mult", 1.5) or 1.5)
    base_risk = max(float(initial_risk_distance or 0.0), stop_atr_mult * atr)
    high_ref = float(highest_close if highest_close is not None else current_price)
    trail_mult = float(
        getattr(
            config,
            "scalper_tight_trail_atr_mult" if tight_trail_armed else "scalper_chandelier_atr_mult",
            0.5 if tight_trail_armed else 2.0,
        )
        or (0.5 if tight_trail_armed else 2.0)
    )
    base_stop = entry_price - base_risk
    if partial_tp_done:
        base_stop = max(base_stop, entry_price)
    trailing_stop = max(base_stop, high_ref - trail_mult * atr)
    r_multiple = (current_price - entry_price) / max(base_risk, 1e-9)
    if current_price <= trailing_stop:
        return True, "SL" if current_price <= entry_price else "ATRStop"
    if r_multiple >= float(getattr(config, "scalper_tp2_r", 2.5) or 2.5):
        return True, "TP"
    if (not partial_tp_done) and r_multiple >= float(getattr(config, "scalper_tp1_r", 1.0) or 1.0):
        return True, "TP1"
    stop_hours = float(
        getattr(config, "scalper_mom_max_hold_bars", 24) if sleeve == "momentum" else getattr(config, "scalper_mr_max_hold_bars", 12)
    )
    if hours_held >= stop_hours:
        if pnl_pct <= 0:
            return True, "TimeStop"
        if not tight_trail_armed:
            return False, "TightTrailArmed"
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
    raw_notional = risk_budget / max(1.5 * atr_pct, 1e-9)
    max_symbol_notional = float(equity) * float(max_symbol_exposure_pct)
    max_gross_notional = max(0.0, float(equity) * float(max_gross_exposure_pct) - float(equity) * float(current_gross_exposure_pct))
    size_cap_notional = max(0.0, float(equity) * max(float(position_size_pct or 0.0), 0.0))
    notional = max(0.0, min(raw_notional, max_symbol_notional, max_gross_notional, float(available_cash) * 0.97, size_cap_notional))
    qty = notional / float(price)
    return qty, notional


def corr_hits_from_state(*, feature_state: dict, symbol, held: list, threshold: float) -> int:
    def _sym_key(sym):
        return getattr(sym, "Value", str(sym))

    def _recent_returns(raw: dict) -> list[float]:
        closes = list(raw.get("close_history_24h", []))
        if len(closes) >= 12:
            out = []
            for i in range(1, len(closes)):
                prev = float(closes[i - 1] or 0.0)
                cur = float(closes[i] or 0.0)
                if prev > 0 and cur > 0:
                    out.append(float(math.log(cur / prev)))
            return out[-24:]
        return list(raw.get("daily_logret", []))[-30:]

    cand = feature_state.get(_sym_key(symbol), {})
    cand_ret = _recent_returns(cand)
    if len(cand_ret) < 10:
        return 0
    hits = 0
    for held_sym in held:
        other = feature_state.get(_sym_key(held_sym), {})
        other_ret = _recent_returns(other)
        n = min(len(cand_ret), len(other_ret))
        if n < 10:
            continue
        corr = float(pd.Series(cand_ret[-n:], dtype=float).corr(pd.Series(other_ret[-n:], dtype=float))) if n > 1 else 0.0
        if math.isfinite(corr) and corr >= threshold:
            hits += 1
    return hits
