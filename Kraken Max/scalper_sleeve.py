from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from config import CONFIG, KrakenMaxConfig


def _z_score(closes: list[float], current: float) -> float:
    if len(closes) < 20:
        return 0.0
    window = closes[-20:]
    mu = sum(window) / len(window)
    var = sum((x - mu) ** 2 for x in window) / max(len(window) - 1, 1)
    sd = math.sqrt(max(var, 0.0))
    if sd <= 0:
        return 0.0
    return (current - mu) / sd


def scalper_regime(feats: dict[str, float], config: KrakenMaxConfig = CONFIG) -> str:
    close = float(feats.get("close", 0.0))
    ema50 = float(feats.get("ema50", 0.0))
    ema200 = float(feats.get("ema200", 0.0))
    adx = float(feats.get("adx", 15.0))
    if close <= 0 or ema50 <= 0 or ema200 <= 0:
        return "neutral"
    if adx < float(config.scalper_adx_range_max):
        return "ranging"
    if close > ema50 > ema200:
        return "uptrend_pullback"
    if close < ema50 < ema200:
        return "downtrend"
    return "neutral"


def build_scalper_features(frame) -> dict[str, float]:
    base = {}
    if frame is None or len(frame) < 30:
        return base
    close_s = frame["close"].astype(float)
    high_s = frame["high"].astype(float)
    low_s = frame["low"].astype(float)
    closes = close_s.tolist()
    c = float(close_s.iloc[-1])
    z = _z_score(closes, c)

    def _ret(h: int) -> float:
        idx = min(len(close_s) - 1, h)
        base_px = float(close_s.iloc[-1 - idx])
        return (c / base_px - 1.0) if base_px > 0 else 0.0

    ema50 = close_s.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close_s.ewm(span=200, adjust=False).mean().iloc[-1]
    tr = (high_s - low_s).abs()
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    delta = close_s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up.iloc[-1] / max(dn.iloc[-1], 1e-9)
    rsi = float(100 - (100 / (1 + rs)))

    vol_recent = float((close_s * frame["volume"].astype(float)).tail(24).median())
    vol_base = float((close_s * frame["volume"].astype(float)).tail(20 * 24).median())
    vol_rel = vol_recent / max(vol_base, 1e-9)

    out = {
        "close": c,
        "z_20h": z,
        "rsi_14": rsi,
        "ret_1h": _ret(1),
        "ret_6h": _ret(6),
        "atr": atr,
        "ema50": float(ema50),
        "ema200": float(ema200),
        "adx": 15.0,
        "volume_rel_20h": vol_rel,
        "rv_21d": float(close_s.pct_change().tail(24 * 21).std() * math.sqrt(24 * 365)),
    }
    out["scalper_regime"] = scalper_regime(out)
    return out


def evaluate_scalper_entry(
    feats: dict[str, float],
    *,
    btc_ret_1h: float,
    btc_ret_6h: float,
    last_trade_hours: float,
    config: KrakenMaxConfig = CONFIG,
) -> tuple[bool, str]:
    regime = str(feats.get("scalper_regime", "neutral"))
    if regime not in {"ranging", "uptrend_pullback", "neutral"}:
        return False, f"regime_block:{regime}"
    z = float(feats.get("z_20h", 0.0))
    entry_z = float(config.scalper_relaxed_z_entry if regime == "ranging" else config.scalper_z_entry)
    if z > entry_z:
        return False, f"z_above:{z:.2f}"
    rsi = float(feats.get("rsi_14", 50.0))
    if not (config.scalper_rsi_min <= rsi <= config.scalper_rsi_max):
        return False, f"rsi_band:{rsi:.1f}"
    if rsi > float(config.scalper_rsi_long_max):
        return False, f"rsi_confirm:{rsi:.1f}"
    if btc_ret_1h < float(config.scalper_btc_1h_floor):
        return False, "btc_1h"
    if btc_ret_6h < float(config.scalper_btc_6h_floor):
        return False, "btc_6h"
    if last_trade_hours < float(config.scalper_anti_churn_hours):
        return False, "anti_churn"
    rv = float(feats.get("rv_21d", 0.0))
    if rv > 1.8:
        return False, f"rv_high:{rv:.2f}"
    return True, "OK"


def evaluate_scalper_exit(
    *,
    entry_price: float,
    entry_time: datetime,
    current_price: float,
    now: datetime,
    feats: dict[str, float],
    highest_close: float,
    entry_atr: float,
    config: KrakenMaxConfig = CONFIG,
) -> tuple[bool, str]:
    if entry_price <= 0 or current_price <= 0:
        return False, ""
    pnl = current_price / entry_price - 1.0
    if pnl <= float(config.scalper_hard_stop_pct):
        return True, "hard_stop"
    z = float(feats.get("z_20h", 0.0))
    if z >= float(config.scalper_overshoot_z):
        return True, "overshoot"
    if z >= float(config.scalper_meanrev_z):
        return True, "mean_revert"
    atr = max(entry_atr, entry_price * 0.008)
    risk = atr * 1.2
    r_mult = (current_price - entry_price) / max(risk, 1e-9)
    if r_mult >= float(config.scalper_tp_r):
        return True, "tp_r"
    high = max(highest_close, current_price)
    trail = high - 1.5 * atr
    if pnl > 0.01 and current_price <= trail:
        return True, "trail"
    hours = (now - entry_time).total_seconds() / 3600.0
    if hours >= float(config.scalper_time_stop_hours):
        return True, "time_stop"
    return False, ""
