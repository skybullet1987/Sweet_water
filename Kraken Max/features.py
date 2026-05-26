from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig

DEFAULT_LOOKBACK = 24 * 30 * 4


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_bar_features(frame: pd.DataFrame, config: KrakenMaxConfig = CONFIG) -> dict[str, float]:
    min_bars = max(60, int(config.feature_min_bars) // max(config.bph(), 1))
    if frame is None or len(frame) < min_bars:
        return {}
    df = frame.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return {}
        df[col] = df[col].astype(float)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    ret = close.pct_change().fillna(0.0)

    def _mom(hours: int) -> float:
        lookback = min(len(close) - 1, max(config.lookback_bars(hours), 1))
        base = float(close.iloc[-1 - lookback])
        if base <= 0:
            return 0.0
        return float(close.iloc[-1] / base - 1.0)

    mom_7d = _mom(7)
    mom_21d = _mom(21)
    mom_63d = _mom(63)
    mom_accel = mom_7d - mom_21d / 3.0

    rv_window = min(len(ret), config.lookback_bars(21))
    rv_21d = float(ret.tail(rv_window).std() * math.sqrt(24 * 365 * config.bph()))
    rv_21d_inv = 1.0 / max(rv_21d, 1e-6)

    ema_fast = _ema(close, config.lookback_bars(24))
    ema_slow = _ema(close, config.lookback_bars(24 * 5))
    trend_quality = float((ema_fast.iloc[-1] / ema_slow.iloc[-1]) - 1.0)

    donchian = float(high.tail(min(len(high), config.lookback_bars(20 * 24))).max())
    breakout_strength = float((close.iloc[-1] / donchian) - 1.0) if donchian > 0 else 0.0

    vol_med = float((close * volume).tail(min(len(close), config.lookback_bars(7 * 24))).median())
    vol_recent = float((close * volume).tail(min(len(close), config.lookback_bars(24))).median())
    volume_24h = vol_recent
    volume_surge = float(vol_recent / max(vol_med, 1e-9) - 1.0)

    rsi = float(_rsi(close).iloc[-1])
    rsi_pullback = max(0.0, (45.0 - rsi) / 45.0) if trend_quality > 0 else 0.0

    dd_63d = float((close.iloc[-1] / close.tail(min(len(close), config.lookback_bars(63 * 24))).max()) - 1.0)
    atr = float(_atr(high, low, close).iloc[-1])
    ema50 = float(_ema(close, 50).iloc[-1])
    ema200 = float(_ema(close, 200).iloc[-1])
    ret_1h = float(close.iloc[-1] / close.iloc[-max(2, 2)] - 1.0) if len(close) > 2 else 0.0
    look6 = min(len(close) - 1, config.lookback_bars(6))
    ret_6h = float(close.iloc[-1] / close.iloc[-1 - look6] - 1.0) if look6 > 0 else 0.0

    return {
        "mom_7d": mom_7d,
        "mom_21d": mom_21d,
        "mom_63d": mom_63d,
        "mom_accel": mom_accel,
        "rv_21d": rv_21d,
        "rv_21d_inv": rv_21d_inv,
        "trend_quality": trend_quality,
        "breakout_strength": breakout_strength,
        "volume_surge": volume_surge,
        "volume_24h": volume_24h,
        "rsi": rsi,
        "rsi_pullback": rsi_pullback,
        "dd_63d": dd_63d,
        "atr": atr,
        "close": float(close.iloc[-1]),
        "ema50": ema50,
        "ema200": ema200,
        "ret_1h": ret_1h,
        "ret_6h": ret_6h,
        "adx": 18.0,
    }


def btc_beta_vs(features: dict[str, float], btc_features: dict[str, float]) -> float:
    """Simple beta proxy: coin 7d momentum minus BTC 7d momentum."""
    return float(features.get("mom_7d", 0.0)) - float(btc_features.get("mom_7d", 0.0))


class FeatureCache:
    """Per-symbol rolling OHLCV cache for hourly Kraken bars."""

    def __init__(self, max_bars: int = DEFAULT_LOOKBACK) -> None:
        self.max_bars = max_bars
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_bars))

    def update(self, symbol_key: str, bar: Any) -> None:
        self._bars[symbol_key].append(
            {
                "time": bar.EndTime,
                "open": float(bar.Open),
                "high": float(bar.High),
                "low": float(bar.Low),
                "close": float(bar.Close),
                "volume": float(bar.Volume),
            }
        )

    def frame(self, symbol_key: str) -> pd.DataFrame:
        rows = list(self._bars.get(symbol_key, []))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def features(self, symbol_key: str) -> dict[str, float]:
        return compute_bar_features(self.frame(symbol_key))


def cross_section_ranks(feature_map: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    vals = [(s, float(f.get(key, 0.0))) for s, f in feature_map.items() if f]
    if not vals:
        return {}
    symbols, numbers = zip(*vals)
    series = pd.Series(numbers, index=symbols)
    ranks = series.rank(pct=True, method="average")
    return {str(s): float(ranks[s]) for s in ranks.index}
