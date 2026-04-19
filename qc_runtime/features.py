from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore

    HAS_TALIB = True
except Exception:  # pragma: no cover
    talib = None
    HAS_TALIB = False


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    return (returns.abs() / dollar_volume.replace(0.0, np.nan)).rolling(window, min_periods=window).mean()


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    delta = close.diff()
    cov = delta.rolling(window, min_periods=window).cov(delta.shift(1))
    out = pd.Series(np.nan, index=close.index, dtype=float)
    valid = cov < 0
    out.loc[valid] = 2.0 * np.sqrt((-cov.loc[valid]).clip(lower=0.0))
    return out


def kyle_lambda_proxy(returns: pd.Series, signed_volume: pd.Series, window: int = 20) -> pd.Series:
    cov = returns.rolling(window, min_periods=window).cov(signed_volume)
    var = signed_volume.rolling(window, min_periods=window).var()
    return cov / var.replace(0.0, np.nan)


def realized_vol(returns: pd.Series, window: int = 24) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(24 * 365)


def ofi_proxy(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    midpoint = (high + low) / 2.0
    return volume * (close - midpoint) / ((high - low).abs() + 1e-12)


def zscore_vs_universe(symbol_returns_df: pd.DataFrame) -> pd.DataFrame:
    mu = symbol_returns_df.mean(axis=1)
    sigma = symbol_returns_df.std(axis=1, ddof=0).replace(0.0, np.nan)
    return symbol_returns_df.sub(mu, axis=0).div(sigma, axis=0)


def rank_momentum(symbol_returns_df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    cumulative = (1.0 + symbol_returns_df).rolling(window=window, min_periods=window).apply(np.prod, raw=True) - 1.0
    return cumulative.rank(axis=1, pct=True, method="average")


def _ema(close: pd.Series, period: int) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.EMA(close.values, timeperiod=period), index=close.index)
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr(high, low, close, period=1).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().fillna(0.0)


class FeatureEngine:
    def __init__(self, lookback: int = 300) -> None:
        self.lookback = lookback
        self._bars: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=lookback))
        self._features: dict[str, dict[str, float]] = {}

    @staticmethod
    def _parse_bar(bar: Any) -> tuple[str, dict[str, float]]:
        if isinstance(bar, dict):
            symbol = str(bar.get("symbol"))
            return symbol, {k: float(bar[k]) for k in ("open", "high", "low", "close", "volume")}
        symbol = str(getattr(bar, "symbol", getattr(bar, "Symbol", "")))
        return symbol, {
            "open": float(getattr(bar, "open", getattr(bar, "Open"))),
            "high": float(getattr(bar, "high", getattr(bar, "High"))),
            "low": float(getattr(bar, "low", getattr(bar, "Low"))),
            "close": float(getattr(bar, "close", getattr(bar, "Close"))),
            "volume": float(getattr(bar, "volume", getattr(bar, "Volume"))),
        }

    def update(self, bar: Any) -> None:
        symbol, parsed = self._parse_bar(bar)
        if not symbol:
            return
        self._bars[symbol].append(parsed)
        frame = pd.DataFrame(self._bars[symbol])
        if len(frame) < 60:
            return
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        volume = frame["volume"].astype(float)
        ret = np.log(close).diff().fillna(0.0)
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        upper = bb_mid + 2.0 * bb_std
        lower = bb_mid - 2.0 * bb_std
        width = max(float((upper - lower).iloc[-1]), 1e-9)
        signed_volume = np.sign(ret.fillna(0.0)) * volume
        self._features[symbol] = {
            "rsi": float(_rsi(close, 14).iloc[-1]),
            "atr": float(_atr(high, low, close, 14).iloc[-1]),
            "adx": float(_adx(high, low, close, 14).iloc[-1]),
            "macd_hist": float((_ema(close, 12) - _ema(close, 26) - (_ema(close, 12) - _ema(close, 26)).ewm(span=9, adjust=False).mean()).iloc[-1]),
            "bb_pos": float((close.iloc[-1] - lower.iloc[-1]) / width),
            "cmo": float((100 * (close.diff().clip(lower=0).rolling(14).sum() - (-close.diff().clip(upper=0)).rolling(14).sum()) / (close.diff().abs().rolling(14).sum().replace(0.0, np.nan))).iloc[-1]),
            "aroon_osc": float(((high.rolling(14).apply(np.argmax, raw=True) - low.rolling(14).apply(np.argmin, raw=True)) * 100 / 14).iloc[-1]),
            "mfi": float((100 - (100 / (1 + (((((high + low + close) / 3.0) * volume).where(((high + low + close) / 3.0).diff() > 0, 0.0).rolling(14).sum()) / ((((high + low + close) / 3.0) * volume).where(((high + low + close) / 3.0).diff() < 0, 0.0).rolling(14).sum().abs().replace(0.0, np.nan)))))).iloc[-1]),
            "cci": float((((high + low + close) / 3.0 - ((high + low + close) / 3.0).rolling(20).mean()) / (0.015 * ((high + low + close) / 3.0).rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).replace(0.0, np.nan))).iloc[-1]),
            "amihud": float(amihud_illiquidity(ret, (close * volume), 20).iloc[-1]),
            "roll_spread": float(roll_spread(close, 20).iloc[-1]),
            "kyle_lambda": float(kyle_lambda_proxy(ret, signed_volume, 20).iloc[-1]),
            "realized_vol": float(realized_vol(ret, 24).iloc[-1]),
            "ofi": float(ofi_proxy(frame["open"], high, low, close, volume).rolling(20, min_periods=20).mean().iloc[-1]),
            "mom_24": float((close.iloc[-1] / close.iloc[-24]) - 1.0),
            "ema20": float(_ema(close, 20).iloc[-1]),
            "ema50": float(_ema(close, 50).iloc[-1]),
        }

    def current_features(self, symbol: str) -> dict[str, float]:
        return dict(self._features.get(symbol, {}))
