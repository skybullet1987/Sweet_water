from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore

    HAS_TALIB = True
except Exception:  # pragma: no cover - import-time environment dependent
    talib = None
    HAS_TALIB = False


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    upper = name.upper()
    lower = name.lower()
    for key in (upper, lower):
        if key in df.columns:
            return df[key].astype(float)
    raise KeyError(f"Missing required column '{name}'")


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    close = _col(df, "close")
    if HAS_TALIB:
        return pd.Series(talib.EMA(close.values, timeperiod=period), index=df.index)
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = _col(df, "close")
    if HAS_TALIB:
        return pd.Series(talib.RSI(close.values, timeperiod=period), index=df.index)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = _col(df, "high"), _col(df, "low"), _col(df, "close")
    if HAS_TALIB:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=df.index)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = _col(df, "high"), _col(df, "low"), _col(df, "close")
    if HAS_TALIB:
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=df.index)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = atr(df, period=1)
    atr_n = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_n.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_n.replace(0.0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = _col(df, "close")
    if HAS_TALIB:
        m, s, h = talib.MACD(close.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.DataFrame({"macd": m, "signal": s, "hist": h}, index=df.index)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=df.index)


def bbands(df: pd.DataFrame, period: int = 20, stdev: float = 2.0) -> pd.DataFrame:
    close = _col(df, "close")
    if HAS_TALIB:
        u, m, l = talib.BBANDS(close.values, timeperiod=period, nbdevup=stdev, nbdevdn=stdev)
        return pd.DataFrame({"upper": u, "middle": m, "lower": l}, index=df.index)
    mid = close.rolling(window=period, min_periods=period).mean()
    sd = close.rolling(window=period, min_periods=period).std()
    return pd.DataFrame({"upper": mid + stdev * sd, "middle": mid, "lower": mid - stdev * sd}, index=df.index)


def cmo(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = _col(df, "close")
    if HAS_TALIB:
        return pd.Series(talib.CMO(close.values, timeperiod=period), index=df.index)
    diff = close.diff()
    up = diff.clip(lower=0.0).rolling(period, min_periods=period).sum()
    down = (-diff.clip(upper=0.0)).rolling(period, min_periods=period).sum()
    return 100 * (up - down) / (up + down).replace(0.0, np.nan)


def aroon(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low = _col(df, "high"), _col(df, "low")
    if HAS_TALIB:
        up, down = talib.AROON(high.values, low.values, timeperiod=period)
        return pd.DataFrame({"aroon_up": up, "aroon_down": down, "oscillator": up - down}, index=df.index)

    def _aroon_up(x: np.ndarray) -> float:
        return 100.0 * (np.argmax(x) + 1) / len(x)

    def _aroon_down(x: np.ndarray) -> float:
        return 100.0 * (np.argmin(x) + 1) / len(x)

    up = high.rolling(period, min_periods=period).apply(_aroon_up, raw=True)
    down = low.rolling(period, min_periods=period).apply(_aroon_down, raw=True)
    return pd.DataFrame({"aroon_up": up, "aroon_down": down, "oscillator": up - down}, index=df.index)


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close, volume = _col(df, "high"), _col(df, "low"), _col(df, "close"), _col(df, "volume")
    if HAS_TALIB:
        return pd.Series(talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period), index=df.index)
    tp = (high + low + close) / 3.0
    money_flow = tp * volume
    sign = np.sign(tp.diff().fillna(0.0))
    pos = money_flow.where(sign > 0, 0.0).rolling(period, min_periods=period).sum()
    neg = money_flow.where(sign < 0, 0.0).rolling(period, min_periods=period).sum().abs()
    mfr = pos / neg.replace(0.0, np.nan)
    return 100 - (100 / (1 + mfr))


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high, low, close = _col(df, "high"), _col(df, "low"), _col(df, "close")
    if HAS_TALIB:
        return pd.Series(talib.CCI(high.values, low.values, close.values, timeperiod=period), index=df.index)
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0.0, np.nan))
