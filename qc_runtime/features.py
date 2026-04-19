from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore

    HAS_TALIB = True
except Exception:  # pragma: no cover
    talib = None
    HAS_TALIB = False

DEFAULT_LOOKBACK_BARS = 300


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


def _cmo(close: pd.Series, period: int = 14) -> pd.Series:
    up = close.diff().clip(lower=0).rolling(period).sum()
    down = (-close.diff().clip(upper=0)).rolling(period).sum()
    return 100 * (up - down) / (close.diff().abs().rolling(period).sum().replace(0.0, np.nan))


def _aroon_osc(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    return (high.rolling(period).apply(np.argmax, raw=True) - low.rolling(period).apply(np.argmin, raw=True)) * 100 / period


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    pos = money_flow.where(typical.diff() > 0, 0.0).rolling(period).sum()
    neg = money_flow.where(typical.diff() < 0, 0.0).rolling(period).sum().abs()
    return 100 - (100 / (1 + (pos / neg.replace(0.0, np.nan))))


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (typical - sma) / (0.015 * mad.replace(0.0, np.nan))


class FeatureEngine:
    def __init__(self, lookback: int = DEFAULT_LOOKBACK_BARS) -> None:
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
        macd_line = _ema(close, 12) - _ema(close, 26)
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
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
            "macd_hist": float((macd_line - macd_signal).iloc[-1]),
            "bb_pos": float((close.iloc[-1] - lower.iloc[-1]) / width),
            "cmo": float(_cmo(close, 14).iloc[-1]),
            "aroon_osc": float(_aroon_osc(high, low, 14).iloc[-1]),
            "mfi": float(_mfi(high, low, close, volume, 14).iloc[-1]),
            "cci": float(_cci(high, low, close, 20).iloc[-1]),
            "amihud": float(amihud_illiquidity(ret, (close * volume), 20).iloc[-1]),
            "roll_spread": float(roll_spread(close, 20).iloc[-1]),
            "kyle_lambda": float(kyle_lambda_proxy(ret, signed_volume, 20).iloc[-1]),
            "realized_vol": float(realized_vol(ret, 24).iloc[-1]),
            "realized_vol_30": float(realized_vol(ret, 30).iloc[-1]),
            "ofi": float(ofi_proxy(frame["open"], high, low, close, volume).rolling(20, min_periods=20).mean().iloc[-1]),
            "mom_24": float((close.iloc[-1] / close.iloc[-24]) - 1.0),
            "mom_168": float((close.iloc[-1] / close.iloc[-168]) - 1.0) if len(close) >= 168 else 0.0,
            "ema20": float(_ema(close, 20).iloc[-1]),
            "ema50": float(_ema(close, 50).iloc[-1]),
            "ema200": float(_ema(close, 200).iloc[-1]),
        }

    def current_features(self, symbol: str) -> dict[str, float]:
        return dict(self._features.get(symbol, {}))


class CvdDivergenceFeature:
    def __init__(self) -> None:
        self._tick_state = defaultdict(lambda: {"prev_price": None, "last_sign": 1.0})
        self._cvd_events = defaultdict(deque)
        self._hourly = defaultdict(lambda: deque(maxlen=24 * 14))
        self._cvd = defaultdict(float)
        self._scores: dict[str, float] = {}

    @staticmethod
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def _trim(self, key: str, now) -> None:
        cutoff = now - timedelta(hours=24)
        dq = self._cvd_events[key]
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _on_trade(self, key: str, ts, price: float, qty: float) -> None:
        state = self._tick_state[key]
        prev = state["prev_price"]
        sign = state["last_sign"]
        if prev is not None:
            if price > prev:
                sign = 1.0
            elif price < prev:
                sign = -1.0
        state["prev_price"] = price
        state["last_sign"] = sign
        self._cvd_events[key].append((ts, sign * max(float(qty), 0.0)))
        self._trim(key, ts)

    def _rolling_cvd(self, key: str) -> float:
        return float(sum(v for _, v in self._cvd_events[key]))

    def _compute_score(self, key: str) -> float:
        hist = self._hourly[key]
        if len(hist) < 25:
            return 0.0
        _, cur_high, cur_low, cur_cvd = hist[-1]
        prev = list(hist)[:-1]
        hi_p, lo_p = max(x[1] for x in prev), min(x[2] for x in prev)
        hi_cvd, lo_cvd = max(x[3] for x in prev), min(x[3] for x in prev)
        if cur_high > hi_p and cur_cvd <= hi_cvd:
            return -1.0
        if cur_low < lo_p and cur_cvd >= lo_cvd:
            return 1.0
        p_z = (cur_high - lo_p) / max(hi_p - lo_p, 1e-9)
        c_z = (cur_cvd - lo_cvd) / max(hi_cvd - lo_cvd, 1e-9)
        return self._clamp(math.tanh((c_z - p_z) * 2.0))

    def update(self, symbol, bar_or_tick) -> None:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        ts = getattr(bar_or_tick, "EndTime", getattr(bar_or_tick, "Time", None))
        if ts is None:
            return
        px = float(getattr(bar_or_tick, "LastPrice", getattr(bar_or_tick, "Price", getattr(bar_or_tick, "Close", 0.0))) or 0.0)
        if px <= 0:
            return
        qty = float(getattr(bar_or_tick, "Quantity", getattr(bar_or_tick, "Volume", 0.0)) or 0.0)
        self._on_trade(key, ts, px, qty if qty > 0 else 1.0)
        self._cvd[key] = self._rolling_cvd(key)
        if hasattr(bar_or_tick, "Close"):
            open_ = float(getattr(bar_or_tick, "Open", getattr(bar_or_tick, "Close", px)) or px)
            close = float(getattr(bar_or_tick, "Close", px) or px)
            high = float(getattr(bar_or_tick, "High", px) or px)
            low = float(getattr(bar_or_tick, "Low", px) or px)
            sign = 1.0 if close > open_ else (-1.0 if close < open_ else self._tick_state[key]["last_sign"])
            if qty > 0:
                self._cvd_events[key].append((ts, sign * qty))
            self._trim(key, ts)
            cvd_now = self._rolling_cvd(key)
            self._cvd[key] = cvd_now
            self._hourly[key].append((ts, high, low, cvd_now))
            self._scores[key] = self._compute_score(key)

    def value(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._cvd.get(key, 0.0))

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))


class OrderFlowImbalanceFeature:
    def __init__(self) -> None:
        self._last_quote = {}
        self._hour_key = {}
        self._hour_acc = defaultdict(float)
        self._hourly = defaultdict(lambda: deque(maxlen=24 * 30))
        self._values = defaultdict(float)
        self._scores = {}
        self._fallback_quote_bar = True

    @staticmethod
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def _event_ofi(self, key: str, bid_p: float, bid_q: float, ask_p: float, ask_q: float) -> float:
        prev = self._last_quote.get(key)
        self._last_quote[key] = (bid_p, bid_q, ask_p, ask_q)
        if prev is None:
            return 0.0
        pbp, pbq, pap, paq = prev
        bid_term = bid_q if bid_p > pbp else (bid_q - pbq if bid_p == pbp else -pbq)
        ask_term = ask_q if ask_p < pap else (ask_q - paq if ask_p == pap else -paq)
        return float(bid_term - ask_term)

    def _roll_hour(self, key: str, hour_key) -> None:
        prev_hour = self._hour_key.get(key)
        if prev_hour is None:
            self._hour_key[key] = hour_key
            return
        if hour_key != prev_hour:
            self._hourly[key].append(self._hour_acc[key])
            self._hour_acc[key] = 0.0
            self._hour_key[key] = hour_key
            self._scores[key] = self._compute_score(key)

    def _compute_score(self, key: str) -> float:
        hist = self._hourly[key]
        if len(hist) < 24:
            return 0.0
        cur = float(hist[-1])
        sample = list(hist)[-24 * 30 :]
        mean = sum(sample) / len(sample)
        var = sum((x - mean) ** 2 for x in sample) / max(len(sample) - 1, 1)
        z = (cur - mean) / math.sqrt(max(var, 1e-12))
        return self._clamp(math.tanh(z / 2.0))

    def update(self, symbol, bar_or_tick) -> None:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        ts = getattr(bar_or_tick, "EndTime", getattr(bar_or_tick, "Time", None))
        if ts is None:
            return
        self._roll_hour(key, (ts.year, ts.month, ts.day, ts.hour))
        bid_p = float(getattr(bar_or_tick, "BidPrice", 0.0) or 0.0)
        ask_p = float(getattr(bar_or_tick, "AskPrice", 0.0) or 0.0)
        bid_q = float(getattr(bar_or_tick, "BidSize", 0.0) or 0.0)
        ask_q = float(getattr(bar_or_tick, "AskSize", 0.0) or 0.0)
        if bid_p > 0 and ask_p > 0 and (bid_q > 0 or ask_q > 0):
            self._fallback_quote_bar = False
            self._hour_acc[key] += self._event_ofi(key, bid_p, bid_q, ask_p, ask_q)
            self._values[key] = self._hour_acc[key]
            return
        if hasattr(bar_or_tick, "Open") and hasattr(bar_or_tick, "Close"):
            open_ = float(getattr(bar_or_tick, "Open", 0.0) or 0.0)
            close = float(getattr(bar_or_tick, "Close", 0.0) or 0.0)
            vol = float(getattr(bar_or_tick, "Volume", 0.0) or 0.0)
            self._hour_acc[key] += (1.0 if close >= open_ else -1.0) * vol
            self._values[key] = self._hour_acc[key]

    def value(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._values.get(key, 0.0))

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))

    def using_fallback(self) -> bool:
        return bool(self._fallback_quote_bar)


class VolConeBreakoutFeature:
    def __init__(self) -> None:
        self._gk = defaultdict(lambda: deque(maxlen=24 * 30))
        self._pv = defaultdict(lambda: deque(maxlen=24 * 5))
        self._scores = {}
        self._last_pct = defaultdict(float)
        self._ranks = defaultdict(float)
        self._decay = defaultdict(int)
        self._decay_sign = defaultdict(float)

    @staticmethod
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    @staticmethod
    def gk_value(open_: float, high: float, low: float, close: float) -> float:
        if min(open_, high, low, close) <= 0:
            return 0.0
        a = 0.5 * (math.log(high / low) ** 2)
        b = (2.0 * math.log(2.0) - 1.0) * (math.log(close / open_) ** 2)
        return max(0.0, a - b)

    def update(self, symbol, bar_or_tick) -> None:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        if not hasattr(bar_or_tick, "Open"):
            return
        open_ = float(getattr(bar_or_tick, "Open", 0.0) or 0.0)
        high = float(getattr(bar_or_tick, "High", 0.0) or 0.0)
        low = float(getattr(bar_or_tick, "Low", 0.0) or 0.0)
        close = float(getattr(bar_or_tick, "Close", 0.0) or 0.0)
        vol = float(getattr(bar_or_tick, "Volume", 0.0) or 0.0)
        if min(open_, high, low, close) <= 0:
            return
        gk = self.gk_value(open_, high, low, close)
        hist = self._gk[key]
        prev = list(hist)
        hist.append(gk)
        self._pv[key].append((close * vol, vol))
        if len(prev) < 24:
            self._scores[key] = 0.0
            return
        rank = sum(1 for x in prev if x <= gk) / max(len(prev), 1)
        self._ranks[key] = rank
        crossed_up = self._last_pct[key] < 0.8 <= rank
        self._last_pct[key] = rank
        total_pv = sum(pv for pv, _ in self._pv[key])
        total_v = sum(v for _, v in self._pv[key])
        vwap = total_pv / total_v if total_v > 0 else close
        if crossed_up:
            s = 1.0 if close > vwap else -1.0
            self._scores[key] = s
            self._decay[key] = 12
            self._decay_sign[key] = s
            return
        if self._decay[key] > 0:
            self._decay[key] -= 1
            self._scores[key] = self._clamp(self._decay_sign[key] * (self._decay[key] / 12.0))
            return
        self._scores[key] = 0.0

    def percentile_rank(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._ranks.get(key, 0.0))

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))


class BtcDominanceRotationFeature:
    def __init__(self, tracked_symbols: list[str] | None = None) -> None:
        self._tracked = set(tracked_symbols or [])
        self._base = {}
        self._latest = {}
        self._proxy_history = deque(maxlen=24 * 15)
        self._delta_history = deque(maxlen=24 * 14)
        self._proxy = 0.0
        self._scores = {}

    @staticmethod
    def _clamp(x: float) -> float:
        return max(-1.0, min(1.0, float(x)))

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        self._tracked = set(symbols)

    def update(self, symbol, bar_or_tick) -> None:
        if not hasattr(bar_or_tick, "Close"):
            return
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        price = float(getattr(bar_or_tick, "Close", 0.0) or 0.0)
        if price <= 0:
            return
        if key not in self._base:
            self._base[key] = price
        self._latest[key] = price
        self._recompute()

    def _recompute(self) -> None:
        if "BTCUSD" not in self._latest:
            return
        tracked = [s for s in self._tracked if s in self._latest] or list(self._latest.keys())
        denom = 0.0
        for s in tracked:
            base = max(self._base.get(s, self._latest[s]), 1e-9)
            denom += self._latest[s] / base
        if denom <= 0:
            return
        btc_norm = self._latest["BTCUSD"] / max(self._base.get("BTCUSD", self._latest["BTCUSD"]), 1e-9)
        proxy = btc_norm / denom
        self._proxy = proxy
        self._proxy_history.append(proxy)
        if len(self._proxy_history) <= 24:
            return
        delta24 = proxy - list(self._proxy_history)[-25]
        self._delta_history.append(delta24)
        if len(self._delta_history) < 24:
            return
        sample = list(self._delta_history)
        mean = sum(sample) / len(sample)
        var = sum((x - mean) ** 2 for x in sample) / max(len(sample) - 1, 1)
        z = (delta24 - mean) / math.sqrt(max(var, 1e-12))
        alt_score = self._clamp(-z)
        for s in tracked:
            self._scores[s] = -0.5 * alt_score if s == "BTCUSD" else alt_score

    def proxy(self) -> float:
        return float(self._proxy)

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))


class SignalFeatureStack:
    def __init__(self, algo=None, tracked_symbols: list[str] | None = None) -> None:
        _ = algo
        self.cvd = CvdDivergenceFeature()
        self.ofi = OrderFlowImbalanceFeature()
        self.vol_cone = VolConeBreakoutFeature()
        self.btc_rotation = BtcDominanceRotationFeature(tracked_symbols or [])

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        self.btc_rotation.set_tracked_symbols(symbols)

    def update(self, symbol, bar_or_tick) -> None:
        self.cvd.update(symbol, bar_or_tick)
        self.ofi.update(symbol, bar_or_tick)
        self.vol_cone.update(symbol, bar_or_tick)
        self.btc_rotation.update(symbol, bar_or_tick)

    def component_scores(self, symbol) -> dict[str, float]:
        return {
            "cvd": self.cvd.score(symbol),
            "ofi": self.ofi.score(symbol),
            "volc": self.vol_cone.score(symbol),
            "rot": self.btc_rotation.score(symbol),
        }

    def init_status(self) -> dict[str, str]:
        return {
            "cvd": "trade-tick if available, bar fallback enabled",
            "ofi": "quote L1" if not self.ofi.using_fallback() else "quote-bar fallback",
            "vol_cone": "garman-klass",
            "btc_rotation": "active",
            "hurst": "R/S(500)",
            "vr": "variance-ratio(6h,24h)",
        }
