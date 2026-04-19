from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import timedelta

try:  # pragma: no cover
    from AlgorithmImports import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


class CvdDivergenceSignal:
    def __init__(self) -> None:
        self._tick_state = defaultdict(lambda: {"prev_price": None, "last_sign": 1.0})
        self._cvd_events = defaultdict(deque)
        self._hourly = defaultdict(lambda: deque(maxlen=24 * 14))
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

    def update(self, symbol, bar_or_tick) -> None:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        ts = getattr(bar_or_tick, "EndTime", getattr(bar_or_tick, "Time", None))
        if ts is None:
            return
        px = float(getattr(bar_or_tick, "LastPrice", getattr(bar_or_tick, "Price", getattr(bar_or_tick, "Close", 0.0))) or 0.0)
        qty = float(getattr(bar_or_tick, "Quantity", getattr(bar_or_tick, "Volume", 0.0)) or 0.0)
        high = float(getattr(bar_or_tick, "High", px) or px)
        low = float(getattr(bar_or_tick, "Low", px) or px)
        close = float(getattr(bar_or_tick, "Close", px) or px)
        open_ = float(getattr(bar_or_tick, "Open", close) or close)
        if px <= 0:
            return

        self._on_trade(key, ts, px, qty if qty > 0 else 1.0)
        if hasattr(bar_or_tick, "Close"):
            if close > open_:
                sign = 1.0
            elif close < open_:
                sign = -1.0
            else:
                sign = self._tick_state[key]["last_sign"]
            if qty > 0:
                self._cvd_events[key].append((ts, sign * qty))
            self._trim(key, ts)
            cvd_now = self._rolling_cvd(key)
            self._hourly[key].append((ts, high, low, cvd_now))
            self._scores[key] = self._compute_score(key)

    def _compute_score(self, key: str) -> float:
        hist = self._hourly[key]
        if len(hist) < 25:
            return 0.0
        current = hist[-1]
        prev = list(hist)[:-1]
        if not prev:
            return 0.0
        _, cur_high, cur_low, cur_cvd = current
        hi_p = max(x[1] for x in prev)
        lo_p = min(x[2] for x in prev)
        hi_cvd = max(x[3] for x in prev)
        lo_cvd = min(x[3] for x in prev)
        price_high_new = cur_high > hi_p
        cvd_high_new = cur_cvd > hi_cvd
        price_low_new = cur_low < lo_p
        cvd_low_new = cur_cvd < lo_cvd
        if price_high_new and not cvd_high_new:
            return -1.0
        if price_low_new and not cvd_low_new:
            return 1.0
        p_span = max(hi_p - lo_p, 1e-9)
        c_span = max(hi_cvd - lo_cvd, 1e-9)
        p_z = (cur_high - lo_p) / p_span
        c_z = (cur_cvd - lo_cvd) / c_span
        return self._clamp(math.tanh((c_z - p_z) * 2.0))

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))
