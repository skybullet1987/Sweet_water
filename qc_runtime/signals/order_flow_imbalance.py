from __future__ import annotations

import math
from collections import defaultdict, deque

try:  # pragma: no cover
    from AlgorithmImports import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


class OrderFlowImbalanceSignal:
    def __init__(self) -> None:
        self._last_quote = {}
        self._hour_key = {}
        self._hour_acc = defaultdict(float)
        self._hourly = defaultdict(lambda: deque(maxlen=24 * 30))
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
        if bid_p > pbp:
            bid_term = bid_q
        elif bid_p == pbp:
            bid_term = bid_q - pbq
        else:
            bid_term = -pbq
        if ask_p < pap:
            ask_term = ask_q
        elif ask_p == pap:
            ask_term = ask_q - paq
        else:
            ask_term = -paq
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
        std = math.sqrt(max(var, 1e-12))
        z = (cur - mean) / std
        return self._clamp(math.tanh(z / 2.0))

    def update(self, symbol, bar_or_tick) -> None:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        ts = getattr(bar_or_tick, "EndTime", getattr(bar_or_tick, "Time", None))
        if ts is None:
            return
        hour_key = (ts.year, ts.month, ts.day, ts.hour)
        self._roll_hour(key, hour_key)

        bid_p = float(getattr(bar_or_tick, "BidPrice", 0.0) or 0.0)
        ask_p = float(getattr(bar_or_tick, "AskPrice", 0.0) or 0.0)
        bid_q = float(getattr(bar_or_tick, "BidSize", 0.0) or 0.0)
        ask_q = float(getattr(bar_or_tick, "AskSize", 0.0) or 0.0)
        if bid_p > 0 and ask_p > 0 and (bid_q > 0 or ask_q > 0):
            self._fallback_quote_bar = False
            self._hour_acc[key] += self._event_ofi(key, bid_p, bid_q, ask_p, ask_q)
            return

        if hasattr(bar_or_tick, "Open") and hasattr(bar_or_tick, "Close"):
            open_ = float(getattr(bar_or_tick, "Open", 0.0) or 0.0)
            close = float(getattr(bar_or_tick, "Close", 0.0) or 0.0)
            vol = float(getattr(bar_or_tick, "Volume", 0.0) or 0.0)
            self._hour_acc[key] += (1.0 if close >= open_ else -1.0) * vol

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))

    def using_fallback(self) -> bool:
        return bool(self._fallback_quote_bar)
