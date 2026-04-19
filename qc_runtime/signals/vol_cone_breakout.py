from __future__ import annotations

import math
from collections import defaultdict, deque

try:  # pragma: no cover
    from AlgorithmImports import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


class VolConeBreakoutSignal:
    def __init__(self) -> None:
        self._gk = defaultdict(lambda: deque(maxlen=24 * 30))
        self._pv = defaultdict(lambda: deque(maxlen=24 * 5))
        self._scores = {}
        self._last_pct = defaultdict(float)
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

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))
