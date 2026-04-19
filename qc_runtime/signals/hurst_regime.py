from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

try:  # pragma: no cover
    from AlgorithmImports import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


class HurstRegimeSignal:
    def __init__(self, window: int = 500) -> None:
        self.window = int(window)
        self._rets = defaultdict(lambda: deque(maxlen=self.window))
        self._last_close = {}
        self._h = {}

    @staticmethod
    def hurst_rs(log_returns: np.ndarray) -> float:
        r = np.asarray(log_returns, dtype=float)
        n = len(r)
        if n < 20:
            return 0.5
        mean = float(np.mean(r))
        dev = r - mean
        z = np.cumsum(dev)
        R = float(np.max(z) - np.min(z))
        S = float(np.std(r, ddof=1))
        if S <= 0 or R <= 0:
            return 0.5
        return max(0.0, min(1.0, math.log(R / S) / math.log(n)))

    def update(self, symbol, bar_or_tick) -> None:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        close = float(getattr(bar_or_tick, "Close", getattr(bar_or_tick, "Price", 0.0)) or 0.0)
        if close <= 0:
            return
        prev = self._last_close.get(key)
        self._last_close[key] = close
        if prev and prev > 0:
            self._rets[key].append(math.log(close / prev))
        if len(self._rets[key]) >= self.window:
            self._h[key] = self.hurst_rs(np.array(self._rets[key], dtype=float))

    def score(self, symbol) -> float:
        return 0.0

    def hurst(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._h.get(key, 0.5))

    def regime(self, symbol) -> str:
        h = self.hurst(symbol)
        if h > 0.55:
            return "trend"
        if h < 0.45:
            return "meanrev"
        return "random"
