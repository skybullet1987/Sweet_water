from __future__ import annotations

import math
from collections import deque

try:  # pragma: no cover
    from AlgorithmImports import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


class BtcDominanceRotationSignal:
    def __init__(self, tracked_symbols: list[str] | None = None) -> None:
        self._tracked = set(tracked_symbols or [])
        self._base = {}
        self._latest = {}
        self._proxy_history = deque(maxlen=24 * 15)
        self._delta_history = deque(maxlen=24 * 14)
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
        tracked = [s for s in self._tracked if s in self._latest]
        if not tracked:
            tracked = list(self._latest.keys())
        denom = 0.0
        for s in tracked:
            base = max(self._base.get(s, self._latest[s]), 1e-9)
            denom += self._latest[s] / base
        if denom <= 0:
            return
        btc_norm = self._latest["BTCUSD"] / max(self._base.get("BTCUSD", self._latest["BTCUSD"]), 1e-9)
        proxy = btc_norm / denom
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

    def score(self, symbol) -> float:
        key = symbol.Value if hasattr(symbol, "Value") else str(symbol)
        return float(self._scores.get(key, 0.0))
