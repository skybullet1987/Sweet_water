from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

try:
    from config import CONFIG, StrategyConfig
except ModuleNotFoundError:  # pragma: no cover
    from .config import CONFIG, StrategyConfig  # type: ignore


class HurstRegimeModel:
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


class RegimeEngine:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self._vol_window = deque(maxlen=30)
        self._state = "risk_on"
        self._probs = {"risk_on": 1.0, "risk_off": 0.0, "chop": 0.0}
        self.vol_stress = 0.0
        self._btc_below_ema = False
        self.hurst = HurstRegimeModel(window=500)

    def update(self, btc_return: float, btc_vol: float, breadth: float, btc_above_ema200: bool = False) -> None:
        self._btc_below_ema = not bool(btc_above_ema200)
        self._vol_window.append(float(btc_vol))
        if len(self._vol_window) >= 30:
            vols = np.asarray(self._vol_window, dtype=float)
            mu = float(np.mean(vols))
            sigma = float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
            z = (float(btc_vol) - mu) / max(sigma, 1e-9)
            self.vol_stress = float(1.0 / (1.0 + np.exp(-z)))
        else:
            self.vol_stress = 0.0
        if self.vol_stress > self.config.vol_stress_threshold:
            self._state = "risk_off"
        elif abs(float(btc_return)) < float(self.config.chop_return_threshold):
            self._state = "chop"
        else:
            self._state = "risk_on"
        self._probs = {"risk_on": 0.0, "risk_off": 0.0, "chop": 0.0}
        self._probs[self._state] = 1.0

    def current_state(self) -> str:
        return self._state

    def current_state_probs(self) -> dict[str, float]:
        return dict(self._probs)

    def gates_pass(self, breadth: float) -> bool:
        if self.vol_stress > self.config.vol_stress_threshold:
            return False
        if breadth < self.config.breadth_threshold:
            return False
        if self._btc_below_ema:
            return False
        return True
