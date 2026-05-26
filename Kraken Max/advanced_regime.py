from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from config import CONFIG, KrakenMaxConfig
from regime import RegimeEngine, RegimeState
from sentiment import SentimentSnapshot, adjust_deployment_cap


class HurstRegimeModel:
    def __init__(self, window: int = 5000) -> None:
        self.window = int(window)
        self._rets: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._last_close: dict[str, float] = {}
        self._h: dict[str, float] = {}

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

    def update(self, ticker: str, close: float) -> None:
        if close <= 0:
            return
        prev = self._last_close.get(ticker)
        self._last_close[ticker] = close
        if prev and prev > 0:
            self._rets[ticker].append(math.log(close / prev))
        if len(self._rets[ticker]) >= 500:
            self._h[ticker] = self.hurst_rs(np.array(self._rets[ticker], dtype=float))

    def hurst(self, ticker: str) -> float:
        return float(self._h.get(ticker, 0.5))

    def regime(self, ticker: str) -> str:
        h = self.hurst(ticker)
        if h > 0.55:
            return "trend"
        if h < 0.45:
            return "meanrev"
        return "random"


class VarianceRatioRegimeModel:
    TREND_THRESHOLD = 1.05
    MEANREV_THRESHOLD = 0.95

    def __init__(self, window: int = 5000, min_samples: int = 500) -> None:
        self.window = int(window)
        self.min_samples = int(min_samples)
        self._rets: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._last_close: dict[str, float] = {}
        self._vr: dict[str, float] = {}

    @staticmethod
    def _variance_ratio(log_returns: np.ndarray, k: int = 24) -> float:
        r = np.asarray(log_returns, dtype=float)
        n = len(r)
        if n < k + 2:
            return 1.0
        var1 = float(np.var(r, ddof=1))
        if var1 <= 0:
            return 1.0
        rolling = np.convolve(r, np.ones(k, dtype=float), mode="valid")
        vark = float(np.var(rolling, ddof=1)) if len(rolling) > 1 else var1
        return vark / (k * var1)

    def update(self, ticker: str, close: float) -> None:
        if close <= 0:
            return
        prev = self._last_close.get(ticker)
        self._last_close[ticker] = close
        if prev and prev > 0:
            self._rets[ticker].append(math.log(close / prev))
        if len(self._rets[ticker]) >= self.min_samples:
            arr = np.array(self._rets[ticker], dtype=float)
            self._vr[ticker] = 0.5 * (self._variance_ratio(arr, 6) + self._variance_ratio(arr, 24))

    def regime(self, ticker: str) -> str:
        vr = float(self._vr.get(ticker, 1.0))
        if vr > self.TREND_THRESHOLD:
            return "trend"
        if vr < self.MEANREV_THRESHOLD:
            return "meanrev"
        return "random"


class AdvancedRegimeEngine(RegimeEngine):
    """v4: Hurst + variance-ratio micro-regime layered on v3 macro regime."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        super().__init__(config)
        self.hurst = HurstRegimeModel()
        self.vr = VarianceRatioRegimeModel()
        self._btc_below_ema200 = False

    def update_btc_bar(self, close: float, ema200: float) -> None:
        if close > 0 and ema200 > 0:
            self._btc_below_ema200 = close < ema200
        self.hurst.update("BTCUSD", close)
        self.vr.update("BTCUSD", close)

    def classify_advanced(
        self,
        *,
        btc_features: dict[str, float],
        breadth: float,
        median_rv: float,
        sentiment: SentimentSnapshot | None = None,
    ) -> RegimeState:
        base = self.classify(
            btc_features=btc_features,
            breadth=breadth,
            median_rv=median_rv,
            sentiment=sentiment,
        )
        h_reg = self.hurst.regime("BTCUSD")
        vr_reg = self.vr.regime("BTCUSD")

        cap = base.deployment_cap
        allow_scalper = base.allow_scalper
        micro = base.micro_regime

        if h_reg == "meanrev" and vr_reg == "meanrev":
            allow_scalper = True
            micro = "meanrev"
            if base.name == "bull":
                cap *= 0.92
        elif h_reg == "trend" and vr_reg == "trend" and base.name == "neutral":
            cap = min(0.99, cap * 1.05)
            micro = "trend"
            allow_scalper = False
        elif self._btc_below_ema200 and base.name != "chaos":
            cap *= 0.88
            micro = "below_ema200"

        if sentiment and base.name == "bull":
            cap = adjust_deployment_cap(cap, sentiment, "bull", self.config)

        return RegimeState(
            name=base.name,
            deployment_cap=max(0.0, min(0.99, cap)),
            allow_new_entries=base.allow_new_entries,
            prefer_symbols=base.prefer_symbols,
            allow_scalper=allow_scalper,
            micro_regime=f"{micro}|H{h_reg}|VR{vr_reg}",
        )
