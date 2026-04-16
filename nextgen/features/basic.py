from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Deque, Dict, List, Optional, Tuple

from nextgen.core.types import Bar, FeatureOutput

# Annualisation factor for 5-minute bars: 12 bars/hour × 24 h × 365 days
_BARS_PER_YEAR_5MIN = 12 * 24 * 365
_SQRT_ANNUALISE_5MIN = math.sqrt(_BARS_PER_YEAR_5MIN)


def _ema_update(prev: Optional[float], value: float, alpha: float) -> float:
    """Single-step exponential moving average update."""
    if prev is None:
        return value
    return (1.0 - alpha) * prev + alpha * value


def _rolling_std(window: deque) -> float:
    """Population std of a deque (Bessel-corrected if len >= 2)."""
    n = len(window)
    if n < 2:
        return 0.0
    mean = sum(window) / n
    var = sum((x - mean) ** 2 for x in window) / (n - 1)
    return math.sqrt(max(var, 0.0))


class _SymbolState:
    """Per-symbol incremental accumulators."""

    __slots__ = (
        "closes", "returns", "highs", "lows", "volumes",
        "returns_short", "returns_long",
        "ema_short", "ema_long",
        "rsi_gains", "rsi_losses",
        "vwap_pv", "vwap_v",
    )

    def __init__(self, short: int, long: int, rsi: int) -> None:
        self.closes: deque = deque(maxlen=max(long, 2))
        self.returns: deque = deque(maxlen=long)
        self.highs: deque = deque(maxlen=long)
        self.lows: deque = deque(maxlen=long)
        self.volumes: deque = deque(maxlen=long)
        self.returns_short: deque = deque(maxlen=short)
        self.returns_long: deque = deque(maxlen=long)
        self.ema_short: Optional[float] = None
        self.ema_long: Optional[float] = None
        self.rsi_gains: deque = deque(maxlen=rsi)
        self.rsi_losses: deque = deque(maxlen=rsi)
        self.vwap_pv: List[float] = []
        self.vwap_v: List[float] = []


class BasicFeatureEngine:
    """
    Feature engine computing real technical indicators from Bar data.

    Features produced (all float, no LEAN dependency):
      momentum_short     – return over `short_window` bars
      momentum_long      – return over `long_window` bars
      trend_strength     – EMA-ratio signal (ema_short/ema_long − 1), bounded
      mean_reversion_score – negative of VWAP deviation, bounded
      realized_vol_short – annualised rolling return std over `vol_short` bars
      realized_vol_long  – annualised rolling return std over `vol_long` bars
      vol_ratio          – realized_vol_short / realized_vol_long (regime signal)
      rsi_norm           – RSI(rsi_window) normalised to [−1, +1]
      vwap_deviation     – (close − VWAP) / VWAP
      volume_ratio       – bar volume / rolling mean volume (short window)
      atr_proxy          – average |Δclose| / close over last `vol_short` bars
      liquidity          – 1 if bar.volume > 0 else 0
      breadth            – 1 if ema_short > ema_long else 0 (single-symbol proxy)
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        ema_short_bars: int = 6,
        ema_long_bars: int = 20,
        rsi_window: int = 7,
        # Backward-compatible alias: lookback maps to long_window.
        lookback: Optional[int] = None,
    ) -> None:
        if lookback is not None:
            long_window = lookback
            short_window = min(short_window, lookback)
        self.short_window = short_window
        self.long_window = long_window
        self.ema_short_bars = ema_short_bars
        self.ema_long_bars = ema_long_bars
        self.rsi_window = rsi_window
        self._alpha_short = 2.0 / (ema_short_bars + 1)
        self._alpha_long = 2.0 / (ema_long_bars + 1)
        self._states: Dict[str, _SymbolState] = {}

    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            self._states[symbol] = _SymbolState(self.short_window, self.long_window, self.rsi_window)
        return self._states[symbol]

    def update(self, bar: Bar) -> FeatureOutput:
        s = self._get_state(bar.symbol)
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        volume = float(bar.volume)

        # Compute return before appending new close
        ret = 0.0
        if s.closes:
            prev_close = s.closes[-1]
            if prev_close > 0:
                ret = (close - prev_close) / prev_close

        s.closes.append(close)
        s.highs.append(high)
        s.lows.append(low)
        s.volumes.append(volume)

        if ret != 0.0 or len(s.closes) > 1:
            s.returns.append(ret)
            s.returns_short.append(ret)
            s.returns_long.append(ret)

        # EMAs
        s.ema_short = _ema_update(s.ema_short, close, self._alpha_short)
        s.ema_long = _ema_update(s.ema_long, close, self._alpha_long)

        # RSI components
        gain = max(ret, 0.0)
        loss = abs(min(ret, 0.0))
        s.rsi_gains.append(gain)
        s.rsi_losses.append(loss)

        # VWAP accumulator (daily; no reset here — reset externally if desired)
        s.vwap_pv.append(close * volume)
        s.vwap_v.append(volume)
        # Safety cap at 576 bars (2 days of 5-min bars)
        if len(s.vwap_pv) > 576:
            s.vwap_pv = s.vwap_pv[-576:]
            s.vwap_v = s.vwap_v[-576:]

        # ── Compute features ──────────────────────────────────────────────

        # Momentum: multi-window returns
        n_closes = len(s.closes)
        momentum_short = 0.0
        momentum_long = 0.0
        if n_closes >= 2:
            oldest_short = s.closes[max(0, n_closes - self.short_window - 1)]
            oldest_long = s.closes[0]
            if oldest_short > 0:
                momentum_short = (close / oldest_short) - 1.0
            if oldest_long > 0:
                momentum_long = (close / oldest_long) - 1.0

        # Realized volatility (annualised)
        rv_short = _rolling_std(s.returns_short) * _SQRT_ANNUALISE_5MIN
        rv_long = _rolling_std(s.returns_long) * _SQRT_ANNUALISE_5MIN
        vol_ratio = rv_short / rv_long if rv_long > 1e-9 else 1.0

        # ATR proxy: average of |Δclose| / close over short window
        atr_proxy = rv_short / _SQRT_ANNUALISE_5MIN  # per-bar vol estimate

        # EMA trend signal
        ema_ratio = 0.0
        trend_strength = 0.0
        breadth = 0.5
        if s.ema_short is not None and s.ema_long is not None and s.ema_long > 0:
            ema_ratio = (s.ema_short / s.ema_long) - 1.0
            trend_strength = max(-1.0, min(1.0, ema_ratio * 10.0))
            breadth = 1.0 if s.ema_short > s.ema_long else 0.0

        # RSI normalised to [−1, +1]
        rsi_norm = 0.0
        if len(s.rsi_gains) >= self.rsi_window:
            avg_gain = sum(s.rsi_gains) / len(s.rsi_gains)
            avg_loss = sum(s.rsi_losses) / len(s.rsi_losses)
            if avg_gain + avg_loss > 1e-12:
                rsi = avg_gain / (avg_gain + avg_loss)  # [0, 1]
                rsi_norm = (rsi - 0.5) * 2.0             # [−1, +1]

        # VWAP deviation
        vwap = 0.0
        vwap_deviation = 0.0
        total_v = sum(s.vwap_v)
        if total_v > 0:
            vwap = sum(s.vwap_pv) / total_v
            if vwap > 0:
                vwap_deviation = (close - vwap) / vwap

        # Mean reversion score: fade deviation from VWAP
        mean_reversion_score = max(-1.0, min(1.0, -vwap_deviation * 20.0))

        # Volume ratio: bar volume vs rolling mean
        volume_ratio = 1.0
        if s.volumes and sum(s.volumes) > 0:
            mean_vol = sum(s.volumes) / len(s.volumes)
            if mean_vol > 0:
                volume_ratio = volume / mean_vol

        # Breakout strength: vol expansion in the direction of momentum
        breakout_strength = max(-1.0, min(1.0, momentum_short * max(0.0, vol_ratio - 1.0) * 5.0))

        # Pullback depth: short-term reversal against longer trend
        pullback_depth = 0.0
        if momentum_long > 0 and momentum_short < 0:
            pullback_depth = abs(momentum_short)
        elif momentum_long < 0 and momentum_short > 0:
            pullback_depth = abs(momentum_short)

        values = {
            "momentum": float(momentum_short),
            "momentum_short": float(momentum_short),
            "momentum_long": float(momentum_long),
            "trend_strength": float(trend_strength),
            "mean_reversion_score": float(mean_reversion_score),
            "realized_vol": float(rv_short),
            "realized_vol_short": float(rv_short),
            "realized_vol_long": float(rv_long),
            "vol_ratio": float(vol_ratio),
            "atr_proxy": float(atr_proxy),
            "rsi_norm": float(rsi_norm),
            "vwap_deviation": float(vwap_deviation),
            "volume_ratio": float(volume_ratio),
            "breakout_strength": float(breakout_strength),
            "pullback_depth": float(pullback_depth),
            "breadth": float(breadth),
            "liquidity": 1.0 if volume > 0 else 0.0,
        }

        ts = bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now(UTC)
        return FeatureOutput(symbol=bar.symbol, timestamp=ts, values=values)

    def reset_vwap(self, symbol: str) -> None:
        """Reset the VWAP accumulator for a symbol (call at UTC midnight)."""
        if symbol in self._states:
            s = self._states[symbol]
            s.vwap_pv = []
            s.vwap_v = []
