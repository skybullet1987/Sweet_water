from __future__ import annotations

import math
from typing import Sequence


def sma(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: Sequence[float], ddof: int = 1) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = sma(values)
    var = sum((v - mu) ** 2 for v in values) / max(n - ddof, 1)
    return math.sqrt(max(var, 0.0))


def z_score_20h(closes_20h: Sequence[float], current_close: float) -> float:
    """Bollinger z of current_close vs trailing 20-hour SMA/STD.
    Returns 0.0 if insufficient data."""
    if len(closes_20h) < 20:
        return 0.0
    window = list(closes_20h)[-20:]
    s = std(window)
    if s <= 0:
        return 0.0
    return (current_close - sma(window)) / s


def rsi_14(closes: Sequence[float]) -> float:
    """Wilder's RSI on the last 15 closes (needs 15 to compute one RSI)."""
    closes = list(closes)
    if len(closes) < 15:
        return 50.0
    closes = closes[-15:]
    gains = []
    losses = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    if avg_loss <= 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def ret_pct(start: float, end: float) -> float:
    if start <= 0:
        return 0.0
    return end / start - 1.0
