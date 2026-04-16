from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Deque, Dict

from nextgen.core.types import Bar, FeatureOutput


class BasicFeatureEngine:
    """Small feature scaffold that computes basic momentum and volatility proxies."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback
        self._closes: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=lookback))

    def update(self, bar: Bar) -> FeatureOutput:
        closes = self._closes[bar.symbol]
        closes.append(bar.close)

        momentum = 0.0
        realized_vol = 0.0
        if len(closes) >= 2:
            momentum = (closes[-1] / closes[0]) - 1.0
            returns = [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))]
            realized_vol = sum(abs(r) for r in returns) / len(returns)

        values = {
            "momentum": float(momentum),
            "trend_strength": float(momentum),
            "mean_reversion_score": float(-momentum),
            "realized_vol": float(realized_vol),
            "liquidity": 1.0 if bar.volume > 0 else 0.0,
            "breadth": float(momentum),
        }
        return FeatureOutput(symbol=bar.symbol, timestamp=bar.timestamp if isinstance(bar.timestamp, datetime) else datetime.now(UTC), values=values)
