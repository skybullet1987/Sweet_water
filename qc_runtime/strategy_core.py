from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class IndicatorState:
    opens: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    highs: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    lows: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    closes: deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    volumes: deque[float] = field(default_factory=lambda: deque(maxlen=5000))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": list(self.opens),
                "high": list(self.highs),
                "low": list(self.lows),
                "close": list(self.closes),
                "volume": list(self.volumes),
            }
        )


def initialize_symbol(algo, symbol) -> None:
    if not hasattr(algo, "crypto_data"):
        algo.crypto_data = {}
    algo.crypto_data[symbol] = IndicatorState()


def update_symbol_data(algo, symbol, bar, quote_bar=None) -> None:
    state: IndicatorState = algo.crypto_data[symbol]
    state.opens.append(float(bar.Open))
    state.highs.append(float(bar.High))
    state.lows.append(float(bar.Low))
    state.closes.append(float(bar.Close))
    state.volumes.append(float(bar.Volume))
