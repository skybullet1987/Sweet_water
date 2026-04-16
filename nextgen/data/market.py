from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable

from nextgen.core.types import Bar


class InMemoryBarStore:
    """Simple market data scaffold for bounded per-symbol bar storage."""

    def __init__(self, max_bars: int = 2048) -> None:
        self._bars: Dict[str, Deque[Bar]] = defaultdict(lambda: deque(maxlen=max_bars))

    def append(self, bar: Bar) -> None:
        self._bars[bar.symbol].append(bar)

    def latest(self, symbol: str) -> Bar | None:
        values = self._bars.get(symbol)
        return values[-1] if values else None

    def iter_symbol(self, symbol: str) -> Iterable[Bar]:
        return tuple(self._bars.get(symbol, ()))
