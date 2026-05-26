from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from config import CONFIG, KrakenMaxConfig


@dataclass
class FillStats:
    limits_submitted: int = 0
    limits_filled: int = 0
    market_filled: int = 0
    cancelled: int = 0
    slippage_bps: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def fill_rate(self) -> float:
        """Filled limits / non-cancelled submissions (cancels are intentional, not failed fills)."""
        attempted = self.limits_submitted - self.cancelled
        if attempted <= 0:
            return 1.0
        return float(self.limits_filled) / float(attempted)

    @property
    def avg_slippage_bps(self) -> float:
        if not self.slippage_bps:
            return 0.0
        return float(sum(self.slippage_bps)) / len(self.slippage_bps)


class FillTracker:
    """Track limit fill quality vs expected prices (v5)."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.stats = FillStats()
        self._pending: dict[int, dict] = {}

    def on_submit(self, order_id: int, *, is_limit: bool, expected_price: float, qty: float) -> None:
        if not is_limit:
            return
        self.stats.limits_submitted += 1
        self._pending[int(order_id)] = {
            "expected_price": float(expected_price),
            "qty": float(qty),
        }

    def on_fill(self, order_id: int, fill_price: float, *, is_limit: bool, tag: str = "") -> None:
        if is_limit:
            self.stats.limits_filled += 1
        else:
            self.stats.market_filled += 1
        meta = self._pending.pop(int(order_id), None)
        if meta and meta["expected_price"] > 0:
            exp = float(meta["expected_price"])
            slip_bps = abs(float(fill_price) / exp - 1.0) * 10_000.0
            self.stats.slippage_bps.append(slip_bps)

    def on_cancel(self, order_id: int) -> None:
        self.stats.cancelled += 1
        self._pending.pop(int(order_id), None)

    def should_alert(self) -> tuple[bool, str]:
        if self.stats.limits_submitted < 5:
            return False, ""
        if self.stats.fill_rate < float(self.config.fill_rate_alert_threshold):
            return True, f"low_fill_rate={self.stats.fill_rate:.2%}"
        if self.stats.avg_slippage_bps > float(self.config.slippage_alert_bps):
            return True, f"high_slippage_bps={self.stats.avg_slippage_bps:.1f}"
        return False, ""

    def summary(self) -> dict[str, float]:
        return {
            "fill_rate": self.stats.fill_rate,
            "avg_slippage_bps": self.stats.avg_slippage_bps,
            "limits_submitted": float(self.stats.limits_submitted),
            "limits_filled": float(self.stats.limits_filled),
        }
