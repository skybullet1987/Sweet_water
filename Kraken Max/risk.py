from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from config import CONFIG, KrakenMaxConfig


@dataclass
class PositionRisk:
    entry_price: float
    entry_time: datetime
    entry_atr: float
    highest_close: float
    pyramid_count: int = 0
    predicted_score: float = 0.0
    strategy_owner: str = "momentum"  # momentum | scalper


class PortfolioRisk:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.peak_equity = 0.0
        self.halted_until: datetime | None = None
        self.orders_today = 0
        self._order_day = None

    def update_peak(self, equity: float) -> float:
        self.peak_equity = max(self.peak_equity, float(equity))
        if self.peak_equity <= 0:
            return 0.0
        return (float(equity) / self.peak_equity) - 1.0

    def drawdown_halted(self, now: datetime, drawdown: float) -> bool:
        if self.halted_until and now < self.halted_until:
            return True
        if drawdown <= float(self.config.drawdown_halt_pct):
            hours = int(self.config.drawdown_cooldown_hours)
            self.halted_until = now + timedelta(hours=hours)
            return True
        if self.halted_until and now >= self.halted_until:
            self.halted_until = None
        return False

    def can_place_order(self, now: datetime) -> bool:
        day = now.date()
        if self._order_day != day:
            self._order_day = day
            self.orders_today = 0
        return self.orders_today < int(self.config.max_orders_per_day)

    def record_order(self) -> None:
        self.orders_today += 1


def should_exit(
    state: PositionRisk,
    *,
    close: float,
    now: datetime,
    hard_stop_pct: float,
    catastrophic_stop_pct: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    chandelier_atr_mult: float,
    activate_trail_pct: float,
    time_stop_hours: float,
) -> tuple[bool, str]:
    if state.entry_price <= 0:
        return False, ""
    pnl_pct = (close / state.entry_price) - 1.0
    if pnl_pct <= catastrophic_stop_pct:
        return True, "catastrophic"
    if pnl_pct <= hard_stop_pct:
        return True, "hard_stop"
    atr = max(state.entry_atr, state.entry_price * 0.01)
    stop = state.entry_price - sl_atr_mult * atr
    if close <= stop:
        return True, "atr_stop"
    tp = state.entry_price + tp_atr_mult * atr
    if close >= tp:
        return True, "take_profit"
    state.highest_close = max(state.highest_close, close)
    if pnl_pct >= activate_trail_pct:
        trail = state.highest_close - chandelier_atr_mult * atr
        if close <= trail:
            return True, "chandelier"
    held = (now - state.entry_time).total_seconds() / 3600.0
    if held >= time_stop_hours and pnl_pct < 0.02:
        return True, "time_stop"
    return False, ""
