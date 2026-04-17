from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from config.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig


@dataclass(frozen=True)
class TripleBarrier:
    side: Literal["long", "short"]
    entry_price: float
    take_profit: float
    stop_loss: float
    expiry_bar: int


def build_barrier(
    entry_price: float,
    atr: float,
    side: Literal["long", "short"],
    entry_bar: int,
    config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> TripleBarrier:
    if side == "long":
        take_profit = entry_price + config.tp_atr_mult * atr
        stop_loss = entry_price - config.stop_atr_mult * atr
    else:
        take_profit = entry_price - config.tp_atr_mult * atr
        stop_loss = entry_price + config.stop_atr_mult * atr
    return TripleBarrier(
        side=side,
        entry_price=entry_price,
        take_profit=take_profit,
        stop_loss=stop_loss,
        expiry_bar=entry_bar + config.time_stop_bars,
    )


def check_barrier_hit(barrier: TripleBarrier, bar_index: int, high: float, low: float, close: float) -> str | None:
    if barrier.side == "long":
        if high >= barrier.take_profit:
            return "take_profit"
        if low <= barrier.stop_loss:
            return "stop_loss"
    else:
        if low <= barrier.take_profit:
            return "take_profit"
        if high >= barrier.stop_loss:
            return "stop_loss"

    if bar_index >= barrier.expiry_bar:
        return "time_stop"
    return None
