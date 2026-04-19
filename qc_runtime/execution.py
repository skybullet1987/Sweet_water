from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG, StrategyConfig


class KrakenFeeModel:
    MAKER_FEE = 0.0025
    TAKER_FEE = 0.0040

    @classmethod
    def estimate_round_trip_cost(cls, maker_ratio: float = 0.7) -> float:
        maker_ratio = max(0.0, min(1.0, maker_ratio))
        one_way = maker_ratio * cls.MAKER_FEE + (1.0 - maker_ratio) * cls.TAKER_FEE
        return 2.0 * one_way


class RealisticSlippage:
    @staticmethod
    def estimate_slippage_bps(spread_bps: float, participation: float) -> float:
        participation = max(0.0, min(1.0, participation))
        return max(1.0, 0.5 * spread_bps + 25.0 * participation)


@dataclass
class _BarrierState:
    entry_price: float
    atr: float
    entry_bar: int
    side: int


class Executor:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.open_positions: dict[str, _BarrierState] = {}
        self.pending_limit_orders: dict[str, int] = {}
        self.cancel_count = 0
        self.escalation_count = 0
        self.fee_model = KrakenFeeModel()
        self.slippage_model = RealisticSlippage()

    def place_entry(self, symbol: str, target_weight: float, score: float) -> dict[str, float | str]:
        spread_bps = 10.0
        slip_bps = self.slippage_model.estimate_slippage_bps(spread_bps=spread_bps, participation=min(abs(target_weight), 1.0))
        cost = self.fee_model.estimate_round_trip_cost() + (2 * slip_bps / 10_000.0)
        order = {
            "symbol": symbol,
            "target_weight": float(target_weight),
            "score": float(score),
            "estimated_cost": float(cost),
            "type": "limit",
        }
        self.pending_limit_orders[symbol] = 0
        return order

    def register_fill(self, symbol: str, price: float, atr: float, side: int, bar_index: int) -> None:
        self.open_positions[symbol] = _BarrierState(entry_price=price, atr=max(atr, 1e-9), entry_bar=bar_index, side=side)
        self.pending_limit_orders.pop(symbol, None)

    def escalate_stale_orders(self, stale_after_bars: int = 3) -> list[str]:
        escalated: list[str] = []
        for symbol in list(self.pending_limit_orders.keys()):
            self.pending_limit_orders[symbol] += 1
            if self.pending_limit_orders[symbol] >= stale_after_bars:
                self.escalation_count += 1
                escalated.append(symbol)
                self.pending_limit_orders.pop(symbol, None)
        return escalated

    def manage_exits(self, open_positions: dict[str, dict[str, float]], bar_index: int | None = None) -> list[tuple[str, str]]:
        exits: list[tuple[str, str]] = []
        for symbol, snap in open_positions.items():
            state = self.open_positions.get(symbol)
            if state is None:
                continue
            tp = state.entry_price + state.side * self.config.tp_atr_mult * state.atr
            sl = state.entry_price - state.side * self.config.stop_atr_mult * state.atr
            high = float(snap.get("high", snap.get("close", state.entry_price)))
            low = float(snap.get("low", snap.get("close", state.entry_price)))
            close = float(snap.get("close", state.entry_price))
            if state.side > 0:
                if high >= tp:
                    exits.append((symbol, "take_profit"))
                elif low <= sl:
                    exits.append((symbol, "stop_loss"))
            else:
                if low <= tp:
                    exits.append((symbol, "take_profit"))
                elif high >= sl:
                    exits.append((symbol, "stop_loss"))
            if bar_index is not None and (bar_index - state.entry_bar) >= self.config.time_stop_bars:
                exits.append((symbol, "time_stop"))
            if exits and exits[-1][0] == symbol:
                self.open_positions.pop(symbol, None)
        return exits
