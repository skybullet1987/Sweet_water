from __future__ import annotations

from collections import deque

from config import CONFIG, StrategyConfig


class Sizer:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.trade_outcomes: deque[float] = deque(maxlen=60)

    def record_trade(self, pnl_fraction: float) -> None:
        self.trade_outcomes.append(float(pnl_fraction))

    def update_returns(self, _ret: float) -> None:
        return

    def _kelly_estimate(self) -> float:
        if len(self.trade_outcomes) < 20:
            return 0.10
        wins = [x for x in self.trade_outcomes if x > 0]
        losses = [-x for x in self.trade_outcomes if x < 0]
        p = len(wins) / max(len(self.trade_outcomes), 1)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss <= 1e-12:
            return 0.10
        b = avg_win / avg_loss
        if b <= 1e-12:
            return 0.0
        return max(0.0, p - (1.0 - p) / b)

    def _vol_weight(self, realized_vol_annual: float) -> float:
        vol = max(float(realized_vol_annual), 1e-9)
        raw = self.config.target_annual_vol / vol
        return max(0.02, min(0.30, raw))

    def size_for_trade(self, symbol: str, score: float, current_portfolio_state: dict[str, float]) -> float:
        _ = symbol
        if float(score) <= 0:
            return 0.0
        realized_vol = float(current_portfolio_state.get("realized_vol_annual", self.config.target_annual_vol))
        atr = max(float(current_portfolio_state.get("atr", 1.0)), 1e-9)
        atrs = current_portfolio_state.get("open_position_atrs", [atr]) or [atr]
        erc = (1.0 / atr) / max(sum(1.0 / max(float(v), 1e-9) for v in atrs), 1e-9)
        kelly_mult = min(self.config.kelly_cap, self._kelly_estimate())
        return max(0.0, self._vol_weight(realized_vol) * kelly_mult * erc)

    def passes_cost_gate(self, symbol: str, score: float, notional: float, fee_model, is_limit: bool = True) -> bool:
        _ = symbol, notional, fee_model, is_limit
        s = abs(float(score))
        if s <= 0:
            return False
        # Require expected edge to exceed cost multiplier on round-trip fees:
        # |score|*N >= m*fee*N -> |score| >= m*fee. Rearranged below to stay monotonic
        # around the threshold used by entry filtering.
        threshold = self.config.score_threshold * (1.0 + self.config.cost_gate_multiplier * self.config.expected_round_trip_fees / s)
        return s >= threshold


__all__ = ["Sizer"]
