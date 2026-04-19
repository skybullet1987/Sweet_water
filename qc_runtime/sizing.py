from __future__ import annotations

from collections import deque

from config import CONFIG, StrategyConfig

BPS_TO_DECIMAL = 10_000.0


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
        _ = symbol
        s = abs(float(score or 0.0))
        n = abs(float(notional or 0.0))
        if s <= 0 or n <= 0:
            return False
        fee_cost = n * float(self.config.expected_round_trip_fees)
        if fee_model is not None and hasattr(fee_model, "estimate_round_trip_cost"):
            try:
                fee_cost = float(fee_model.estimate_round_trip_cost(symbol, n, is_limit=is_limit))
            except Exception:
                fee_cost = n * float(self.config.expected_round_trip_fees)
        spread_cost = n * (float(getattr(self.config, "assumed_spread_bps", 12.0)) / BPS_TO_DECIMAL)
        slippage_cost = n * (float(getattr(self.config, "assumed_slippage_bps", 8.0)) / BPS_TO_DECIMAL)
        total_cost_pct = (fee_cost + spread_cost + slippage_cost) / max(n, 1e-9)
        expected_edge = s * float(getattr(self.config, "edge_scale", 0.02))
        return expected_edge > total_cost_pct * float(getattr(self.config, "edge_cost_multiplier", 2.5))


__all__ = ["Sizer"]
