from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG, StrategyConfig


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    adjusted_target_weight: float
    reason: str


class RiskManager:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.peak_equity = 0.0
        self.correlation_proxy = 0.0

    def _circuit_breaker_active(self, equity: float) -> bool:
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity <= 0:
            return False
        drawdown = (self.peak_equity - equity) / self.peak_equity
        return drawdown >= self.config.drawdown_halt_pct

    def evaluate(self, target: dict[str, float]) -> RiskDecision:
        target_weight = float(target.get("target_weight", 0.0))
        equity = float(target.get("equity", 0.0))
        gross = float(target.get("gross_exposure", 0.0))
        net = float(target.get("net_exposure", 0.0))
        correlation = float(target.get("correlation", 0.0))

        if self._circuit_breaker_active(equity):
            return RiskDecision(False, 0.0, "circuit_breaker_halt")
        adjusted = max(-self.config.max_position_weight, min(self.config.max_position_weight, target_weight))
        if abs(adjusted) < abs(target_weight):
            reason = "position_limit"
        else:
            reason = "approved"
        if gross + abs(adjusted) > 1.5:
            return RiskDecision(False, 0.0, "gross_exposure_cap")
        if abs(net + adjusted) > 1.0:
            return RiskDecision(False, 0.0, "net_exposure_cap")
        if abs(correlation) > self.config.correlation_throttle:
            adjusted *= 0.5
            reason = "correlation_throttle"
        return RiskDecision(abs(adjusted) > 0, adjusted, reason)
