from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from nextgen.core.models import PortfolioTarget, RiskDecision


@dataclass(frozen=True)
class RiskConfig:
    target_portfolio_volatility: float = 0.25
    max_position_weight: float = 0.20
    max_position_risk: float = 0.05
    max_gross_exposure: float = 1.0
    max_net_exposure: float = 0.5
    drawdown_throttle_level: float = 0.10
    kill_switch_drawdown: float = 0.20


@dataclass(frozen=True)
class PositionState:
    weight: float
    annualized_volatility: float


@dataclass(frozen=True)
class PortfolioState:
    estimated_portfolio_volatility: float
    current_drawdown: float
    gross_exposure: float
    net_exposure: float
    positions: Mapping[str, PositionState] = field(default_factory=dict)


class UnifiedRiskEngine:
    """Single risk decision authority for approval and position sizing."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def evaluate_target(self, target: PortfolioTarget, portfolio: PortfolioState) -> RiskDecision:
        reasons: list[str] = []

        if portfolio.current_drawdown >= self.config.kill_switch_drawdown:
            return RiskDecision(False, 0.0, ("kill_switch_drawdown",))

        adjusted = target.target_weight

        if portfolio.estimated_portfolio_volatility > self.config.target_portfolio_volatility > 0:
            vol_scale = self.config.target_portfolio_volatility / portfolio.estimated_portfolio_volatility
            adjusted *= vol_scale
            reasons.append("target_volatility_scaling")

        if portfolio.current_drawdown >= self.config.drawdown_throttle_level:
            span = max(1e-9, self.config.kill_switch_drawdown - self.config.drawdown_throttle_level)
            throttle = max(0.0, 1.0 - ((portfolio.current_drawdown - self.config.drawdown_throttle_level) / span))
            adjusted *= throttle
            reasons.append("drawdown_throttle")

        current_position = portfolio.positions.get(target.symbol)
        position_vol = current_position.annualized_volatility if current_position else 0.0
        if position_vol > 0 and abs(adjusted) * position_vol > self.config.max_position_risk:
            adjusted = (self.config.max_position_risk / position_vol) * (1 if adjusted >= 0 else -1)
            reasons.append("position_risk_cap")

        if abs(adjusted) > self.config.max_position_weight:
            adjusted = self.config.max_position_weight * (1 if adjusted >= 0 else -1)
            reasons.append("position_weight_cap")

        projected_gross = portfolio.gross_exposure - abs(current_position.weight) if current_position else portfolio.gross_exposure
        projected_gross += abs(adjusted)
        if projected_gross > self.config.max_gross_exposure:
            return RiskDecision(False, 0.0, tuple(reasons + ["gross_exposure_cap"]))

        projected_net = portfolio.net_exposure - (current_position.weight if current_position else 0.0) + adjusted
        if abs(projected_net) > self.config.max_net_exposure:
            return RiskDecision(False, 0.0, tuple(reasons + ["net_exposure_cap"]))

        approved = abs(adjusted) > 0
        return RiskDecision(approved, adjusted if approved else 0.0, tuple(reasons))
