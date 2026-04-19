from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

from config import CONFIG, StrategyConfig


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    adjusted_target_weight: float
    reason: str


@dataclass(frozen=True)
class PortfolioTarget:
    symbol: str
    target_weight: float


@dataclass(frozen=True)
class UnifiedRiskDecision:
    approved: bool
    adjusted_target_weight: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnifiedRiskConfig:
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
    """Single risk authority, restored from nextgen/risk/engine.py semantics."""

    def __init__(self, config: UnifiedRiskConfig | None = None) -> None:
        self.config = config or UnifiedRiskConfig()

    def evaluate_target(self, target: PortfolioTarget, portfolio: PortfolioState) -> UnifiedRiskDecision:
        reasons: list[str] = []

        if portfolio.current_drawdown >= self.config.kill_switch_drawdown:
            return UnifiedRiskDecision(False, 0.0, ("kill_switch_drawdown",))

        adjusted = float(target.target_weight)

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
            return UnifiedRiskDecision(False, 0.0, tuple(reasons + ["gross_exposure_cap"]))

        projected_net = portfolio.net_exposure - (current_position.weight if current_position else 0.0) + adjusted
        if abs(projected_net) > self.config.max_net_exposure:
            return UnifiedRiskDecision(False, 0.0, tuple(reasons + ["net_exposure_cap"]))

        approved = abs(adjusted) > 0
        return UnifiedRiskDecision(approved, adjusted if approved else 0.0, tuple(reasons))


class DrawdownCircuitBreaker:
    """Stops entries when drawdown breaches threshold; resets only on recovery."""

    def __init__(self, max_drawdown_pct: float = -0.10, recovery_pct: float = 0.02):
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.recovery_pct = float(recovery_pct)
        self.peak_equity = 0.0
        self.breaker_triggered = False
        self.trigger_time = None
        self.trigger_equity = None
        self.equity_at_trigger = 0.0

    def update(self, algorithm) -> None:
        current_equity = float(getattr(algorithm.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.breaker_triggered:
            recovery_target = self.equity_at_trigger * (1 + self.recovery_pct)
            if current_equity >= recovery_target:
                try:
                    algorithm.Debug(f"BREAKER RESET: equity recovered to ${current_equity:.2f}")
                except Exception:
                    pass
                self.breaker_triggered = False
                self.trigger_time = None

        if self.peak_equity > 0:
            drawdown = (current_equity - self.peak_equity) / self.peak_equity
            if drawdown <= self.max_drawdown_pct and not self.breaker_triggered:
                self.breaker_triggered = True
                self.trigger_time = getattr(algorithm, "Time", None)
                self.trigger_equity = current_equity
                self.equity_at_trigger = current_equity
                try:
                    algorithm.Debug(
                        f"⚠️ CIRCUIT BREAKER: drawdown={drawdown:.2%}, "
                        f"equity=${current_equity:.2f}, peak=${self.peak_equity:.2f}. Pausing new entries."
                    )
                except Exception:
                    pass

    def is_triggered(self) -> bool:
        return bool(self.breaker_triggered)

    def get_status(self) -> str:
        if self.breaker_triggered:
            return f"BREAKER ACTIVE (triggered {self.trigger_time})"
        return "Normal"


class RollingMaxDrawdown:
    """Tracks rolling maximum drawdown for reporting/monitoring."""

    def __init__(self, lookback_bars: int = 252):
        self.lookback_bars = int(lookback_bars)
        self.equity_history: deque[float] = deque(maxlen=self.lookback_bars)

    def update(self, equity: float) -> None:
        self.equity_history.append(float(equity))

    def get_max_drawdown(self) -> float:
        if len(self.equity_history) < 2:
            return 0.0
        running_peak = self.equity_history[0]
        max_dd = 0.0
        for equity in self.equity_history:
            if equity > running_peak:
                running_peak = equity
            if running_peak > 0:
                dd = (equity - running_peak) / running_peak
                if dd < max_dd:
                    max_dd = dd
        return max_dd

    def get_max_drawdown_pct(self) -> str:
        return f"{self.get_max_drawdown():.2%}"


class RiskManager:
    """Compatibility facade for simplified callers, now backed by UnifiedRiskEngine."""

    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.peak_equity = 0.0
        self.correlation_proxy = 0.0
        self.unified = UnifiedRiskEngine(
            UnifiedRiskConfig(
                target_portfolio_volatility=max(0.05, config.target_annual_vol),
                max_position_weight=max(0.01, config.max_position_weight),
                max_position_risk=0.05,
                max_gross_exposure=1.5,
                max_net_exposure=1.0,
                drawdown_throttle_level=max(0.05, config.drawdown_halt_pct * 0.5),
                kill_switch_drawdown=max(0.10, config.drawdown_halt_pct),
            )
        )

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
        est_vol = float(target.get("estimated_portfolio_volatility", self.config.target_annual_vol))
        drawdown = float(target.get("current_drawdown", 0.0))

        if self._circuit_breaker_active(equity):
            return RiskDecision(False, 0.0, "circuit_breaker_halt")

        adjusted = max(-self.config.max_position_weight, min(self.config.max_position_weight, target_weight))
        reason = "position_limit" if abs(adjusted) < abs(target_weight) else "approved"

        if gross + abs(adjusted) > 1.5:
            return RiskDecision(False, 0.0, "gross_exposure_cap")
        if abs(net + adjusted) > 1.0:
            return RiskDecision(False, 0.0, "net_exposure_cap")
        if abs(correlation) > self.config.correlation_throttle:
            adjusted *= 0.5
            reason = "correlation_throttle"

        u = self.unified.evaluate_target(
            PortfolioTarget(symbol=str(target.get("symbol", "__any__")), target_weight=adjusted),
            PortfolioState(
                estimated_portfolio_volatility=max(est_vol, 1e-9),
                current_drawdown=max(drawdown, 0.0),
                gross_exposure=max(gross, 0.0),
                net_exposure=net,
                positions={},
            ),
        )
        if not u.approved:
            final_reason = u.reasons[-1] if u.reasons else reason
            return RiskDecision(False, 0.0, final_reason)
        final_reason = u.reasons[-1] if u.reasons else reason
        return RiskDecision(abs(u.adjusted_target_weight) > 0, u.adjusted_target_weight, final_reason)


__all__ = [
    "RiskDecision",
    "RiskManager",
    "UnifiedRiskEngine",
    "UnifiedRiskConfig",
    "PortfolioTarget",
    "PortfolioState",
    "PositionState",
    "UnifiedRiskDecision",
    "DrawdownCircuitBreaker",
    "RollingMaxDrawdown",
]
