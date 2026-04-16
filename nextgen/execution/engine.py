from __future__ import annotations

from dataclasses import dataclass

from nextgen.core.types import ExecutionRequest, ExecutionResult, PortfolioTarget


@dataclass(frozen=True)
class ExecutionConfig:
    default_urgency: float = 0.5
    max_slippage_bps: float = 15.0


class ExecutionPlanner:
    """Execution request/result scaffold decoupled from alpha/risk logic."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def to_request(self, target: PortfolioTarget) -> ExecutionRequest:
        return ExecutionRequest(
            symbol=target.symbol,
            target_weight=target.target_weight,
            urgency=self.config.default_urgency,
            max_slippage_bps=self.config.max_slippage_bps,
        )

    def stub_fill(self, request: ExecutionRequest, success: bool = True) -> ExecutionResult:
        return ExecutionResult(
            symbol=request.symbol,
            requested_weight=request.target_weight,
            filled_weight=request.target_weight if success else 0.0,
            avg_price=None,
            success=success,
            reason="" if success else "not_executed",
        )
