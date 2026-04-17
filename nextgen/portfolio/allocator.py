from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from nextgen.core.models import PortfolioTarget, RegimeState, SignalOutput


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class AllocationConfig:
    sleeve_weights: Mapping[str, float] = field(default_factory=dict)
    min_signal_confidence: float = 0.05
    max_position_weight: float = 0.25
    max_gross_target: float = 1.0


class SignalPortfolioAllocator:
    def __init__(self, config: AllocationConfig | None = None) -> None:
        self.config = config or AllocationConfig()

    def allocate(self, signals: Sequence[SignalOutput], regime: RegimeState) -> Sequence[PortfolioTarget]:
        by_symbol: dict[str, float] = {}

        for signal in signals:
            if signal.confidence < self.config.min_signal_confidence:
                continue
            sleeve_weight = self.config.sleeve_weights.get(signal.sleeve, 1.0)
            regime_weight = max(0.2, regime.liquidity_quality) * (1.0 - 0.5 * regime.volatility_stress)
            by_symbol[signal.symbol] = by_symbol.get(signal.symbol, 0.0) + signal.score * sleeve_weight * regime_weight

        if not by_symbol:
            return ()

        gross = sum(abs(v) for v in by_symbol.values())
        if gross <= 0:
            return ()

        scaling = min(1.0, self.config.max_gross_target / gross)
        targets = []
        for symbol, raw in sorted(by_symbol.items()):
            weight = _clamp(raw * scaling, -self.config.max_position_weight, self.config.max_position_weight)
            targets.append(PortfolioTarget(symbol=symbol, target_weight=weight, source_score=raw))

        return tuple(targets)
