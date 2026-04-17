from __future__ import annotations

from typing import Protocol, Sequence

from .models import (
    Bar,
    FeatureOutput,
    PortfolioTarget,
    RegimeState,
    RiskDecision,
    SignalOutput,
)


class FeatureEngine(Protocol):
    def update(self, bar: Bar) -> FeatureOutput:
        ...


class RegimeEngine(Protocol):
    def update(self, features: Sequence[FeatureOutput]) -> RegimeState:
        ...


class SignalSleeve(Protocol):
    def generate(self, feature: FeatureOutput, regime: RegimeState) -> SignalOutput:
        ...


class PortfolioAllocator(Protocol):
    def allocate(self, signals: Sequence[SignalOutput], regime: RegimeState) -> Sequence[PortfolioTarget]:
        ...


class RiskEngine(Protocol):
    def evaluate(self, target: PortfolioTarget) -> RiskDecision:
        ...


class AccountingEngine(Protocol):
    def ingest_fill(self, symbol: str, quantity: float, price: float, fee: float = 0.0, funding: float = 0.0) -> None:
        ...
