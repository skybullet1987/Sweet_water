from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Optional


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FeatureOutput:
    symbol: str
    timestamp: datetime
    values: Mapping[str, float]


@dataclass(frozen=True)
class RegimeState:
    timestamp: datetime
    trend_confidence: float
    mean_reversion_confidence: float
    volatility_stress: float
    liquidity_quality: float
    breadth_strength: float
    active_regime: str
    regime_persistence_steps: int


@dataclass(frozen=True)
class SignalOutput:
    sleeve: str
    symbol: str
    timestamp: datetime
    score: float
    confidence: float
    metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioTarget:
    symbol: str
    target_weight: float
    source_score: float
    metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    adjusted_target_weight: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionRequest:
    symbol: str
    target_weight: float
    urgency: float
    max_slippage_bps: float


@dataclass(frozen=True)
class ExecutionResult:
    symbol: str
    requested_weight: float
    filled_weight: float
    avg_price: Optional[float]
    success: bool
    reason: str = ""


@dataclass(frozen=True)
class FillRecord:
    symbol: str
    timestamp: datetime
    quantity: float
    price: float
    fee: float
    funding: float = 0.0


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    opened_at: datetime
    closed_at: datetime
    quantity: float
    entry_price: float
    exit_price: float
    realized_pnl: float
    fees: float
    funding: float


@dataclass(frozen=True)
class PnLRecord:
    symbol: str
    timestamp: datetime
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    funding: float


@dataclass(frozen=True)
class AccountingRecord:
    fill: FillRecord
    trade: Optional[TradeRecord]
    pnl: Optional[PnLRecord]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    symbols: tuple[str, ...]
    target_volatility: float
    max_gross_exposure: float
    max_net_exposure: float
    max_position_weight: float
    drawdown_throttle: float
    kill_switch_drawdown: float
