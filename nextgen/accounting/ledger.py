from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from nextgen.core.types import FillRecord, PnLRecord


@dataclass(frozen=True)
class AccountingConfig:
    default_fee_rate: float = 0.001


class UnifiedAccountingLedger:
    """Centralized accounting path for fills, fees/funding and realized PnL."""

    def __init__(self, config: AccountingConfig | None = None) -> None:
        self.config = config or AccountingConfig()
        self.position_qty: dict[str, float] = {}
        self.position_cost: dict[str, float] = {}
        self.fills: list[FillRecord] = []
        self.pnl_history: list[PnLRecord] = []

    def ingest_fill(self, symbol: str, quantity: float, price: float, fee: float | None = None, funding: float = 0.0, timestamp: datetime | None = None) -> FillRecord:
        fee_value = float(abs(quantity) * price * self.config.default_fee_rate if fee is None else fee)
        record = FillRecord(
            symbol=symbol,
            timestamp=timestamp or datetime.now(UTC),
            quantity=quantity,
            price=price,
            fee=fee_value,
            funding=funding,
        )
        self.fills.append(record)

        prev_qty = self.position_qty.get(symbol, 0.0)
        prev_cost = self.position_cost.get(symbol, 0.0)
        new_qty = prev_qty + quantity
        new_cost = prev_cost + (quantity * price)
        self.position_qty[symbol] = new_qty
        self.position_cost[symbol] = new_cost
        return record

    def realized_pnl_on_close(self, symbol: str, close_quantity: float, close_price: float, fee: float = 0.0, funding: float = 0.0, timestamp: datetime | None = None) -> PnLRecord:
        current_qty = self.position_qty.get(symbol, 0.0)
        if current_qty == 0:
            pnl = -fee + funding
            record = PnLRecord(symbol=symbol, timestamp=timestamp or datetime.now(UTC), realized_pnl=pnl, unrealized_pnl=0.0, fees=fee, funding=funding)
            self.pnl_history.append(record)
            return record

        avg_entry = self.position_cost.get(symbol, 0.0) / current_qty if current_qty else 0.0
        realized = close_quantity * (close_price - avg_entry) - fee + funding

        self.position_qty[symbol] = current_qty - close_quantity
        self.position_cost[symbol] = self.position_cost.get(symbol, 0.0) - close_quantity * avg_entry

        record = PnLRecord(
            symbol=symbol,
            timestamp=timestamp or datetime.now(UTC),
            realized_pnl=realized,
            unrealized_pnl=0.0,
            fees=fee,
            funding=funding,
        )
        self.pnl_history.append(record)
        return record
