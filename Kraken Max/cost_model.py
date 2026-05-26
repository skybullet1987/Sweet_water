from __future__ import annotations

from dataclasses import dataclass

from config import CONFIG, KrakenMaxConfig

BPS = 10_000.0


@dataclass(frozen=True)
class CostBreakdown:
    fee_pct: float
    spread_pct: float
    slippage_pct: float

    @property
    def total_pct(self) -> float:
        return self.fee_pct + self.spread_pct + self.slippage_pct


class CalibratedCostModel:
    """v7: round-trip cost from FillTracker live stats with config fallbacks."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def from_fill_tracker(self, fill_tracker) -> CostBreakdown:
        fee = float(self.config.expected_round_trip_fees)
        spread = float(self.config.assumed_spread_bps) / BPS
        slip = float(self.config.assumed_slippage_bps) / BPS
        if fill_tracker is None:
            return CostBreakdown(fee_pct=fee, spread_pct=spread, slippage_pct=slip)
        stats = fill_tracker.stats
        n = int(stats.limits_filled) + int(stats.market_filled)
        if n >= int(self.config.cost_calibration_min_fills):
            slip = max(slip, float(stats.avg_slippage_bps) / BPS)
            fill_rate = float(stats.fill_rate)
            if fill_rate < 0.7:
                spread *= 1.0 + (0.7 - fill_rate)
        return CostBreakdown(fee_pct=fee, spread_pct=spread, slippage_pct=slip)

    def round_trip_pct(self, algo) -> float:
        ft = getattr(algo, "fill_tracker", None)
        return self.from_fill_tracker(ft).total_pct

    def passes_edge_gate(self, score: float, notional: float, algo) -> bool:
        if score <= 0 or notional <= 0:
            return False
        cost_pct = self.round_trip_pct(algo)
        edge = score * float(self.config.edge_scale)
        return edge > cost_pct * float(self.config.edge_cost_multiplier)
