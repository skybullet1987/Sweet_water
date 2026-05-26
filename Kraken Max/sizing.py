from __future__ import annotations

from collections import deque

from config import CONFIG, KrakenMaxConfig

BPS = 10_000.0


class AggressiveSizer:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.outcomes: deque[float] = deque(maxlen=40)

    def record_trade(self, pnl_fraction: float) -> None:
        self.outcomes.append(float(pnl_fraction))

    def _kelly(self) -> float:
        if len(self.outcomes) < 8:
            return 0.18
        wins = [x for x in self.outcomes if x > 0]
        losses = [-x for x in self.outcomes if x < 0]
        if not losses:
            return min(float(self.config.kelly_cap), 0.25)
        p = len(wins) / len(self.outcomes)
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = sum(losses) / len(losses)
        b = avg_win / max(avg_loss, 1e-9)
        raw = p - (1.0 - p) / max(b, 1e-9)
        return min(float(self.config.kelly_cap), max(0.05, raw))

    def weight_for_score(self, score: float, rv_annual: float, rank_pct: float) -> float:
        if score < float(self.config.entry_score_threshold):
            return 0.0
        vol = max(float(rv_annual), 1e-6)
        vol_w = min(0.55, float(self.config.target_annual_vol) / vol)
        conviction = min(1.0, (score - self.config.entry_score_threshold) / 0.8)
        rank_boost = 0.5 + 0.5 * max(0.0, rank_pct - 0.5)
        raw = vol_w * self._kelly() * conviction * rank_boost
        return min(float(self.config.max_position_pct), max(0.0, raw))

    def passes_cost_gate(self, score: float, notional: float, algo=None) -> bool:
        if score <= 0 or notional <= 0:
            return False
        if algo is not None and bool(self.config.use_calibrated_costs):
            from cost_model import CalibratedCostModel

            return CalibratedCostModel(self.config).passes_edge_gate(score, notional, algo)
        fee = notional * float(self.config.expected_round_trip_fees)
        spread = notional * (float(self.config.assumed_spread_bps) / BPS)
        slip = notional * (float(self.config.assumed_slippage_bps) / BPS)
        cost_pct = (fee + spread + slip) / notional
        edge = score * float(self.config.edge_scale)
        return edge > cost_pct * float(self.config.edge_cost_multiplier)


def can_afford(algo, qty: float, price: float) -> tuple[bool, float, float]:
    if qty <= 0:
        return True, 0.0, free_cash_usd(algo)
    safety = float(CONFIG.cash_safety_factor)
    required = float(qty) * max(float(price), 0.0)
    available = free_cash_usd(algo) * safety
    return required <= available + 1e-9, required, available


def free_cash_usd(algo) -> float:
    try:
        cash = float(algo.Portfolio.CashBook["USD"].Amount)
    except Exception:
        cash = float(getattr(algo.Portfolio, "Cash", 0.0) or 0.0)
    reserved = 0.0
    try:
        orders = algo.Transactions.GetOpenOrders()
    except Exception:
        orders = []
    for order in orders:
        qty = float(getattr(order, "Quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        px = float(getattr(order, "LimitPrice", 0.0) or 0.0) or float(getattr(order, "Price", 0.0) or 0.0)
        reserved += qty * max(px, 0.0)
    return max(0.0, cash - reserved)


def round_qty(qty: float, min_qty: float) -> float:
    if qty <= 0:
        return 0.0
    if min_qty <= 0:
        return qty
    steps = int(qty / min_qty)
    return steps * min_qty
