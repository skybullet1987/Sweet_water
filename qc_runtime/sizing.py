from __future__ import annotations

from collections import deque
import math

from config import CONFIG, StrategyConfig

BPS_TO_DECIMAL = 10_000.0


class Sizer:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.trade_outcomes: deque[float] = deque(maxlen=60)

    def record_trade(self, pnl_fraction: float) -> None:
        self.trade_outcomes.append(float(pnl_fraction))

    def update_returns(self, _ret: float) -> None:
        return

    def _kelly_estimate(self) -> float:
        if len(self.trade_outcomes) < 20:
            return 0.05
        wins = [x for x in self.trade_outcomes if x > 0]
        losses = [-x for x in self.trade_outcomes if x < 0]
        p = len(wins) / max(len(self.trade_outcomes), 1)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss <= 1e-12:
            return min(float(self.config.kelly_cap), 0.10)
        b = avg_win / avg_loss
        if b <= 1e-12:
            return 0.0
        return min(float(self.config.kelly_cap), max(0.0, p - (1.0 - p) / b))

    def _vol_weight(self, realized_vol_annual: float) -> float:
        vol = max(float(realized_vol_annual), 1e-9)
        raw = self.config.target_annual_vol / vol
        return max(0.02, min(0.30, raw))

    def size_for_trade(self, symbol: str, score: float, current_portfolio_state: dict[str, float]) -> float:
        _ = symbol
        if float(score) <= 0:
            return 0.0
        realized_vol = float(current_portfolio_state.get("realized_vol_annual", self.config.target_annual_vol))
        atr = max(float(current_portfolio_state.get("atr", 1.0)), 1e-9)
        atrs = current_portfolio_state.get("open_position_atrs", [atr]) or [atr]
        erc = (1.0 / atr) / max(sum(1.0 / max(float(v), 1e-9) for v in atrs), 1e-9)
        kelly_mult = min(self.config.kelly_cap, self._kelly_estimate())
        return max(0.0, self._vol_weight(realized_vol) * kelly_mult * erc)

    def passes_cost_gate(self, symbol: str, score: float, notional: float, fee_model, is_limit: bool = True) -> bool:
        _ = symbol
        s = abs(float(score or 0.0))
        n = abs(float(notional or 0.0))
        if s <= 0 or n <= 0:
            return False
        fee_cost = n * float(self.config.expected_round_trip_fees)
        if fee_model is not None and hasattr(fee_model, "estimate_round_trip_cost"):
            try:
                fee_cost = float(fee_model.estimate_round_trip_cost(symbol, n, is_limit=is_limit))
            except Exception:
                fee_cost = n * float(self.config.expected_round_trip_fees)
        spread_cost = n * (float(getattr(self.config, "assumed_spread_bps", 12.0)) / BPS_TO_DECIMAL)
        slippage_cost = n * (float(getattr(self.config, "assumed_slippage_bps", 8.0)) / BPS_TO_DECIMAL)
        total_cost_pct = (fee_cost + spread_cost + slippage_cost) / max(n, 1e-9)
        expected_edge = s * float(getattr(self.config, "edge_scale", 0.02))
        return expected_edge > total_cost_pct * float(getattr(self.config, "edge_cost_multiplier", 2.5))


def _order_qty(order) -> float:
    qty = float(getattr(order, "Quantity", 0.0) or 0.0)
    if qty != 0.0:
        return qty
    abs_qty = float(getattr(order, "AbsoluteQuantity", 0.0) or 0.0)
    direction = str(getattr(order, "Direction", "") or "").lower()
    return abs_qty if abs_qty > 0 and "buy" in direction else 0.0


def reserved_cash_usd(algo) -> float:
    try:
        orders = algo.Transactions.GetOpenOrders()
    except Exception:
        orders = []
    reserved = 0.0
    for order in orders:
        qty = _order_qty(order)
        if qty <= 0:
            continue
        px = float(getattr(order, "LimitPrice", 0.0) or 0.0) or float(getattr(order, "Price", 0.0) or 0.0)
        reserved += qty * max(px, 0.0)
    return max(0.0, reserved)


def free_cash_usd(algo) -> float:
    try:
        cash = float(algo.Portfolio.CashBook["USD"].Amount)
    except Exception:
        cash = float(getattr(algo.Portfolio, "Cash", 0.0) or 0.0)
    return max(0.0, cash - reserved_cash_usd(algo))


def can_afford(algo, symbol, qty: float, price: float, cash_safety_factor: float | None = None) -> tuple[bool, float, float]:
    _ = symbol
    if qty <= 0:
        return True, 0.0, free_cash_usd(algo)
    safety = float(cash_safety_factor if cash_safety_factor is not None else getattr(getattr(algo, "config", CONFIG), "cash_safety_factor", 0.97))
    safety = min(max(safety, 0.1), 1.0)
    required = float(qty) * max(float(price or 0.0), 0.0)
    available = free_cash_usd(algo) * safety
    return bool(price and math.isfinite(price) and required <= available + 1e-9), required, available


def is_price_stale(algo, symbol, price: float, max_age_minutes: int | None = None) -> bool:
    if not math.isfinite(float(price)) or float(price) <= 0:
        return True
    max_age = int(max_age_minutes if max_age_minutes is not None else getattr(getattr(algo, "config", CONFIG), "stale_price_minutes", 90) or 90)
    sec = getattr(getattr(algo, "Securities", None), "get", lambda *_args, **_kwargs: None)(symbol)
    if sec is None:
        return False
    now = getattr(algo, "Time", None)
    if now is None:
        return False
    last = None
    try:
        data = sec.GetLastData()
        last = getattr(data, "EndTime", None) or getattr(data, "Time", None)
    except Exception:
        last = getattr(sec, "LocalTime", None)
    if last is None:
        return False
    try:
        return (now - last).total_seconds() / 60.0 > max(1, max_age)
    except Exception:
        return False


def _rebalance_fail_state(algo):
    if not hasattr(algo, "_rebalance_fail_streak"):
        algo._rebalance_fail_streak = {}
    if not hasattr(algo, "_rebalance_blocked_date"):
        algo._rebalance_blocked_date = {}
    return algo._rebalance_fail_streak, algo._rebalance_blocked_date


def rebalance_symbol_blocked(algo, symbol) -> bool:
    _, blocked = _rebalance_fail_state(algo)
    return blocked.get(symbol) == getattr(getattr(algo, "Time", None), "date", lambda: None)()


def mark_rebalance_failure(algo, symbol, reason: str = "") -> None:
    _ = reason
    streak, blocked = _rebalance_fail_state(algo)
    cap = int(getattr(getattr(algo, "config", CONFIG), "rebalance_invalid_retry_cap", 3) or 3)
    streak[symbol] = int(streak.get(symbol, 0)) + 1
    if streak[symbol] >= max(1, cap):
        blocked[symbol] = getattr(getattr(algo, "Time", None), "date", lambda: None)()


def clear_rebalance_failure(algo, symbol) -> None:
    streak, blocked = _rebalance_fail_state(algo)
    streak.pop(symbol, None)
    blocked.pop(symbol, None)


class Executor:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.entry_prices: dict[str, float] = {}
        self.entry_atrs: dict[str, float] = {}
        self.highest_close: dict[str, float] = {}

    def place_entry(self, symbol: str, target_weight: float, score: float) -> dict[str, float | str]:
        from execution import KrakenTieredFeeModel

        return {
            "symbol": symbol,
            "target_weight": float(target_weight),
            "score": float(score),
            "estimated_cost": float(KrakenTieredFeeModel().estimate_round_trip_cost(symbol, 100.0, is_limit=True) / 100.0),
            "type": "limit",
        }

    def register_fill(self, symbol: str, price: float, atr: float, side: int, bar_index: int) -> None:
        _ = side, bar_index
        self.entry_prices[symbol] = float(price)
        self.entry_atrs[symbol] = float(atr)
        self.highest_close[symbol] = float(price)

    def manage_exits(self, open_positions: dict[str, dict[str, float]], bar_index: int | None = None) -> list[tuple[str, str]]:
        _ = bar_index
        out = []
        for symbol, snap in open_positions.items():
            entry = self.entry_prices.get(symbol)
            atr = self.entry_atrs.get(symbol, 0.0)
            if entry is None or atr <= 0:
                continue
            high = float(snap.get("high", entry))
            low = float(snap.get("low", entry))
            close = float(snap.get("close", entry))
            self.highest_close[symbol] = max(self.highest_close.get(symbol, entry), close)
            tp = entry + self.config.tp_atr_mult * atr
            sl = entry - self.config.sl_atr_mult * atr
            chand = self.highest_close[symbol] - self.config.chandelier_atr_mult * atr
            if high >= tp:
                out.append((symbol, "TP"))
            elif low <= max(sl, chand):
                out.append((symbol, "Chandelier" if chand >= sl else "SL"))
        return out


__all__ = [
    "Sizer",
    "Executor",
    "reserved_cash_usd",
    "free_cash_usd",
    "can_afford",
    "is_price_stale",
    "rebalance_symbol_blocked",
    "mark_rebalance_failure",
    "clear_rebalance_failure",
]
