from __future__ import annotations

import math
from datetime import datetime, timezone

try:  # pragma: no cover
    from AlgorithmImports import OrderDirection, OrderType
    from QuantConnect.Orders.Fees import FeeModel, OrderFee  # type: ignore
    from QuantConnect.Securities import CashAmount  # type: ignore
except Exception:  # pragma: no cover
    class OrderDirection:
        Buy = 1
        Sell = -1

    class OrderType:
        Limit = 1

    class FeeModel:
        pass

    class CashAmount:
        def __init__(self, amount, currency):
            self.Amount = amount
            self.Currency = currency

    class OrderFee:
        def __init__(self, value):
            self.Value = value

from config import CONFIG, StrategyConfig


def get_effective_round_trip_fee(algo) -> float:
    return max(0.0, min(float(getattr(algo, "expected_round_trip_fees", CONFIG.expected_round_trip_fees)), 0.05))


def get_min_quantity(algo, symbol) -> float:
    ticker = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    try:
        sec = algo.Securities[symbol]
        min_size = float(getattr(sec.SymbolProperties, "MinimumOrderSize", 0.0) or 0.0)
        if min_size > 0:
            return min_size
    except Exception:
        pass
    cfg = getattr(algo, "config", CONFIG)
    return float(getattr(cfg, "min_qty_fallback", {}).get(ticker, 0.0001))


def estimate_min_qty(algo, symbol) -> float:
    price = float(getattr(algo.Securities.get(symbol), "Price", 0.0) or 0.0)
    if price <= 0:
        return 1.0
    return max(get_min_quantity(algo, symbol), 1.0 / price)


def get_min_notional_usd(algo, symbol) -> float:
    ticker = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    cfg = getattr(algo, "config", CONFIG)
    floor = float(getattr(cfg, "min_notional_fallback", {}).get(ticker, getattr(algo, "min_notional", 1.0)))
    try:
        price = float(algo.Securities[symbol].Price)
    except Exception:
        price = 0.0
    implied = price * get_min_quantity(algo, symbol) if price > 0 else 0.0
    return max(floor, implied, float(getattr(algo, "min_notional", 1.0)))


def round_quantity(algo, symbol, quantity: float) -> float:
    try:
        lot = float(getattr(algo.Securities[symbol].SymbolProperties, "LotSize", 0.0) or 0.0)
    except Exception:
        lot = 0.0
    if lot <= 0:
        return float(quantity)
    sign = 1.0 if quantity >= 0 else -1.0
    return sign * math.floor(abs(quantity) / lot) * lot


def is_invested_not_dust(algo, symbol) -> bool:
    if symbol not in algo.Portfolio:
        return False
    holding = algo.Portfolio[symbol]
    qty = float(getattr(holding, "Quantity", 0.0) or 0.0)
    if qty <= 0:
        return False
    price = float(getattr(algo.Securities.get(symbol), "Price", getattr(holding, "Price", 0.0)) or 0.0)
    return qty >= get_min_quantity(algo, symbol) * 0.5 or qty * price >= get_min_notional_usd(algo, symbol) * 0.5


def get_actual_position_count(algo) -> int:
    return sum(1 for item in algo.Portfolio for s in [getattr(item, "Key", item)] if is_invested_not_dust(algo, s))


def debug_limited(algo, msg: str) -> None:
    budget = int(getattr(algo, "log_budget", 0) or 0)
    if budget > 0:
        algo.Debug(msg)
        algo.log_budget = budget - 1


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=30, tag="Entry", force_market=False, signal_score=None):
    _ = timeout_seconds, signal_score
    if force_market:
        return algo.MarketOrder(symbol, quantity, tag=tag)
    sec = algo.Securities[symbol]
    price = float(getattr(sec, "Price", 0.0) or 0.0)
    if price <= 0:
        return None
    bid = float(getattr(sec, "BidPrice", 0.0) or 0.0)
    ask = float(getattr(sec, "AskPrice", 0.0) or 0.0)
    if bid > 0 and ask > 0:
        limit_price = bid if quantity > 0 else ask
    else:
        limit_price = price
    ticket = algo.LimitOrder(symbol, quantity, limit_price, tag=tag)
    algo._submitted_orders[symbol] = {
        "order_id": ticket.OrderId,
        "time": getattr(algo, "Time", datetime.now(timezone.utc)),
        "quantity": float(quantity),
        "intent": "entry" if quantity > 0 else "exit",
    }
    debug_limited(algo, f"ORD key=maker symbol={symbol.Value} qty={quantity:.8f}")
    return ticket


def place_entry(algo, symbol, quantity, tag="Entry", force_market=False, signal_score=None):
    if getattr(algo, "long_only", True) and float(quantity) <= 0:
        return None
    if is_invested_not_dust(algo, symbol):
        return None
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return None
    quantity = round_quantity(algo, symbol, float(quantity))
    if quantity <= 0:
        return None
    sec = algo.Securities.get(symbol)
    price = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
    if price <= 0:
        return None
    min_notional = get_min_notional_usd(algo, symbol)
    if quantity * price < min_notional:
        quantity = round_quantity(algo, symbol, max(quantity, estimate_min_qty(algo, symbol)))
    if quantity * price < min_notional:
        return None
    return place_limit_or_market(algo, symbol, quantity, tag=tag, force_market=force_market, signal_score=signal_score)


def cleanup_position(algo, symbol, record_pnl=False, exit_price=None):
    entry_price = algo.entry_prices.get(symbol)
    if record_pnl and entry_price and exit_price and entry_price > 0 and exit_price > 0:
        pnl = (exit_price - entry_price) / entry_price - get_effective_round_trip_fee(algo)
        algo.pnl_by_tag.setdefault(getattr(algo, "_last_exit_tag", "Unknown"), []).append(pnl)
        algo.pnl_by_regime.setdefault(getattr(algo, "market_regime", "unknown"), []).append(pnl)
    algo.entry_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    algo.highest_close.pop(symbol, None)
    algo.entry_atrs.pop(symbol, None)


def smart_liquidate(algo, symbol, tag="Liquidate"):
    if symbol not in algo.Portfolio:
        return False
    qty = float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
    if qty <= 0:
        return False
    qty = min(qty, float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0))
    qty = round_quantity(algo, symbol, qty)
    if qty <= 0:
        return False
    ticket = place_limit_or_market(algo, symbol, -qty, tag=tag)
    return ticket is not None


def manage_open_positions(algo, data=None):
    exits = []
    for symbol in list(getattr(algo, "entry_prices", {}).keys()):
        if not is_invested_not_dust(algo, symbol):
            cleanup_position(algo, symbol)
            continue
        sec = algo.Securities.get(symbol)
        if sec is None:
            continue
        bar = None
        if data is not None:
            bar = data.Bars.get(symbol) if hasattr(data, "Bars") else None
        close = float(getattr(bar, "Close", getattr(sec, "Price", 0.0)) or 0.0)
        high = float(getattr(bar, "High", close) or close)
        low = float(getattr(bar, "Low", close) or close)
        current_atr = float(getattr(algo.feature_engine, "current_features", lambda *_: {})(symbol.Value).get("atr", 0.0) or 0.0)

        entry_price = float(algo.entry_prices.get(symbol, close) or close)
        entry_atr = float(algo.entry_atrs.get(symbol, current_atr) or current_atr)
        if entry_atr <= 0:
            entry_atr = max(entry_price * 0.01, 1e-6)

        highest = float(algo.highest_close.get(symbol, entry_price) or entry_price)
        highest = max(highest, close)
        algo.highest_close[symbol] = highest

        bars_held = 0
        if symbol in algo.entry_times:
            bars_held = int((algo.Time - algo.entry_times[symbol]).total_seconds() / 3600)

        tp = entry_price + algo.config.tp_atr_mult * entry_atr
        hard_sl = entry_price - algo.config.sl_atr_mult * entry_atr
        trailing_armed = close > entry_price * (1.0 + algo.config.activate_trailing_above_pct)
        trailing_sl = highest - algo.config.chandelier_atr_mult * max(current_atr, 1e-9)
        effective_sl = max(hard_sl, trailing_sl) if trailing_armed else hard_sl

        tag = None
        exit_px = close
        if high >= tp:
            tag = "TP"
            exit_px = tp
        elif low <= effective_sl:
            tag = "Chandelier" if trailing_armed and trailing_sl >= hard_sl else "SL"
            exit_px = effective_sl
        elif bars_held >= algo.config.time_stop_bars:
            tag = "TimeStop"
            exit_px = close

        if tag:
            algo._last_exit_tag = tag
            if smart_liquidate(algo, symbol, tag=tag):
                cleanup_position(algo, symbol, record_pnl=True, exit_price=exit_px)
            else:
                cleanup_position(algo, symbol)
            exits.append((symbol, tag))
    return exits


def execute_regime_entries(algo, candidates, regime_tag="regime"):
    try:
        available = float(algo.Portfolio.CashBook["USD"].Amount)
    except Exception:
        available = float(getattr(algo.Portfolio, "Cash", 0.0) or 0.0)
    min_notional = float(getattr(algo, "min_notional", 1.0) or 1.0)
    hour = getattr(algo, "Time", None)
    if hour is not None:
        hour = hour.replace(minute=0, second=0, microsecond=0)

    if available < min_notional * 1.2 or get_actual_position_count(algo) >= int(algo.config.max_positions):
        if getattr(algo, "_last_cash_gate_log", None) != hour:
            debug_limited(algo, f"GATE key=cash available={available:.2f}")
            algo._last_cash_gate_log = hour
        return []

    placed = []
    for symbol, score, weight in candidates:
        if float(score) <= 0:
            continue
        if is_invested_not_dust(algo, symbol) or len(algo.Transactions.GetOpenOrders(symbol)) > 0:
            continue
        sec = algo.Securities.get(symbol)
        if sec is None:
            continue
        price = float(getattr(sec, "Price", 0.0) or 0.0)
        if price <= 0:
            continue

        equity = float(getattr(algo.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        floor = max(get_min_notional_usd(algo, symbol) * 1.5, algo.config.min_position_floor_usd)
        ceiling = min(equity * algo.config.max_position_pct, available * 0.95)
        notional = max(0.0, min(ceiling, equity * abs(float(weight))))
        if notional < floor:
            debug_limited(algo, f"GATE key=floor symbol={symbol.Value}")
            continue

        qty = round_quantity(algo, symbol, notional / max(price, 1e-9))
        if qty <= 0 or qty * price < floor:
            continue

        ticket = place_entry(algo, symbol, qty, tag=f"{regime_tag}:entry", signal_score=score)
        if ticket is not None:
            placed.append(ticket)
            available -= qty * price
            if available < min_notional:
                break
    return placed


def escalate_stale_orders(algo):
    stale_after = int(getattr(algo, "stale_order_bars", getattr(CONFIG, "stale_order_bars", 3)))
    escalated = []
    for symbol, info in list(getattr(algo, "_submitted_orders", {}).items()):
        age = int(info.get("age_bars", 0)) + 1
        info["age_bars"] = age
        algo._submitted_orders[symbol] = info
        if age < stale_after:
            continue
        oid = info.get("order_id")
        try:
            if oid is not None:
                algo.Transactions.CancelOrder(int(oid))
        except Exception:
            pass
        qty = float(info.get("quantity", 0.0) or 0.0)
        if qty != 0:
            place_limit_or_market(algo, symbol, qty, tag="[StaleEsc]", force_market=True)
        algo._submitted_orders.pop(symbol, None)
        escalated.append(symbol)
    return escalated


def get_hold_bucket(hold_hours):
    if hold_hours < 0.5:
        return "<30min"
    if hold_hours < 2:
        return "30min-2h"
    if hold_hours < 6:
        return "2h-6h"
    return "6h+"


def slip_log(algo, symbol, direction, fill_price):
    _ = algo, symbol, direction, fill_price
    return


class KrakenTieredFeeModel(FeeModel):
    LIMIT_TAKER_RATIO = 0.25
    FEE_TIERS = [
        (500_000, 0.0008, 0.0018), (250_000, 0.0010, 0.0020), (100_000, 0.0012, 0.0022),
        (50_000, 0.0014, 0.0024), (25_000, 0.0020, 0.0035), (10_000, 0.0022, 0.0038),
        (2_500, 0.0030, 0.0060), (0, 0.0040, 0.0080),
    ]

    def __init__(self, comparison_mode=False, fixed_maker_rate=None, fixed_taker_rate=None):
        self._comparison_mode = bool(comparison_mode)
        self._fixed_maker_rate = self.FEE_TIERS[-1][1] if fixed_maker_rate is None else float(fixed_maker_rate)
        self._fixed_taker_rate = self.FEE_TIERS[-1][2] if fixed_taker_rate is None else float(fixed_taker_rate)
        self._cumulative_volume = 0.0

    def _current_rates(self):
        if self._comparison_mode:
            return self._fixed_maker_rate, self._fixed_taker_rate
        maker, taker = self.FEE_TIERS[-1][1], self.FEE_TIERS[-1][2]
        for min_vol, m, t in self.FEE_TIERS:
            if self._cumulative_volume >= min_vol:
                maker, taker = m, t
                break
        return maker, taker

    def estimate_round_trip_cost(self, symbol, notional, is_limit=True):
        _ = symbol
        n = abs(float(notional or 0.0))
        if n <= 0:
            return 0.0
        maker, taker = self._current_rates()
        side_pct = ((1 - self.LIMIT_TAKER_RATIO) * maker + self.LIMIT_TAKER_RATIO * taker) if is_limit else taker
        return n * side_pct * 2.0

    def GetOrderFee(self, parameters):  # pragma: no cover
        order = parameters.Order
        price = parameters.Security.Price
        notional = order.AbsoluteQuantity * price
        if not self._comparison_mode:
            self._cumulative_volume += notional
        maker, taker = self._current_rates()
        pct = ((1 - self.LIMIT_TAKER_RATIO) * maker + self.LIMIT_TAKER_RATIO * taker) if order.Type == OrderType.Limit else taker
        return OrderFee(CashAmount(notional * pct, "USD"))


class RealisticCryptoSlippage:
    def __init__(self, algo=None, stress_mult=1.0):
        self.algo = algo
        self.base_slippage_pct = 0.0040 * max(0.1, float(stress_mult))
        self.volume_impact_factor = 0.25
        self.max_slippage_pct = 0.0500

    def _estimate_slippage_pct(self, price, notional, volume=0.0, bid=0.0, ask=0.0):
        if price <= 0:
            return 0.0
        slip = self.base_slippage_pct
        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            spread = (ask - bid) / max(mid, 1e-9)
            participation = abs(notional) / max(volume * price, 1e-9) if volume > 0 else 0.0
            slip += 0.5 * spread * max(0.25, participation)
        if volume > 0:
            participation = abs(notional) / max(volume * price, 1e-9)
            slip += self.volume_impact_factor * math.sqrt(max(participation, 0.0))
        return min(slip, self.max_slippage_pct)

    def estimate_slippage_bps(self, symbol, notional, price, volume=0.0, bid=0.0, ask=0.0):
        _ = symbol
        return self._estimate_slippage_pct(price, notional, volume=volume, bid=bid, ask=ask) * 10_000.0

    def GetSlippageApproximation(self, asset, order):  # pragma: no cover
        price = float(getattr(asset, "Price", 0.0) or 0.0)
        notional = abs(float(getattr(order, "Quantity", 0.0) or 0.0)) * price
        pct = self._estimate_slippage_pct(
            price=price,
            notional=notional,
            volume=float(getattr(asset, "Volume", 0.0) or 0.0),
            bid=float(getattr(asset, "BidPrice", 0.0) or 0.0),
            ask=float(getattr(asset, "AskPrice", 0.0) or 0.0),
        )
        return price * pct


class Executor:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.entry_prices: dict[str, float] = {}
        self.entry_atrs: dict[str, float] = {}
        self.highest_close: dict[str, float] = {}

    def place_entry(self, symbol: str, target_weight: float, score: float) -> dict[str, float | str]:
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
