from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

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

try:
    from config import CONFIG, StrategyConfig
except ModuleNotFoundError:  # pragma: no cover
    from .config import CONFIG, StrategyConfig  # type: ignore
from sizing import (
    Executor,
    can_afford,
    clear_rebalance_failure,
    free_cash_usd,
    is_price_stale,
    mark_rebalance_failure,
    rebalance_symbol_blocked,
)

POSITION_TOLERANCE = 1e-9
DEFAULT_LAZY_ATR_PCT = 0.05


@dataclass
class PositionState:
    entry_price: float
    highest_close: float
    entry_atr: float
    entry_time: datetime | None
    strategy_owner: str = "momentum"
    initial_risk_distance: float = 0.0
    partial_tp_done: bool = False
    tight_trail_armed: bool = False


def position_status(algo, symbol) -> Literal["flat", "long", "pending"]:
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return "pending"
    try:
        qty = float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
    except Exception:
        qty = 0.0
    return "long" if qty > 0 else "flat"


def _position_state(algo) -> dict:
    if not hasattr(algo, "position_state"):
        algo.position_state = {}
    return algo.position_state


def _state_get(algo, symbol):
    pstate = _position_state(algo)
    state = pstate.get(symbol)
    if state is None:
        key = getattr(symbol, "Value", None)
        if key is not None:
            state = pstate.get(key)
    return state


def _state_pop(algo, symbol):
    pstate = _position_state(algo)
    pstate.pop(symbol, None)
    key = getattr(symbol, "Value", None)
    if key is not None:
        pstate.pop(key, None)


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
    securities = getattr(algo, "Securities", {})
    sec = securities.get(symbol) if hasattr(securities, "get") else None
    price = float(getattr(sec, "Price", getattr(holding, "Price", 0.0)) or 0.0)
    return qty >= get_min_quantity(algo, symbol) * 0.5 or qty * price >= get_min_notional_usd(algo, symbol) * 0.5


def get_actual_position_count(algo) -> int:
    count = 0
    for item in getattr(algo, "Portfolio", []):
        if hasattr(item, "Key"):
            symbol = item.Key
            holding = getattr(item, "Value", None)
        else:
            symbol = item
            holding = None
        if holding is None or not hasattr(holding, "Quantity"):
            try:
                holding = algo.Portfolio[symbol]
            except Exception:
                holding = None
        try:
            qty = float(getattr(holding, "Quantity", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        if qty > 0:
            count += 1
    return count


def debug_limited(algo, msg: str) -> None:
    budget = int(getattr(algo, "log_budget", 0) or 0)
    if budget > 0:
        algo.Debug(msg)
        algo.log_budget = budget - 1


def _hour_bucket(algo):
    now = getattr(algo, "Time", None)
    if now is None:
        return None
    try:
        return now.replace(minute=0, second=0, microsecond=0)
    except Exception:
        return now


def _log_once_per_hour(algo, key: str, message: str) -> None:
    bucket = _hour_bucket(algo)
    if not hasattr(algo, "_hourly_log_keys"):
        algo._hourly_log_keys = {}
    stamp = algo._hourly_log_keys.get(key)
    if stamp != bucket:
        debug_limited(algo, message)
        algo._hourly_log_keys[key] = bucket


def _order_qty(order) -> float:
    qty = float(getattr(order, "Quantity", 0.0) or 0.0)
    if qty != 0.0:
        return qty
    abs_qty = float(getattr(order, "AbsoluteQuantity", 0.0) or 0.0)
    direction = getattr(order, "Direction", None)
    if abs_qty <= 0 or direction is None:
        return 0.0
    return abs_qty if direction == OrderDirection.Buy else -abs_qty


def reserved_qty(algo, symbol) -> float:
    reserved = 0.0
    try:
        open_orders = algo.Transactions.GetOpenOrders(symbol)
    except Exception:
        open_orders = []
    for order in open_orders:
        qty = _order_qty(order)
        if qty < 0:
            reserved += abs(qty)
    return max(0.0, reserved)


def _holding_qty(algo, symbol) -> float:
    try:
        return max(0.0, float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0))
    except Exception:
        return 0.0


def _free_qty(algo, symbol) -> float:
    return max(0.0, _holding_qty(algo, symbol) - reserved_qty(algo, symbol))


def has_open_exit_order(algo, symbol) -> bool:
    return reserved_qty(algo, symbol) > POSITION_TOLERANCE


def _order_cap_allows_submit(algo, *, is_entry: bool) -> bool:
    if not is_entry:
        return True
    limit = int(getattr(getattr(algo, "config", CONFIG), "max_orders_per_day", 0) or 0)
    if limit <= 0:
        return True
    now = getattr(algo, "Time", None)
    day_key = getattr(now, "date", lambda: None)()
    if getattr(algo, "_orders_today_date", None) != day_key:
        algo._orders_today_date = day_key
        algo._orders_today = 0
    used = int(getattr(algo, "_orders_today", 0) or 0)
    if used >= limit:
        _log_once_per_hour(algo, "order_cap", f"ORD key=cap_reached used={used} max={limit}")
        return False
    return True


def _safe_submit_order(algo, symbol, quantity: float, submit_fn):
    qty = float(quantity or 0.0)
    if qty == 0:
        return None
    current = 0.0
    try:
        current = float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
    except Exception:
        current = 0.0
    if qty < 0:
        post = current + qty
        if current <= 0 or post < -POSITION_TOLERANCE:
            sym = getattr(symbol, "Value", str(symbol))
            debug_limited(algo, f"ORD key=blocked_short symbol={sym} qty={qty:.8f} hold={current:.8f}")
            return None
    return submit_fn()


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=30, tag="Entry", force_market=False, signal_score=None):
    _ = timeout_seconds, signal_score
    quantity = float(quantity or 0.0)
    if quantity > 0:
        failed = getattr(algo, "_failed_escalations", {}) or {}
        cutoff = algo.Time - timedelta(hours=24)
        expired = [s for s, t in failed.items() if t is None or t < cutoff]
        for s in expired:
            failed.pop(s, None)
        last_fail = failed.get(symbol)
        if last_fail is not None:
            try:
                hours_since = (algo.Time - last_fail).total_seconds() / 3600.0
            except Exception:
                hours_since = 999.0
            cooldown_hours = float(getattr(algo.config, "failed_esc_cooldown_hours", 6.0) or 6.0)
            if hours_since < cooldown_hours:
                _log_once_per_hour(
                    algo,
                    f"esc_cooldown:{getattr(symbol, 'Value', symbol)}",
                    f"ORD key=esc_cooldown sym={getattr(symbol, 'Value', symbol)} hrs={hours_since:.1f}",
                )
                return None
    abandoned_dust = getattr(algo, "_abandoned_dust", None)
    if quantity < 0 and abandoned_dust is not None and symbol in abandoned_dust:
        return None
    if quantity < 0:
        hold = _holding_qty(algo, symbol)
        reserved = reserved_qty(algo, symbol)
        free_qty = max(0.0, hold - reserved)
        if free_qty <= POSITION_TOLERANCE:
            _log_once_per_hour(
                algo,
                f"exit_reserved:{getattr(symbol, 'Value', symbol)}",
                f"ORD key=exit_skipped_reserved sym={getattr(symbol, 'Value', symbol)} hold={hold:.8f} reserved={reserved:.8f}",
            )
            return None
        desired = abs(quantity)
        sized = round_quantity(algo, symbol, min(free_qty, desired))
        if sized <= POSITION_TOLERANCE:
            _log_once_per_hour(
                algo,
                f"exit_rounded:{getattr(symbol, 'Value', symbol)}",
                f"ORD key=exit_skipped_round sym={getattr(symbol, 'Value', symbol)} free={free_qty:.8f}",
            )
            return None
        if has_open_exit_order(algo, symbol):
            _log_once_per_hour(
                algo,
                f"exit_duplicate:{getattr(symbol, 'Value', symbol)}",
                f"ORD key=exit_skipped_duplicate sym={getattr(symbol, 'Value', symbol)}",
            )
            return None
        quantity = -sized
    if not _order_cap_allows_submit(algo, is_entry=(quantity > 0)):
        return None
    if quantity > 0 and rebalance_symbol_blocked(algo, symbol) and str(tag).startswith("Rebalance"):
        _log_once_per_hour(
            algo,
            f"rebalance_blocked:{getattr(symbol, 'Value', symbol)}",
            f"REBAL_BLOCK sym={getattr(symbol, 'Value', symbol)} reason=retry_cap",
        )
        return None
    if force_market:
        securities = getattr(algo, "Securities", {})
        sec = securities.get(symbol) if hasattr(securities, "get") else None
        est_px = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else None
        if est_px is None or est_px <= 0:
            algo.Debug(f"FORCE_MARKET_SKIP sym={getattr(symbol, 'Value', symbol)} reason=no_price tag={tag}")
            return None
        if is_price_stale(algo, symbol, est_px):
            algo.Debug(f"FORCE_MARKET_SKIP sym={getattr(symbol, 'Value', symbol)} reason=stale_price tag={tag}")
            return None
        ok, required, available = can_afford(algo, symbol, quantity, est_px)
        if quantity > 0 and not ok:
            algo.Debug(
                "FORCE_MARKET_SKIP "
                f"sym={getattr(symbol, 'Value', symbol)} reason=insufficient_bp req={required:.2f} avail={available:.2f}"
            )
            if str(tag).startswith("Rebalance") or str(tag).startswith("[StaleEsc]"):
                mark_rebalance_failure(algo, symbol, "insuff_funds")
            return None
        ticket = _safe_submit_order(algo, symbol, quantity, lambda: algo.MarketOrder(symbol, quantity, tag=tag))
        if ticket is not None and quantity > 0:
            algo._orders_today = int(getattr(algo, "_orders_today", 0) or 0) + 1
        return ticket
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
    if is_price_stale(algo, symbol, limit_price):
        _log_once_per_hour(
            algo,
            f"stale_px:{getattr(symbol, 'Value', symbol)}",
            (
                "STALE_PRICE "
                f"sym={getattr(symbol, 'Value', symbol)} px={limit_price} "
                f"tag={tag}"
            ),
        )
        return None
    ok, required, available = can_afford(algo, symbol, quantity, limit_price)
    if quantity > 0 and not ok:
        _log_once_per_hour(
            algo,
            f"insuff:{getattr(symbol, 'Value', symbol)}",
            (
                "INSUFF_FUNDS "
                f"sym={getattr(symbol, 'Value', symbol)} req={required:.4f} avail={available:.4f} "
                f"tag={tag}"
            ),
        )
        if str(tag).startswith("Rebalance"):
            mark_rebalance_failure(algo, symbol, "insuff_funds")
        return None
    ticket = _safe_submit_order(algo, symbol, quantity, lambda: algo.LimitOrder(symbol, quantity, limit_price, tag=tag))
    if ticket is None:
        return None
    if quantity > 0:
        algo._orders_today = int(getattr(algo, "_orders_today", 0) or 0) + 1
        if str(tag).startswith("Rebalance"):
            clear_rebalance_failure(algo, symbol)
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
    if position_status(algo, symbol) != "flat":
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
    pstate = _state_get(algo, symbol)
    entry_price = getattr(pstate, "entry_price", None)
    if record_pnl and entry_price and exit_price and entry_price > 0 and exit_price > 0:
        pnl = (exit_price - entry_price) / entry_price - get_effective_round_trip_fee(algo)
        algo.pnl_by_tag.setdefault(getattr(algo, "_last_exit_tag", "Unknown"), []).append(pnl)
        algo.pnl_by_regime.setdefault(getattr(algo, "market_regime", "unknown"), []).append(pnl)
    _state_pop(algo, symbol)


def smart_liquidate(algo, symbol, tag="Liquidate"):
    if symbol not in algo.Portfolio:
        return False
    qty = _free_qty(algo, symbol)
    if qty <= 0:
        return False
    qty = round_quantity(algo, symbol, qty)
    if qty <= 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = float(getattr(algo.Securities.get(symbol), "Price", 0.0) or 0.0)
    if qty < min_qty or (price > 0 and qty * price < min_notional):
        if not hasattr(algo, "_abandoned_dust"):
            algo._abandoned_dust = set()
        already_dust = symbol in algo._abandoned_dust
        algo._abandoned_dust.add(symbol)
        _state_pop(algo, symbol)
        if not already_dust:
            debug_limited(
                algo,
                (
                    "DUST key=skip "
                    f"sym={getattr(symbol, 'Value', symbol)} "
                    f"qty={qty:.10f} min_qty={min_qty:.10f} "
                    f"notional={qty * price:.6f} min_notional={min_notional:.4f}"
                ),
            )
            debug_limited(
                algo,
                (
                    "DUST_PURGE "
                    f"sym={getattr(symbol, 'Value', symbol)} "
                    f"residual_qty={qty:.10f}"
                ),
            )
        return False
    ticket = place_limit_or_market(algo, symbol, -qty, tag=tag)
    return ticket is not None


def liquidate_all_positions(algo, tag="Liquidate"):
    if not hasattr(algo, "_abandoned_dust"):
        algo._abandoned_dust = set()
    closed = []
    for item in list(getattr(algo, "Portfolio", [])):
        symbol = getattr(item, "Key", item)
        if symbol in algo._abandoned_dust:
            continue
        try:
            qty = float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        if not is_invested_not_dust(algo, symbol):
            algo._abandoned_dust.add(symbol)
            continue
        if qty > 0 and smart_liquidate(algo, symbol, tag=tag):
            closed.append(symbol)
    return closed


def manage_open_positions(algo, data=None):
    exits = []
    holdings = []
    for kv in getattr(algo, "Portfolio", []):
        if hasattr(kv, "Key"):
            symbol = kv.Key
            holding = getattr(kv, "Value", None)
        else:
            symbol = kv
            holding = None
        if holding is None or not hasattr(holding, "Quantity"):
            try:
                holding = algo.Portfolio[symbol]
            except Exception:
                holding = None
        try:
            qty = float(getattr(holding, "Quantity", 0.0) or 0.0)
        except Exception:
            qty = 0.0
        if qty > 0:
            holdings.append(symbol)

    holding_set = set(holdings)
    for symbol in [s for s in list(_position_state(algo).keys()) if hasattr(s, "Value")]:
        if symbol not in holding_set and position_status(algo, symbol) == "flat":
            cleanup_position(algo, symbol)
    for symbol in holdings:
        state = _state_get(algo, symbol)
        if state is not None and getattr(state, "strategy_owner", "momentum") == "scalper":
            continue
        if _state_get(algo, symbol) is None:
            try:
                avg_px = float(algo.Portfolio[symbol].AveragePrice or 0.0)
            except Exception:
                avg_px = 0.0
            if avg_px <= 0:
                sec = algo.Securities.get(symbol)
                avg_px = float(getattr(sec, "Price", 0.0) or 0.0)
            if avg_px <= 0:
                continue
            feats = algo.feature_engine.current_features(getattr(symbol, "Value", str(symbol))) or {}
            atr = float(feats.get("atr", 0.0) or 0.0)
            if atr <= 0:
                atr = avg_px * DEFAULT_LAZY_ATR_PCT
            state_new = PositionState(
                entry_price=avg_px,
                highest_close=avg_px,
                entry_atr=atr,
                entry_time=algo.Time - timedelta(hours=1),
            )
            _position_state(algo)[symbol] = state_new
            _position_state(algo)[getattr(symbol, "Value", str(symbol))] = state_new
            algo.Debug(
                f"LAZY_SEED sym={getattr(symbol,'Value',symbol)} "
                f"entry_time_set={algo.Time - timedelta(hours=1)} avg_px={avg_px:.6f} atr={atr:.6f} reason=missing_state"
            )
        pstate = _state_get(algo, symbol)
        if pstate is None:
            continue
        sec = algo.Securities.get(symbol)
        if sec is None:
            continue
        price = float(getattr(sec, "Price", 0.0) or 0.0)
        if price <= 0:
            continue
        state = pstate
        if state.entry_time is not None:
            hours_held = (algo.Time - state.entry_time).total_seconds() / 3600.0
        else:
            hours_held = 0.0
        time_stop_hours = float(getattr(getattr(algo, "config", CONFIG), "time_stop_hours", 120.0))
        if hours_held >= time_stop_hours:
            if smart_liquidate(algo, symbol, tag="TimeStop"):
                algo.Debug(f"TIME_STOP sym={getattr(symbol,'Value',symbol)} hours={hours_held:.1f}")
                cleanup_position(algo, symbol, record_pnl=True, exit_price=price)
                exits.append((symbol, "TimeStop"))
                continue
        state.highest_close = max(state.highest_close, price)
        sl_mult = float(getattr(getattr(algo, "config", CONFIG), "sl_atr_multiplier", 1.5))
        tp_mult = float(getattr(getattr(algo, "config", CONFIG), "tp_atr_multiplier", 3.0))
        sl_price = state.entry_price - sl_mult * state.entry_atr
        chandelier_active = (state.highest_close - state.entry_price) >= state.entry_atr
        chandelier_stop = state.highest_close - 2.0 * state.entry_atr if chandelier_active else None
        effective_stop = max(sl_price, chandelier_stop) if chandelier_stop is not None else sl_price
        tp_price = state.entry_price + tp_mult * state.entry_atr
        if price <= effective_stop:
            tag = "Chandelier" if chandelier_active and chandelier_stop is not None and effective_stop == chandelier_stop else "SL"
            if smart_liquidate(algo, symbol, tag=tag):
                algo.Debug(f"{tag} sym={getattr(symbol,'Value',symbol)} px={price:.4f} stop={effective_stop:.4f}")
                cleanup_position(algo, symbol, record_pnl=True, exit_price=price)
                exits.append((symbol, tag))
                continue
        elif price >= tp_price:
            if smart_liquidate(algo, symbol, tag="TP"):
                algo.Debug(f"TP sym={getattr(symbol,'Value',symbol)} px={price:.4f} tp={tp_price:.4f}")
                cleanup_position(algo, symbol, record_pnl=True, exit_price=price)
                exits.append((symbol, "TP"))
                continue
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
        if position_status(algo, symbol) != "flat":
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
        intent = str(info.get("intent", "") or "")
        if qty == 0:
            algo._submitted_orders.pop(symbol, None)
            continue
        if intent == "exit" and has_open_exit_order(algo, symbol):
            algo._submitted_orders.pop(symbol, None)
            continue
        replacement = place_limit_or_market(algo, symbol, qty, tag="[StaleEsc]", force_market=True)
        algo._submitted_orders.pop(symbol, None)
        if replacement is None:
            algo.Debug(f"STALE_SKIP sym={getattr(symbol, 'Value', symbol)} reason=replacement_skipped")
            continue
        escalated.append(symbol)
    return escalated


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
