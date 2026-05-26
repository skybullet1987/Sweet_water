from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from config import CONFIG
from risk import PositionRisk, should_exit
from scalper_sleeve import evaluate_scalper_exit
from sizing import can_afford, free_cash_usd, round_qty


POSITION_TOLERANCE = 1e-9


def get_min_qty(algo, symbol) -> float:
    ticker = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    try:
        sec = algo.Securities[symbol]
        min_size = float(getattr(sec.SymbolProperties, "MinimumOrderSize", 0.0) or 0.0)
        if min_size > 0:
            return min_size
        lot = float(getattr(sec.SymbolProperties, "LotSize", 0.0) or 0.0)
        if lot > 0:
            return lot
    except Exception:
        pass
    return float(CONFIG.min_qty_fallback.get(ticker, 0.0001))


def round_quantity(algo, symbol, quantity: float) -> float:
    try:
        lot = float(getattr(algo.Securities[symbol].SymbolProperties, "LotSize", 0.0) or 0.0)
    except Exception:
        lot = 0.0
    if lot <= 0:
        return round_qty(float(quantity), get_min_qty(algo, symbol))
    sign = 1.0 if quantity >= 0 else -1.0
    return sign * math.floor(abs(quantity) / lot) * lot


def position_qty(algo, symbol) -> float:
    try:
        return float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
    except Exception:
        return 0.0


def hourly_dollar_volume(algo, symbol) -> float:
    try:
        sec = algo.Securities[symbol]
        vol = float(getattr(sec, "Volume", 0.0) or 0.0)
        px = float(getattr(sec, "Price", 0.0) or 0.0)
        return max(vol * px, 0.0)
    except Exception:
        return 0.0


def cap_notional_by_participation(algo, symbol, usd_notional: float) -> float:
    hourly_vol = hourly_dollar_volume(algo, symbol)
    if hourly_vol <= 0:
        return usd_notional
    cap = hourly_vol * float(CONFIG.max_participation_rate)
    return min(usd_notional, max(cap, 0.0))


def is_price_stale(algo, symbol, price: float) -> bool:
    if not math.isfinite(price) or price <= 0:
        return True
    max_age = int(CONFIG.stale_price_minutes)
    try:
        sec = algo.Securities[symbol]
        now = algo.Time
        data = sec.GetLastData()
        last = getattr(data, "EndTime", None) or getattr(sec, "LocalTime", None)
        if last is None:
            return False
        return (now - last).total_seconds() / 60.0 > max_age
    except Exception:
        return False


def _limit_price(algo, symbol, quantity: float) -> float:
    sec = algo.Securities[symbol]
    price = float(getattr(sec, "Price", 0.0) or 0.0)
    bid = float(getattr(sec, "BidPrice", 0.0) or 0.0)
    ask = float(getattr(sec, "AskPrice", 0.0) or 0.0)
    if quantity > 0 and bid > 0:
        return bid
    if quantity < 0 and ask > 0:
        return ask
    return price


def place_limit_or_market(
    algo,
    symbol,
    quantity: float,
    *,
    tag: str = "Entry",
    force_market: bool = False,
) -> bool:
    quantity = round_quantity(algo, symbol, float(quantity))
    if quantity == 0:
        return False
    if quantity > 0 and getattr(algo, "long_only", True):
        pass
    elif quantity < 0:
        hold = position_qty(algo, symbol)
        quantity = -min(abs(quantity), hold)
        if quantity == 0:
            return False

    price_est = _limit_price(algo, symbol, quantity)
    if is_price_stale(algo, symbol, price_est):
        return False
    ok, _req, _avail = can_afford(algo, quantity, price_est)
    if quantity > 0 and not ok:
        return False

    use_limit = bool(CONFIG.use_limit_orders) and not force_market
    if use_limit and quantity != 0:
        limit_px = price_est
        ticket = algo.LimitOrder(symbol, quantity, limit_px, tag=tag)
        if ticket is None:
            return False
        if not hasattr(algo, "_pending_limits"):
            algo._pending_limits = {}
        algo._pending_limits[symbol] = {
            "order_id": getattr(ticket, "OrderId", None),
            "submitted": getattr(algo, "Time", datetime.now(timezone.utc)),
            "qty": quantity,
            "tag": tag,
        }
        return True
    algo.MarketOrder(symbol, quantity, tag=tag)
    return True


def place_buy_notional(algo, symbol, usd_notional: float, *, tag: str = "Entry", force_market: bool = False) -> bool:
    price = float(algo.Securities[symbol].Price)
    if price <= 0 or usd_notional <= 0:
        return False
    notional = cap_notional_by_participation(algo, symbol, usd_notional)
    min_qty = get_min_qty(algo, symbol)
    qty = round_quantity(algo, symbol, notional / price)
    if qty < min_qty:
        return False
    cash = free_cash_usd(algo)
    max_affordable = round_quantity(algo, symbol, (cash * float(CONFIG.cash_safety_factor)) / price)
    qty = min(qty, max_affordable)
    if qty < min_qty:
        return False
    return place_limit_or_market(algo, symbol, qty, tag=tag, force_market=force_market)


def liquidate_symbol(algo, symbol, *, force_market: bool = True) -> None:
    qty = position_qty(algo, symbol)
    if qty > 0:
        place_limit_or_market(algo, symbol, -qty, tag="Exit", force_market=force_market)


def escalate_stale_limits(algo) -> None:
    pending = getattr(algo, "_pending_limits", {}) or {}
    if not pending:
        return
    timeout = timedelta(seconds=int(CONFIG.limit_order_timeout_seconds))
    now = getattr(algo, "Time", datetime.now(timezone.utc))
    for symbol, meta in list(pending.items()):
        submitted = meta.get("submitted")
        if submitted is None or (now - submitted) < timeout:
            continue
        order_id = meta.get("order_id")
        open_ids = set()
        try:
            for o in algo.Transactions.GetOpenOrders(symbol):
                open_ids.add(getattr(o, "Id", getattr(o, "OrderId", None)))
        except Exception:
            pass
        if order_id in open_ids:
            try:
                algo.Transactions.CancelOrder(order_id)
            except Exception:
                pass
        qty = float(meta.get("qty", 0.0))
        if qty != 0:
            place_limit_or_market(algo, symbol, qty, tag=str(meta.get("tag", "Escalate")), force_market=True)
        pending.pop(symbol, None)


def manage_exits(algo, symbol, state: PositionRisk, close: float, now: datetime, feats: dict | None = None) -> bool:
    if state.strategy_owner == "scalper":
        exit_now, reason = evaluate_scalper_exit(
            entry_price=state.entry_price,
            entry_time=state.entry_time,
            current_price=close,
            now=now,
            feats=feats or {},
            highest_close=state.highest_close,
            entry_atr=state.entry_atr,
        )
    else:
        exit_now, reason = should_exit(
            state,
            close=close,
            now=now,
            hard_stop_pct=float(CONFIG.hard_stop_pct),
            catastrophic_stop_pct=float(CONFIG.catastrophic_stop_pct),
            tp_atr_mult=float(CONFIG.tp_atr_mult),
            sl_atr_mult=float(CONFIG.sl_atr_mult),
            chandelier_atr_mult=float(CONFIG.chandelier_atr_mult),
            activate_trail_pct=float(CONFIG.activate_trail_above_pct),
            time_stop_hours=float(CONFIG.time_stop_hours),
        )
    if exit_now:
        liquidate_symbol(algo, symbol, force_market=True)
        if hasattr(algo, "Debug"):
            algo.Debug(f"KRAKEN_MAX exit {symbol} owner={state.strategy_owner} reason={reason}")
        return True
    return False
