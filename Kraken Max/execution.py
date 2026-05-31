"""Kraken Max — order execution (`execution.py`)."""
from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import CONFIG
from risk import PositionRisk, should_exit

# --- from execution.py ---


from config import CONFIG
from risk import PositionRisk, should_exit
from core import can_afford, evaluate_scalper_exit, free_cash_usd, round_qty


POSITION_TOLERANCE = 1e-9


def _order_tag(order) -> str:
    return str(getattr(order, "Tag", "") or getattr(order, "tag", "") or "")


def _open_orders(algo, symbol) -> list:
    try:
        return list(algo.Transactions.GetOpenOrders(symbol))
    except Exception:
        return []


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


def effective_lot_size(algo, symbol) -> float:
    lot = 0.0
    min_size = 0.0
    try:
        sp = algo.Securities[symbol].SymbolProperties
        lot = float(getattr(sp, "LotSize", 0.0) or 0.0)
        min_size = float(getattr(sp, "MinimumOrderSize", 0.0) or 0.0)
    except Exception:
        pass
    if lot <= 0 and min_size > 0:
        lot = min_size
    if lot <= 0:
        lot = get_min_qty(algo, symbol)
    return max(lot, 1e-6)


def floor_holdings_to_lot(hold: float, lot: float) -> float:
    if hold <= 0:
        return 0.0
    if lot <= 0:
        return 0.0
    return math.floor(hold / lot + 1e-12) * lot


def hold_is_dust(algo, symbol) -> bool:
    """True when holdings are below one lot or economic minimum — do not place exits."""
    if is_dust_symbol(algo, symbol):
        return True
    hold = max(0.0, position_qty(algo, symbol))
    if hold <= POSITION_TOLERANCE:
        return True
    lot = effective_lot_size(algo, symbol)
    if hold + 1e-15 < lot:
        return True
    try:
        px = float(algo.Securities[symbol].Price)
    except Exception:
        px = 0.0
    dust_usd = float(getattr(CONFIG, "dust_notional_usd", 1.0) or 1.0)
    if px > 0 and hold * px < dust_usd:
        return True
    return False


def mark_dust_symbol(algo, symbol, reason: str = "") -> None:
    dust = getattr(algo, "_abandoned_dust", None)
    if dust is None:
        algo._abandoned_dust = set()
        dust = algo._abandoned_dust
    if symbol not in dust and hasattr(algo, "Debug") and reason:
        algo.Debug(f"KM_DUST {getattr(symbol, 'Value', symbol)} {reason}")
    dust.add(symbol)


def is_dust_symbol(algo, symbol) -> bool:
    return symbol in getattr(algo, "_abandoned_dust", set())


def round_quantity(algo, symbol, quantity: float) -> float:
    lot = effective_lot_size(algo, symbol)
    sign = 1.0 if quantity >= 0 else -1.0
    return sign * floor_holdings_to_lot(abs(float(quantity)), lot)


def position_qty(algo, symbol) -> float:
    try:
        return float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
    except Exception:
        return 0.0


def _order_signed_qty(order) -> float:
    qty = float(getattr(order, "Quantity", 0.0) or 0.0)
    if qty != 0.0:
        return qty
    abs_qty = float(getattr(order, "AbsoluteQuantity", 0.0) or 0.0)
    if abs_qty <= 0:
        return 0.0
    direction = getattr(order, "Direction", None)
    if direction is not None:
        try:
            from AlgorithmImports import OrderDirection

            if direction == OrderDirection.Sell:
                return -abs_qty
            if direction == OrderDirection.Buy:
                return abs_qty
        except Exception:
            pass
        name = str(getattr(direction, "name", direction) or "").lower()
        if "sell" in name:
            return -abs_qty
        if "buy" in name:
            return abs_qty
    return 0.0


def reserved_sell_qty(algo, symbol) -> float:
    """Quantity already committed by open sell/stop orders (cash model)."""
    reserved = 0.0
    for order in _open_orders(algo, symbol):
        qty = _order_signed_qty(order)
        if qty < 0:
            reserved += abs(qty)
    return max(0.0, reserved)


def available_sell_qty(algo, symbol) -> float:
    hold = max(0.0, position_qty(algo, symbol))
    return max(0.0, hold - reserved_sell_qty(algo, symbol))


def sellable_qty_for_exit(algo, symbol) -> float:
    """Sell qty floored to lot steps — never round up above portfolio (cash/dust safe)."""
    if hold_is_dust(algo, symbol):
        mark_dust_symbol(algo, symbol, "sellable_dust")
        return 0.0
    avail = available_sell_qty(algo, symbol)
    if avail <= POSITION_TOLERANCE:
        return 0.0
    lot = effective_lot_size(algo, symbol)
    qty = floor_holdings_to_lot(avail, lot)
    if qty <= POSITION_TOLERANCE:
        mark_dust_symbol(algo, symbol, f"hold={avail:.8f}<lot={lot:.8f}")
        return 0.0
    buffer_lots = int(getattr(CONFIG, "exit_sell_buffer_lots", 1) or 1)
    if qty > buffer_lots * lot:
        qty = floor_holdings_to_lot(qty - buffer_lots * lot, lot)
    if qty <= POSITION_TOLERANCE:
        mark_dust_symbol(algo, symbol, "buffer_zero")
        return 0.0
    return min(qty, avail)


def cancel_open_orders(algo, symbol) -> None:
    for order in _open_orders(algo, symbol):
        oid = getattr(order, "Id", None) or getattr(order, "OrderId", None)
        if oid is None:
            continue
        try:
            algo.Transactions.CancelOrder(oid)
        except Exception:
            pass


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
    raw = float(quantity)
    if raw < 0:
        if hold_is_dust(algo, symbol):
            mark_dust_symbol(algo, symbol, "sell_blocked_dust")
            return False
        hold = max(0.0, position_qty(algo, symbol))
        avail = available_sell_qty(algo, symbol)
        lot = effective_lot_size(algo, symbol)
        sell_abs = floor_holdings_to_lot(min(abs(raw), avail, hold), lot)
        if sell_abs <= POSITION_TOLERANCE:
            mark_dust_symbol(algo, symbol, "sell_zero")
            return False
        quantity = -sell_abs
    else:
        quantity = round_quantity(algo, symbol, raw)
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


def _exit_blocked(algo, symbol) -> bool:
    until = (getattr(algo, "_exit_fail_until", {}) or {}).get(symbol)
    now = getattr(algo, "Time", None)
    return until is not None and now is not None and now < until


def _mark_exit_fail(algo, symbol) -> None:
    hours = float(getattr(CONFIG, "exit_retry_cooldown_hours", 12.0) or 12.0)
    now = getattr(algo, "Time", datetime.now(timezone.utc))
    if not hasattr(algo, "_exit_fail_until"):
        algo._exit_fail_until = {}
    algo._exit_fail_until[symbol] = now + timedelta(hours=hours)


def _order_status_text(algo, order_id) -> str:
    if order_id is None:
        return ""
    try:
        order = algo.Transactions.GetOrderById(int(order_id))
        return str(getattr(order, "Status", "") or "")
    except Exception:
        return ""


def _status_filled(status: str) -> bool:
    return "fill" in str(status).lower()


def _limit_buy_already_filled(algo, symbol, pending_qty: float) -> bool:
    if pending_qty <= 0:
        return False
    return position_qty(algo, symbol) >= pending_qty - POSITION_TOLERANCE


def clear_pending_limit(algo, symbol, order_id=None) -> None:
    pending = getattr(algo, "_pending_limits", None)
    if not pending:
        return
    meta = pending.get(symbol)
    if meta is None:
        return
    if order_id is None or meta.get("order_id") == order_id:
        pending.pop(symbol, None)


def place_buy_notional(algo, symbol, usd_notional: float, *, tag: str = "Entry", force_market: bool = False) -> bool:
    if symbol in (getattr(algo, "_pending_limits", {}) or {}):
        return False
    if position_qty(algo, symbol) > POSITION_TOLERANCE:
        return False
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


def liquidate_symbol(
    algo,
    symbol,
    *,
    force_market: bool = True,
    tag: str = "Exit",
    emergency: bool = False,
) -> bool:
    if hold_is_dust(algo, symbol):
        mark_dust_symbol(algo, symbol, "liquidate_dust")
        return False
    if not emergency and _exit_blocked(algo, symbol):
        return False
    cancel_open_orders(algo, symbol)
    qty = sellable_qty_for_exit(algo, symbol)
    if qty <= POSITION_TOLERANCE:
        return False
    ok = place_limit_or_market(algo, symbol, -qty, tag=tag, force_market=force_market)
    if not ok:
        _mark_exit_fail(algo, symbol)
    return ok


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
        qty = float(meta.get("qty", 0.0))
        tag = str(meta.get("tag", "Escalate"))

        status = _order_status_text(algo, order_id)
        if _status_filled(status) or _limit_buy_already_filled(algo, symbol, qty):
            pending.pop(symbol, None)
            continue

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
            if _limit_buy_already_filled(algo, symbol, qty):
                pending.pop(symbol, None)
                continue
        elif order_id is not None and not status:
            if _limit_buy_already_filled(algo, symbol, qty):
                pending.pop(symbol, None)
                continue

        if qty > 0 and _limit_buy_already_filled(algo, symbol, qty):
            pending.pop(symbol, None)
            continue
        held = position_qty(algo, symbol)
        remaining = max(0.0, qty - held)
        min_qty = get_min_qty(algo, symbol)
        if remaining >= min_qty:
            place_limit_or_market(algo, symbol, remaining, tag=tag, force_market=True)
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
        emergency = reason in ("catastrophic", "hard_stop") or str(reason).startswith("hard")
        liquidate_symbol(
            algo,
            symbol,
            force_market=True,
            tag=f"KM:{state.strategy_owner}:Exit",
            emergency=emergency,
        )
        if hasattr(algo, "Debug"):
            algo.Debug(f"KRAKEN_MAX exit_submit {symbol} owner={state.strategy_owner} reason={reason}")
    return False

# --- from execution_bridge.py ---


import importlib.util
import sys
import types
from pathlib import Path

_KRAKEN_MAX = Path(__file__).resolve().parent
_REPO = _KRAKEN_MAX.parent


def _resolve_qc_runtime_dir() -> Path:
    """QC cloud: files are usually flat in the project; qc_runtime/ is a subfolder."""
    for candidate in (_KRAKEN_MAX / "qc_runtime", _REPO / "qc_runtime"):
        if (candidate / "execution.py").is_file():
            return candidate
    return _KRAKEN_MAX / "qc_runtime"


_QC_RUNTIME = _resolve_qc_runtime_dir()

_qc_execution = None
_USE_PRO = False


def _load_module(name: str, path: Path, prepend: Path):
    saved = sys.path[:]
    try:
        sys.path = [str(prepend)] + [p for p in sys.path if p not in {str(prepend), str(_KRAKEN_MAX)}]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path = saved


try:
    _qc_execution = _load_module("qc_runtime_execution", _QC_RUNTIME / "execution.py", _QC_RUNTIME)
    _USE_PRO = True
except Exception:
    _qc_execution = None
    _USE_PRO = False

_local_execution = types.SimpleNamespace(
    place_buy_notional=place_buy_notional,
    liquidate_symbol=liquidate_symbol,
    escalate_stale_limits=escalate_stale_limits,
    manage_exits=manage_exits,
    position_qty=position_qty,
)


def init_execution_state(algo) -> None:
    algo._submitted_orders = getattr(algo, "_submitted_orders", {})
    algo._pending_limits = getattr(algo, "_pending_limits", {})
    algo._abandoned_dust = getattr(algo, "_abandoned_dust", set())
    algo._failed_escalations = getattr(algo, "_failed_escalations", {})
    algo._exit_fail_until = getattr(algo, "_exit_fail_until", {})
    algo.stale_order_bars = int(getattr(getattr(algo, "config", None), "stale_order_bars", 3) or 3)
    algo.min_notional = float(getattr(getattr(algo, "config", None), "min_position_floor_usd", 25.0) or 25.0)


def track_order_submit(algo, ticket, *, symbol, qty: float, expected_price: float, force_market: bool = False) -> None:
    """Notify FillTracker when a limit (or market) order is submitted (v6)."""
    ft = getattr(algo, "fill_tracker", None)
    if ft is None or ticket is None:
        return
    oid = int(getattr(ticket, "OrderId", 0) or 0)
    if oid <= 0:
        return
    is_limit = not force_market and bool(getattr(getattr(algo, "config", None), "use_limit_orders", True))
    ft.on_submit(oid, is_limit=is_limit, expected_price=float(expected_price), qty=abs(float(qty)))


def place_buy_notional(algo, symbol, usd_notional: float, *, tag: str = "Entry", force_market: bool = False) -> bool:
    if _USE_PRO and _qc_execution is not None:
        from config import CONFIG as KM_CONFIG

        price = float(algo.Securities[symbol].Price)
        if price <= 0:
            return False
        hourly = float(getattr(algo.Securities[symbol], "Volume", 0.0) or 0.0) * price
        cap = hourly * float(KM_CONFIG.max_participation_rate)
        notional = min(float(usd_notional), cap) if cap > 0 else float(usd_notional)
        qty = _qc_execution.round_quantity(algo, symbol, notional / price)
        if qty <= 0:
            return False
        ticket = _qc_execution.place_entry(algo, symbol, qty, tag=tag, force_market=force_market)
        if ticket is not None:
            track_order_submit(algo, ticket, symbol=symbol, qty=qty, expected_price=price, force_market=force_market)
        return ticket is not None
    if _local_execution is not None:
        ok = bool(_local_execution.place_buy_notional(algo, symbol, usd_notional, tag=tag, force_market=force_market))
        if ok:
            pending = (getattr(algo, "_pending_limits", {}) or {}).get(symbol)
            if pending:
                px = float(algo.Securities[symbol].Price)
                order_qty = float(pending.get("qty", 0.0))
                if order_qty > 0 and px > 0:
                    track_order_submit(
                        algo,
                        type("T", (), {"OrderId": pending.get("order_id")})(),
                        symbol=symbol,
                        qty=order_qty,
                        expected_price=px,
                        force_market=force_market,
                    )
        return ok
    return False


def liquidate_symbol(
    algo,
    symbol,
    *,
    force_market: bool = True,
    tag: str = "Exit",
    emergency: bool = False,
) -> bool:
    # Always prefer local cash-safe liquidate (qc_runtime smart_liquidate can oversell dust).
    if _local_execution is not None:
        return bool(
            _local_execution.liquidate_symbol(
                algo, symbol, force_market=force_market, tag=tag, emergency=emergency
            )
        )
    if _USE_PRO and _qc_execution is not None:
        return bool(_qc_execution.smart_liquidate(algo, symbol, tag=tag))
    return False


def escalate_orders(algo) -> list:
    if _USE_PRO and _qc_execution is not None:
        return list(_qc_execution.escalate_stale_orders(algo) or [])
    if _local_execution is not None:
        _local_execution.escalate_stale_limits(algo)
    return []


def manage_position_exit(algo, symbol, state, close: float, now, feats: dict | None = None) -> bool:
    if _local_execution is not None:
        return _local_execution.manage_exits(algo, symbol, state, close, now, feats)
    return False


def position_qty(algo, symbol) -> float:
    if _USE_PRO and _qc_execution is not None:
        try:
            return float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
        except Exception:
            return 0.0
    if _local_execution is not None:
        return _local_execution.position_qty(algo, symbol)
    return 0.0


def estimate_fee_pct(algo, notional: float, is_limit: bool = True) -> float:
    if _USE_PRO and _qc_execution is not None:
        model = _qc_execution.KrakenTieredFeeModel(comparison_mode=True)
        return float(model.estimate_round_trip_cost("BTCUSD", notional, is_limit=is_limit) / max(notional, 1e-9))
    from config import CONFIG

    return float(CONFIG.expected_round_trip_fees)

# --- from brackets.py ---


def set_bracket_prices(state: PositionRisk, entry_price: float, atr: float) -> None:
    atr = max(float(atr), entry_price * 0.008)
    state.stop_price = entry_price - float(CONFIG.sl_atr_mult) * atr
    state.take_profit_price = entry_price + float(CONFIG.tp_atr_mult) * atr


def sync_brackets(algo, symbol, state: PositionRisk, qty: float) -> dict:
    """Stop-market SL only on cash spot — full SL+TP limits double-book holdings."""
    if not bool(getattr(CONFIG, "enable_brackets", False)):
        return {"has_sl": False, "has_tp": False}
    qty_abs = abs(float(qty))
    if qty_abs <= POSITION_TOLERANCE:
        return {"has_sl": True, "has_tp": True}
    stop_px = float(getattr(state, "stop_price", 0.0) or 0.0)
    if stop_px <= 0:
        return {"has_sl": False, "has_tp": False}

    free = available_sell_qty(algo, symbol)
    exit_qty = round_quantity(algo, symbol, -min(qty_abs, free))
    if exit_qty == 0:
        return {"has_sl": False, "has_tp": False}

    has_sl = False
    has_tp = False
    for order in _open_orders(algo, symbol):
        tag = _order_tag(order)
        if tag == "SL":
            has_sl = True
        if tag == "TP":
            try:
                oid = getattr(order, "Id", None) or getattr(order, "OrderId", None)
                if oid is not None:
                    algo.Transactions.CancelOrder(oid)
            except Exception:
                pass

    if not has_sl:
        try:
            algo.StopMarketOrder(symbol, exit_qty, stop_px, tag="SL")
            has_sl = True
        except Exception as exc:
            if hasattr(algo, "Debug"):
                algo.Debug(f"BRACKET_SL_FAIL {symbol} {exc}")
    return {"has_sl": has_sl, "has_tp": has_tp}