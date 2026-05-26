from __future__ import annotations

from config import CONFIG
from risk import PositionRisk


def _order_tag(order) -> str:
    return str(getattr(order, "Tag", "") or getattr(order, "tag", "") or "")


def _open_orders(algo, symbol) -> list:
    try:
        return list(algo.Transactions.GetOpenOrders(symbol))
    except Exception:
        return []


def set_bracket_prices(state: PositionRisk, entry_price: float, atr: float) -> None:
    atr = max(float(atr), entry_price * 0.008)
    state.stop_price = entry_price - float(CONFIG.sl_atr_mult) * atr
    state.take_profit_price = entry_price + float(CONFIG.tp_atr_mult) * atr


def sync_brackets(algo, symbol, state: PositionRisk, qty: float) -> dict:
    """Attach SL stop-market + TP limit for long spot (Kraken via QC)."""
    if not bool(getattr(CONFIG, "enable_brackets", True)):
        return {"has_sl": False, "has_tp": False}
    qty_abs = abs(float(qty))
    if qty_abs <= 0:
        return {"has_sl": True, "has_tp": True}
    stop_px = float(getattr(state, "stop_price", 0.0) or 0.0)
    tp_px = float(getattr(state, "take_profit_price", 0.0) or 0.0)
    if stop_px <= 0 or tp_px <= 0:
        return {"has_sl": False, "has_tp": False}

    exit_qty = -qty_abs
    has_sl = False
    has_tp = False
    for order in _open_orders(algo, symbol):
        tag = _order_tag(order)
        if tag == "SL":
            has_sl = True
        if tag == "TP":
            has_tp = True

    if not has_sl:
        try:
            algo.StopMarketOrder(symbol, exit_qty, stop_px, tag="SL")
            has_sl = True
        except Exception as exc:
            if hasattr(algo, "Debug"):
                algo.Debug(f"BRACKET_SL_FAIL {symbol} {exc}")
    if not has_tp:
        try:
            algo.LimitOrder(symbol, exit_qty, tp_px, tag="TP")
            has_tp = True
        except Exception as exc:
            if hasattr(algo, "Debug"):
                algo.Debug(f"BRACKET_TP_FAIL {symbol} {exc}")
    return {"has_sl": has_sl, "has_tp": has_tp}
