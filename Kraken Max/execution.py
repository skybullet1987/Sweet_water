from __future__ import annotations

from datetime import datetime

from config import CONFIG
from risk import PositionRisk, should_exit
from sizing import free_cash_usd, round_qty


def get_min_qty(algo, symbol) -> float:
    ticker = symbol.Value if hasattr(symbol, "Value") else str(symbol)
    try:
        sec = algo.Securities[symbol]
        min_size = float(getattr(sec.SymbolProperties, "MinimumOrderSize", 0.0) or 0.0)
        if min_size > 0:
            return min_size
    except Exception:
        pass
    return float(CONFIG.min_qty_fallback.get(ticker, 0.0001))


def position_qty(algo, symbol) -> float:
    try:
        return float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
    except Exception:
        return 0.0


def market_value(algo, symbol) -> float:
    try:
        return float(getattr(algo.Portfolio[symbol], "HoldingsValue", 0.0) or 0.0)
    except Exception:
        qty = position_qty(algo, symbol)
        price = float(algo.Securities[symbol].Price)
        return qty * price


def place_buy(algo, symbol, usd_notional: float) -> bool:
    price = float(algo.Securities[symbol].Price)
    if price <= 0 or usd_notional <= 0:
        return False
    min_qty = get_min_qty(algo, symbol)
    qty = round_qty(usd_notional / price, min_qty)
    if qty < min_qty:
        return False
    cash = free_cash_usd(algo)
    if qty * price > cash * 0.995:
        qty = round_qty(cash * 0.98 / price, min_qty)
    if qty < min_qty:
        return False
    algo.MarketOrder(symbol, qty)
    return True


def liquidate_symbol(algo, symbol) -> None:
    qty = position_qty(algo, symbol)
    if qty > 0:
        algo.MarketOrder(symbol, -qty)


def manage_exits(algo, symbol, state: PositionRisk, close: float, now: datetime) -> bool:
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
        liquidate_symbol(algo, symbol)
        if hasattr(algo, "Debug"):
            algo.Debug(f"KRAKEN_MAX exit {symbol} reason={reason}")
        return True
    return False
