# region imports
from AlgorithmImports import *
import json
import math
import numpy as np
import random
from collections import deque
from datetime import timedelta
# endregion

# Seed once at module load for deterministic backtests (backtest-only; non-fill simulation gated on not algo.LiveMode).
# Default seed 42 preserves existing behaviour. Call reseed_non_fill_simulation() in Initialize to override.
random.seed(42)

# Additive non-fill penalty for breakout/momentum entries (vol-ignition-only, no mean-reversion support).
# Configurable via algo.breakout_nonfill_penalty; conservative default 0.08 (+8 pp above base rate).
_BREAKOUT_NONFILL_PENALTY_DEFAULT = 0.08
_BACKTEST_ENTRY_ADVERSE_OFFSET_DEFAULT = 0.0018
_BACKTEST_ENTRY_NOQUOTE_OFFSET_DEFAULT = 0.0022


def reseed_non_fill_simulation(seed):
    """Re-seed the module-level RNG used for non-fill simulation.

    Call this from QCAlgorithm.Initialize() to make the non-fill seed configurable
    while preserving deterministic behaviour (default seed = 42).
    """
    random.seed(int(seed))

# Round-trip fee estimate for PnL tracking (Kelly/circuit-breakers). Kraken 0.25% maker + 0.40% taker = 0.65%.
ESTIMATED_ROUND_TRIP_FEE = 0.0065  # 0.65% round-trip

# STRUCTURAL blacklist: a-priori structural exclusions (not look-ahead biased).
SYMBOL_BLACKLIST_STRUCTURAL = {
    "BTCUSD",  # Reference symbol only — never trade (too expensive for small capital, always dust)
    "USDTUSD", "USDCUSD", "PYUSDUSD", "EURCUSD", "USTUSD",
    "DAIUSD", "TUSDUSD", "WETHUSD", "WBTCUSD", "WAXLUSD",
    "SHIBUSD", "XMRUSD", "ZECUSD", "DASHUSD",
    "XNYUSD",
    "BDXNUSD", "RAIINUSD", "LUNAUSD", "LUNCUSD", "USTCUSD", "ABORDUSD",
    "BONDUSD", "KEEPUSD", "ORNUSD",
    "MUSD", "ICNTUSD",
    "EPTUSD", "LMWRUSD",
    "CPOOLUSD",
    "ARCUSD", "PAXGUSD",
    "PARTIUSD", "RAREUSD", "BANANAS31USD",
    # Micro-caps / meme coins: insufficient capacity for live execution
    "SEIUSD", "WIFUSD", "BONKUSD", "PEPEUSD", "FLOKIUSD", "ORDIUSD", "TIAUSD",
    # Forex pairs
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "JPYUSD", "CADUSD", "CHFUSD", "CNYUSD", "HKDUSD", "SGDUSD",
    "SEKUSD", "NOKUSD", "DKKUSD", "KRWUSD", "TRYUSD", "ZARUSD", "MXNUSD", "INRUSD", "BRLUSD",
    "PLNUSD", "THBUSD",
}

# BACKTEST-DERIVED blacklist: look-ahead bias — excluded from observed backtest losses. Keep False for honest runs.
SYMBOL_BLACKLIST_BACKTEST_DERIVED = {
    "XCNUSD",    # Catastrophic stop-loss losses (e.g. -$5,386 and -$1.9M single trades)
    "KTAUSD",    # Immediate stop losses on first trades
    "NANOUSD",   # Immediate stop losses on first trades
    "FWOGUSD",   # Repeated breakeven stops and poor fills
    "STRKUSD",   # Repeated breakeven stops and poor fills
}

# Toggle: False = honest backtest (default). True = look-ahead-biased mode (removes observed losses retroactively).
INCLUDE_BACKTEST_DERIVED_BLACKLIST = False

# Combined blacklist.
SYMBOL_BLACKLIST = (
    SYMBOL_BLACKLIST_STRUCTURAL | SYMBOL_BLACKLIST_BACKTEST_DERIVED
    if INCLUDE_BACKTEST_DERIVED_BLACKLIST
    else SYMBOL_BLACKLIST_STRUCTURAL
)

# UTC hour thresholds for spread time-of-day multipliers (_estimate_backtest_spread).
_TOD_ASIAN_END = 8; _TOD_EU_END = 13; _TOD_US_END = 21  # session boundaries
# Backtest queue-priority: participation-rate rejection constants.
_QUEUE_PARTICIPATION_THRESHOLD = 0.02; _QUEUE_REJECTION_SLOPE = 10.0; _QUEUE_MAX_REJECTION_PROB = 0.90

# Known fiat currency codes used to filter forex pairs from the crypto universe
KNOWN_FIAT_CURRENCIES = frozenset({
    "EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF", "CNY", "HKD", "SGD",
    "SEK", "NOK", "DKK", "KRW", "TRY", "ZAR", "MXN", "INR", "BRL", "PLN", "THB",
})

# Haircut before re-rounding sell qty: prevents floating-point overshoot above lot_size boundary.
QUANTITY_HAIRCUT_FACTOR = 0.9999

# Tolerance for IEEE 754 floating-point overshoot when comparing rounded qty to actual holding.
QUANTITY_OVERSHOOT_TOLERANCE = 1.0001

KRAKEN_MIN_QTY_FALLBACK = {
    'AXSUSD': 5.0, 'SANDUSD': 10.0, 'MANAUSD': 10.0, 'ADAUSD': 10.0,
    'MATICUSD': 10.0, 'DOTUSD': 1.0, 'LINKUSD': 0.5, 'AVAXUSD': 0.2,
    'ATOMUSD': 0.5, 'NEARUSD': 1.0, 'SOLUSD': 0.05,
    'ALGOUSD': 10.0, 'XLMUSD': 30.0, 'TRXUSD': 50.0, 'ENJUSD': 10.0,
    'BATUSD': 10.0, 'CRVUSD': 5.0, 'SNXUSD': 3.0, 'COMPUSD': 0.1,
    'AAVEUSD': 0.05, 'MKRUSD': 0.01, 'YFIUSD': 0.001, 'UNIUSD': 1.0,
    'SUSHIUSD': 5.0, '1INCHUSD': 5.0, 'GRTUSD': 10.0, 'FTMUSD': 10.0,
    'IMXUSD': 5.0, 'APEUSD': 2.0, 'GMTUSD': 10.0, 'OPUSD': 5.0,
    'LDOUSD': 5.0, 'ARBUSD': 5.0, 'LPTUSD': 5.0, 'KTAUSD': 10.0,
    'GUNUSD': 50.0, 'BANANAS31USD': 500.0, 'CHILLHOUSEUSD': 500.0,
    'PHAUSD': 50.0, 'MUSD': 50.0, 'ICNTUSD': 50.0,
    'SHIBUSD': 50000.0, 'XRPUSD': 2.0,
}

MIN_NOTIONAL_FALLBACK = {
    'EWTUSD': 2.0, 'SANDUSD': 8.0, 'CTSIUSD': 18.0, 'MKRUSD': 0.01,
    'AUDUSD': 10.0, 'LPTUSD': 0.3, 'OXTUSD': 40.0, 'ENJUSD': 15.0,
    'UNIUSD': 0.5, 'LSKUSD': 3.0, 'BCHUSD': 1.0,
}

# Fee buffer: prevents overselling when Kraken deducts fees from base asset at buy time.
KRAKEN_SELL_FEE_BUFFER = 0.006  # 0.6% (0.4% base fee + 0.2% safety margin)


def get_min_quantity(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    try:
        if symbol in algo.Securities:
            sec = algo.Securities[symbol]
            if hasattr(sec, 'SymbolProperties') and sec.SymbolProperties is not None:
                min_size = sec.SymbolProperties.MinimumOrderSize
                if min_size is not None and min_size > 0:
                    return float(min_size)
    except Exception as e:
        algo.Debug(f"Error getting min quantity for {ticker}: {e}")
        pass
    if ticker in KRAKEN_MIN_QTY_FALLBACK:
        return KRAKEN_MIN_QTY_FALLBACK[ticker]
    return estimate_min_qty(algo, symbol)


def estimate_min_qty(algo, symbol):
    try:
        price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    except Exception as e:
        algo.Debug(f"Error getting price for min qty estimate: {e}")
        price = 0
    if price <= 0: return 50.0
    if price < 0.001: return 1000.0
    elif price < 0.01: return 500.0
    elif price < 0.1: return 50.0
    elif price < 1.0: return 10.0
    elif price < 10.0: return 5.0
    elif price < 100.0: return 1.0
    elif price < 1000.0: return 0.1
    else: return 0.01


def get_min_notional_usd(algo, symbol):
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    fallback = MIN_NOTIONAL_FALLBACK.get(ticker, algo.min_notional)
    try:
        price = algo.Securities[symbol].Price
        min_qty = get_min_quantity(algo, symbol)
        implied = price * min_qty if price > 0 else fallback
        return max(fallback, implied, algo.min_notional)
    except Exception as e:
        algo.Debug(f"Error in get_min_notional_usd for {symbol.Value}: {e}")
        return max(fallback, algo.min_notional)


def round_quantity(algo, symbol, quantity):
    try:
        lot_size = algo.Securities[symbol].SymbolProperties.LotSize
        if lot_size is not None and lot_size > 0:
            rounded = float(math.floor(quantity / lot_size)) * lot_size
            # Clamp: never return more than the input (IEEE 754 safety)
            if rounded > quantity:
                rounded = rounded - lot_size
                if rounded < 0:
                    rounded = 0.0
            return rounded
        return quantity
    except Exception as e:
        algo.Debug(f"Error rounding quantity for {symbol.Value}: {e}")
        return quantity


def track_exit_order(algo, symbol, ticket, quantity):
    """Helper to track exit order in _submitted_orders for verification."""
    if hasattr(algo, '_submitted_orders') and ticket is not None:
        algo._submitted_orders[symbol] = {
            'order_id': ticket.OrderId,
            'time': algo.Time,
            'quantity': quantity,
            'intent': 'exit'
        }


def smart_liquidate(algo, symbol, tag="Liquidate"):
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    if symbol in algo._cancel_cooldowns and algo.Time < algo._cancel_cooldowns[symbol]:
        return False
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price * abs(holding_qty) < min_notional * 0.9:
        return False
    # Verify fee reserve before selling (Kraken cash account requirement)
    if algo.LiveMode and holding_qty > 0:
        estimated_fee = price * abs(holding_qty) * 0.006  # 0.6% fee estimate (0.4% base + 0.2% safety buffer)
        try:
            available_usd = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_usd = algo.Portfolio.Cash
        is_stop_loss = "Stop Loss" in tag or "Stop" in tag
        if available_usd < estimated_fee and not is_stop_loss:
            algo.Debug(f"⚠️ SKIP SELL {symbol.Value}: fee reserve too low "
                       f"(need ${estimated_fee:.4f}, have ${available_usd:.4f})")
            if symbol not in algo.entry_prices:
                algo.entry_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.highest_prices[symbol] = algo.Portfolio[symbol].AveragePrice
                algo.entry_times[symbol] = algo.Time
            return False
    algo.Transactions.CancelOpenOrders(symbol)
    # Use MinimumOrderSize (not lot_size) for exits — lot_size can be < MinimumOrderSize on some assets.
    try:
        sec = algo.Securities[symbol]
        min_order_size = sec.SymbolProperties.MinimumOrderSize
        if min_order_size is not None and min_order_size > 0:
            exit_min_qty = float(min_order_size)
        else:
            exit_min_qty = min_qty
    except Exception as e:
        algo.Debug(f"Warning: could not get MinimumOrderSize for {symbol.Value}: {e}")
        exit_min_qty = min_qty
    if abs(holding_qty) < exit_min_qty:
        return False
    # No sell-side fee buffer needed: portfolio already reflects post-fee quantity.
    safe_qty = round_quantity(algo, symbol, abs(holding_qty))
    # Apply haircut before re-rounding to prevent floating-point overshoot on Kraken Cash Modeling.
    actual_qty = abs(algo.Portfolio[symbol].Quantity)
    if safe_qty > actual_qty * QUANTITY_OVERSHOOT_TOLERANCE:  # tolerance for floating-point
        safe_qty = round_quantity(algo, symbol, actual_qty)
    # Hard cap: never exceed actual holding.
    safe_qty = min(safe_qty, actual_qty)
    if safe_qty < exit_min_qty:
        # Position rounded down below MinimumOrderSize — treat as dust, cannot sell
        return False
    if safe_qty > 0:
        direction_mult = -1 if holding_qty > 0 else 1
        # Spread-aware exit logic
        is_stop_loss = "Stop Loss" in tag
        if not is_stop_loss:
            spread_pct = get_spread_pct(algo, symbol)
            if spread_pct is not None:
                if spread_pct > 0.03:  # 3% spread - log warning but still exit with market
                    algo.Debug(f"⚠️ WIDE SPREAD EXIT: {symbol.Value} spread={spread_pct:.2%}, using market order")
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
                elif algo.LiveMode and spread_pct > 0.015:  # 1.5% spread - use limit order with fallback (live mode only)
                    try:
                        sec = algo.Securities[symbol]
                        bid = sec.BidPrice
                        ask = sec.AskPrice
                        if bid > 0 and ask > 0:
                            mid = 0.5 * (bid + ask)
                            limit_order = algo.LimitOrder(symbol, safe_qty * direction_mult, mid, tag=tag)
                            # Track for fallback in VerifyOrderFills (90 second timeout handled there)
                            if hasattr(algo, '_submitted_orders'):
                                algo._submitted_orders[symbol] = {
                                    'order_id': limit_order.OrderId,
                                    'time': algo.Time,
                                    'quantity': safe_qty * direction_mult,  # Store signed quantity
                                    'is_limit_exit': True,
                                    'intent': 'exit'
                                }
                            algo.Debug(f"LIMIT EXIT: {symbol.Value} at mid ${mid:.4f} (spread={spread_pct:.2%})")
                            return True
                        else:
                            ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                            track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                            return True
                    except Exception as e:
                        algo.Debug(f"Error placing limit exit for {symbol.Value}: {e}")
                        ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                        track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                        return True
                else:
                    # Backtest realism: by default, don't assume maker-favorable exit fills.
                    if not algo.LiveMode and getattr(algo, 'backtest_use_market_exits', True):
                        ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    else:
                        exit_price = algo.Securities[symbol].Price
                        try:
                            ticket = algo.LimitOrder(symbol, safe_qty * direction_mult, exit_price, tag=tag)
                        except Exception:
                            ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                    track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                    return True
            else:
                if not algo.LiveMode and getattr(algo, 'backtest_use_market_exits', True):
                    ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                else:
                    # Spread unknown – use limit order at last price to capture maker fee
                    exit_price = algo.Securities[symbol].Price
                    try:
                        ticket = algo.LimitOrder(symbol, safe_qty * direction_mult, exit_price, tag=tag)
                    except Exception:
                        ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
                track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
                return True
        else:
            # Stop loss — use market order immediately.
            # A limit-then-fallback approach is too slow: the 5–15% downside during
            # a fast crash far outweighs the ~0.5% taker premium we would save.
            ticket = algo.MarketOrder(symbol, safe_qty * direction_mult, tag=tag)
            track_exit_order(algo, symbol, ticket, safe_qty * direction_mult)
            return True
    else:
        algo.Debug(f"Warning: {symbol.Value} holding {holding_qty} rounds to 0")
        return False


def partial_smart_sell(algo, symbol, fraction, tag="Partial TP"):
    """
    Sell a fraction (0.0–1.0) of the current position.
    Both the sell portion and the remaining portion must meet the minimum
    order size; otherwise falls back to a full smart_liquidate.
    Returns True if an order was placed successfully, False otherwise.
    """
    if symbol not in algo.Portfolio or algo.Portfolio[symbol].Quantity == 0:
        return False
    if len(algo.Transactions.GetOpenOrders(symbol)) > 0:
        return False
    holding_qty = algo.Portfolio[symbol].Quantity
    if holding_qty == 0:
        return False
    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
    if price <= 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    sell_qty = round_quantity(algo, symbol, abs(holding_qty) * fraction)
    remaining_qty = round_quantity(algo, symbol, abs(holding_qty) * (1.0 - fraction))
    # Both halves must be tradeable; fall back to full exit if not
    if sell_qty < min_qty or remaining_qty < min_qty:
        return smart_liquidate(algo, symbol, tag)
    if sell_qty * price < min_notional * 0.9:
        return smart_liquidate(algo, symbol, tag)
    # Flag as partial so OnOrderEvent skips position cleanup
    if hasattr(algo, '_partial_sell_symbols'):
        algo._partial_sell_symbols.add(symbol)
    direction_mult = -1 if holding_qty > 0 else 1
    # Use limit order at current price to capture maker fee (0.25% vs 0.40% taker)
    try:
        ticket = algo.LimitOrder(symbol, sell_qty * direction_mult, price, tag=tag)
    except Exception:
        ticket = algo.MarketOrder(symbol, sell_qty * direction_mult, tag=tag)
    if ticket is not None:
        algo.Debug(f"PARTIAL SELL: {symbol.Value} | frac={fraction:.0%} qty={sell_qty:.6f} of {abs(holding_qty):.6f}")
    return ticket is not None



def effective_stale_timeout(algo):
    return algo.live_stale_order_timeout_seconds if algo.LiveMode else algo.stale_order_timeout_seconds



def is_invested_not_dust(algo, symbol):
    if symbol not in algo.Portfolio:
        return False
    h = algo.Portfolio[symbol]
    if not h.Invested or h.Quantity == 0:
        return False
    min_qty = get_min_quantity(algo, symbol)
    min_notional = get_min_notional_usd(algo, symbol)
    price = algo.Securities[symbol].Price if symbol in algo.Securities else h.Price
    notional_ok = (price > 0) and (abs(h.Quantity) * price >= min_notional * 0.5)
    qty_ok = abs(h.Quantity) >= min_qty * 0.5
    return notional_ok or qty_ok


def get_actual_position_count(algo):
    return sum(1 for kvp in algo.Portfolio if is_invested_not_dust(algo, kvp.Key))


def has_open_orders(algo, symbol=None):
    if symbol is None:
        return len(algo.Transactions.GetOpenOrders()) > 0
    return len(algo.Transactions.GetOpenOrders(symbol)) > 0


def has_non_stale_open_orders(algo, symbol):
    """Check if symbol has open orders that are NOT stale (younger than timeout)."""
    try:
        orders = algo.Transactions.GetOpenOrders(symbol)
        if len(orders) == 0:
            return False
        timeout_seconds = effective_stale_timeout(algo)
        for order in orders:
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age <= timeout_seconds:
                return True  # At least one order is not stale
        return False  # All orders are stale
    except Exception:
        return False


def get_spread_pct(algo, symbol):
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                return (ask - bid) / mid
    except Exception as e:
        algo.Debug(f"Error getting spread for {symbol.Value}: {e}")
        pass
    return None


def _estimate_backtest_spread(algo, symbol):
    """
    Estimate bid-ask spread from dollar volume for backtest realism.
    Uses empirical relationship: spread ≈ k / sqrt(dollar_volume).
    Calibrated against typical Kraken altcoin spreads:
      - $1M daily vol → ~0.20% spread
      - $100K daily vol → ~0.63% spread
      - $10K daily vol → ~2.0% spread
    Only called in backtest when real quote data is unavailable.
    """
    crypto = algo.crypto_data.get(symbol)
    if crypto is None:
        return None

    dv_list = list(crypto.get('dollar_volume', []))
    if len(dv_list) < 5:
        return None

    # Average dollar volume per bar, extrapolated to daily (288 5-min bars/day)
    avg_bar_dv = float(np.mean(dv_list[-20:]))
    daily_dv = avg_bar_dv * 288  # 288 = 5-min bars per day (60min/5min × 24h)

    if daily_dv <= 0:
        return 0.05  # 5% default for zero-volume — will be rejected by spread cap

    # Empirical constant: k=2.0 → 0.20% at $1M, 0.63% at $100K, 2.0% at $10K
    k = 2.0
    estimated_spread = k / (daily_dv ** 0.5)

    # Floor: even the most liquid alts have >= 0.05% spread on Kraken
    # Cap: don't estimate above 10% — if it's that wide, the filter will catch it
    estimated_spread = max(estimated_spread, 0.0007)
    estimated_spread = min(estimated_spread, 0.10)

    # Time-of-day: 00-08 UTC Asian (×1.5), 08-13 EU (×1.0), 13-21 US (×0.85), 21+ (×1.2).
    h = algo.Time.hour
    if h < _TOD_ASIAN_END:
        tod_multiplier = 1.5
    elif h < _TOD_EU_END:
        tod_multiplier = 1.0
    elif h < _TOD_US_END:
        tod_multiplier = 0.85
    else:
        tod_multiplier = 1.2
    estimated_spread *= tod_multiplier

    # Stress mode: optional pessimistic spread multiplier for robustness testing.
    # Set algo.stress_spread_mult > 1.0 (e.g. 1.5 or 2.0) to simulate harsher spreads.
    stress_mult = getattr(algo, 'stress_spread_mult', 1.0)
    if stress_mult != 1.0:
        estimated_spread *= stress_mult

    estimated_spread = min(estimated_spread, 0.10)

    # NOTE: do NOT append to crypto['spreads'] here — this function may be called
    # more than once per bar (e.g. once in screening and once in execution).
    # The single per-bar append is performed in update_symbol_data (strategy_core.py).

    return estimated_spread


def spread_ok(algo, symbol):
    sp = get_spread_pct(algo, symbol)
    if sp is None:
        if algo.LiveMode:
            # Allow unknown spreads in live — spread cap check will catch them once data arrives.
            now = algo.Time
            last_warn = algo._spread_warning_times.get(symbol.Value)
            if last_warn is None or (now - last_warn).total_seconds() >= 3600:
                debug_limited(algo, f"SPREAD UNKNOWN: {symbol.Value} — allowing (no bid/ask yet)")
                algo._spread_warning_times[symbol.Value] = now
            return True
        else:
            # BACKTEST REALISM: estimate spread from dollar volume
            sp = _estimate_backtest_spread(algo, symbol)
            if sp is None:
                return True  # no data at all, allow (conservative fallback)
            # Fall through to normal spread checks below with estimated spread
    effective_spread_cap = algo.max_spread_pct
    if algo.volatility_regime == "high" or algo.market_regime == "sideways":
        effective_spread_cap = min(effective_spread_cap, 0.003)
    if sp > effective_spread_cap:
        return False
    crypto = algo.crypto_data.get(symbol)
    if crypto and len(crypto.get('spreads', [])) >= 4:
        median_sp = np.median(list(crypto['spreads']))
        if median_sp > 0 and sp > algo.spread_widen_mult * median_sp:
            return False
    return True


def intraday_volume_ok(algo, symbol, order_value):
    """Returns True if recent intraday dollar volume is sufficient for the order."""
    crypto = algo.crypto_data.get(symbol)
    if crypto is None:
        return True  # no data, allow (conservative)

    dv_list = list(crypto.get('dollar_volume', []))
    if len(dv_list) < 5:
        return True  # not enough history yet

    # Min $500 avg dollar volume per bar; rejects low-activity windows
    recent_avg_dv = float(np.mean(dv_list[-10:])) if len(dv_list) >= 10 else float(np.mean(dv_list))
    if recent_avg_dv < 500:
        return False

    if recent_avg_dv > 0 and order_value > 0:
        implied_participation = order_value / recent_avg_dv
        if implied_participation > algo.max_participation_rate:
            return False

    return True


def cleanup_position(algo, symbol, record_pnl=False, exit_price=None):
    """
    Clean up position tracking for a symbol.
    If record_pnl=True, records the PnL before cleanup using record_exit_pnl helper.
    """
    entry_price = algo.entry_prices.get(symbol, None)
    if record_pnl and entry_price is not None and entry_price > 0:
        if exit_price is None:
            try:
                exit_price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
            except Exception:
                exit_price = 0
        if exit_price > 0:
            record_exit_pnl(algo, symbol, entry_price, exit_price)
    # Then do existing cleanup
    algo.entry_prices.pop(symbol, None)
    algo.highest_prices.pop(symbol, None)
    algo.entry_times.pop(symbol, None)
    if symbol in algo.crypto_data:
        algo.crypto_data[symbol]['trail_stop'] = None
    if hasattr(algo, '_spike_entries'):
        algo._spike_entries.pop(symbol, None)
    if hasattr(algo, '_partial_tp_taken'):
        algo._partial_tp_taken.pop(symbol, None)
    if hasattr(algo, '_partial_tp_tier'):
        algo._partial_tp_tier.pop(symbol, None)
    if hasattr(algo, '_entry_signal_combos'):
        algo._entry_signal_combos.pop(symbol, None)


def sync_existing_positions(algo):
    algo.Debug("=" * 50)
    algo.Debug("=== SYNCING EXISTING POSITIONS ===")
    synced_count = 0
    positions_to_close = []
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        holding = algo.Portfolio[symbol]
        ticker = symbol.Value
        if symbol in algo.entry_prices:
            continue
        if symbol not in algo.Securities:
            try:
                algo.AddCrypto(ticker, Resolution.Minute, Market.Kraken)
            except Exception as e:
                algo.Debug(f"Error adding crypto {ticker}: {e}")
                continue
        algo.entry_prices[symbol] = holding.AveragePrice
        algo.highest_prices[symbol] = holding.AveragePrice
        algo.entry_times[symbol] = algo.Time
        synced_count += 1
        current_price = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
        pnl_pct = (current_price - holding.AveragePrice) / holding.AveragePrice if holding.AveragePrice > 0 else 0
        algo.Debug(f"SYNCED: {ticker} | Entry: ${holding.AveragePrice:.4f} | Now: ${current_price:.4f} | PnL: {pnl_pct:+.2%}")
        if current_price > holding.AveragePrice:
            algo.highest_prices[symbol] = current_price
        if pnl_pct >= algo.quick_take_profit:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync TP"))
        elif pnl_pct <= -algo.tight_stop_loss:
            positions_to_close.append((symbol, ticker, pnl_pct, "Sync SL"))
    algo.Debug(f"Synced {synced_count} positions")
    algo.Debug(f"Cash: ${algo.Portfolio.Cash:.2f}")
    algo.Debug("=" * 50)
    for symbol, ticker, pnl_pct, reason in positions_to_close:
        algo.Debug(f"IMMEDIATE {reason}: {ticker} at {pnl_pct:+.2%}")
        sold = smart_liquidate(algo, symbol, reason)
        if not sold:
            algo.Debug(f"⚠️ IMMEDIATE {reason} FAILED: {ticker} — position unsellable, cleaning up tracking")
            cleanup_position(algo, symbol)
        # Let OnOrderEvent handle cleanup and PnL tracking on fill


def debug_limited(algo, msg):
    if "CANCELED" in msg or "ZOMBIE" in msg or "INVALID" in msg:
        algo.Debug(msg)
        return
    if algo.log_budget > 0:
        algo.Debug(msg)
        algo.log_budget -= 1
    elif algo.LiveMode:
        algo.Debug(msg)


def slip_log(algo, symbol, direction, fill_price):
    """Enhanced slip_log with live outlier alert and symbol-level slippage tracking."""
    try:
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice
        if bid <= 0 or ask <= 0:
            return
        mid = 0.5 * (bid + ask)
        if mid <= 0:
            return
        side = 1 if direction == OrderDirection.Buy else -1
        slip = side * (fill_price - mid) / mid
        abs_slip = abs(slip)
        algo._slip_abs.append(abs_slip)
        
        # Track slippage per symbol (store (time, abs_slippage) tuples for time-based decay)
        if hasattr(algo, '_symbol_slippage_history'):
            ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
            if ticker not in algo._symbol_slippage_history:
                algo._symbol_slippage_history[ticker] = deque(maxlen=30)
            algo._symbol_slippage_history[ticker].append((algo.Time, abs_slip))
        
        # Live slippage alert for unusually high slippage
        if algo.LiveMode and abs(slip) > algo.slip_outlier_threshold:
            algo.Debug(f"⚠️ HIGH SLIPPAGE: {symbol.Value} | {abs(slip):.4%} | dir={direction}")
    except Exception as e:
        algo.Debug(f"Error in slip_log for {symbol.Value}: {e}")
        pass


def persist_state(algo):
    """Enhanced persist_state with trade_count and peak_value from main_opus."""
    if not algo.LiveMode:
        return
    try:
        spike_entries = [s.Value for s in getattr(algo, '_spike_entries', {}).keys() if hasattr(s, 'Value')]
        state = {
            "session_blacklist": list(algo._session_blacklist),
            "winning_trades": algo.winning_trades,
            "losing_trades": algo.losing_trades,
            "total_pnl": algo.total_pnl,
            "consecutive_losses": algo.consecutive_losses,
            "daily_trade_count": algo.daily_trade_count,
            "trade_count": algo.trade_count,
            "peak_value": algo.peak_value if algo.peak_value is not None else 0,
            "spike_entries": spike_entries,
        }
        algo.ObjectStore.Save("live_state", json.dumps(state))
    except Exception as e:
        algo.Debug(f"Persist error: {e}")


def load_persisted_state(algo):
    """Enhanced load_persisted_state with trade_count and peak_value from main_opus."""
    try:
        if algo.LiveMode and algo.ObjectStore.ContainsKey("live_state"):
            raw = algo.ObjectStore.Read("live_state")
            data = json.loads(raw)
            algo._session_blacklist = set(data.get("session_blacklist", []))
            algo.winning_trades = data.get("winning_trades", 0)
            algo.losing_trades = data.get("losing_trades", 0)
            algo.total_pnl = data.get("total_pnl", 0.0)
            algo.consecutive_losses = data.get("consecutive_losses", 0)
            algo.daily_trade_count = data.get("daily_trade_count", 0)
            algo.trade_count = data.get("trade_count", 0)
            peak = data.get("peak_value", 0)
            if peak > 0:
                algo.peak_value = peak
            # Restore spike entries: map string values back to Symbol objects
            spike_entry_values = set(data.get("spike_entries", []))
            if spike_entry_values and hasattr(algo, '_spike_entries'):
                for symbol in algo.Securities.Keys:
                    if hasattr(symbol, 'Value') and symbol.Value in spike_entry_values:
                        algo._spike_entries[symbol] = True
            algo.Debug(f"Loaded persisted state: blacklist {len(algo._session_blacklist)}, "
                       f"trades W:{algo.winning_trades}/L:{algo.losing_trades}")
    except Exception as e:
        algo.Debug(f"Load persist error: {e}")


def cleanup_object_store(algo):
    """From main_opus._cleanup_object_store."""
    try:
        n = 0
        for i in algo.ObjectStore.GetEnumerator():
            k = i.Key if hasattr(i, 'Key') else str(i)
            if k != "live_state":
                try:
                    algo.ObjectStore.Delete(k)
                    n += 1
                except Exception as e:
                    algo.Debug(f"Error deleting key {k}: {e}")
                    pass
        if n:
            algo.Debug(f"Cleaned {n} keys")
    except Exception as e:
        algo.Debug(f"Cleanup err: {e}")


def live_safety_checks(algo):
    """Extra safety checks for live trading from main_opus."""
    if not algo.LiveMode:
        return True
    
    # Check if we have minimum viable cash
    try:
        cash = algo.Portfolio.CashBook["USD"].Amount
    except Exception as e:
        algo.Debug(f"Error getting cash from CashBook, using Portfolio.Cash: {e}")
        cash = algo.Portfolio.Cash
    
    if cash < 2.0:
        debug_limited(algo, "LIVE SAFETY: Cash below $2, pausing new entries")
        return False
    
    # Rate limit: don't trade more than once per 90 seconds in live
    if hasattr(algo, '_last_live_trade_time') and algo._last_live_trade_time is not None:
        seconds_since = (algo.Time - algo._last_live_trade_time).total_seconds()
        if seconds_since < 90:
            return False
    
    return True


def kelly_fraction(algo):
    """
    Half-Kelly position sizing multiplier, normalized so 1.0 = neutral.

    Normalization: divide by 0.25, the expected half-Kelly for a typical strategy
    with full-Kelly ≈ 0.50. This means:
      half_kelly=0.25 → returns 1.0  (neutral sizing)
      half_kelly=0.375 → returns 1.5 (max boost, capped)
      half_kelly=0.125 → returns 0.5 (min floor)
    Result is clamped to [0.5, 1.5] to prevent extreme under/over-sizing.
    """
    if len(algo._rolling_wins) < 10:
        return 1.0
    win_rate = sum(algo._rolling_wins) / len(algo._rolling_wins)
    if win_rate <= 0 or win_rate >= 1:
        return 1.0
    avg_win = np.mean(list(algo._rolling_win_sizes)) if len(algo._rolling_win_sizes) > 0 else 0.02
    avg_loss = np.mean(list(algo._rolling_loss_sizes)) if len(algo._rolling_loss_sizes) > 0 else 0.02
    if avg_loss <= 0:
        return 1.0
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    half_kelly = kelly * 0.5
    return max(0.5, min(1.5, half_kelly / 0.25))


def get_slippage_penalty(algo, symbol):
    """
    Calculate position size multiplier based on historical slippage for a symbol.
    Returns a value between 0.3 and 1.0. Ignores slippage entries older than 48 hours.
    """
    if not hasattr(algo, '_symbol_slippage_history'):
        return 1.0
    
    ticker = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
    if ticker not in algo._symbol_slippage_history:
        return 1.0
    
    slippage_history = algo._symbol_slippage_history[ticker]
    if len(slippage_history) == 0:
        return 1.0
    
    # Filter out entries older than 48 hours
    cutoff = algo.Time - timedelta(hours=48)
    recent_slips = [slip for entry_time, slip in slippage_history if entry_time >= cutoff]
    if len(recent_slips) == 0:
        return 1.0
    
    avg_slippage = sum(recent_slips) / len(recent_slips)
    
    # Apply penalties based on average slippage
    if avg_slippage > 0.010:  # > 1.0%
        return 0.3
    elif avg_slippage > 0.005:  # > 0.5%
        return 0.6
    elif avg_slippage > 0.003:  # > 0.3%
        return 0.8
    else:
        return 1.0


def place_limit_or_market(algo, symbol, quantity, timeout_seconds=30, tag="Entry", is_breakout=False):
    """
    Place entry orders using limit orders to capture maker fees.
    In backtest: volatility-correlated non-fill simulation, then falls through
    to the same limit order logic as live.
    In live: use limit orders with timeout to capture maker fees.

    Parameters
    ----------
    is_breakout : bool
        When True (vol-ignition-heavy / momentum entry with no mean-reversion
        support) an extra non-fill penalty is applied in backtest.  Momentum
        entries are unlikely to fill passively as a maker — using a higher
        rejection probability produces more realistic backtest results.
        Configurable via algo.breakout_nonfill_penalty (default 0.08 / +8 pp).

    Returns the ticket from the order placement.
    """
    try:
        # BACKTEST REALISM: volatility-correlated non-fill simulation
        # Real non-fills are adversely selected — higher volatility = more non-fills
        # because price moves away faster. NOT random.
        if not algo.LiveMode:
            crypto = algo.crypto_data.get(symbol)
            if crypto and len(crypto.get('volatility', [])) > 0:
                recent_vol = float(crypto['volatility'][-1])
                # Base non-fill rate 8%, scales up to 30% in high-vol environments.
                # Real limit order fill rates on Kraken altcoins at bid+0.05% are 70-92%
                # within a 5-minute bar; the 8% base reflects a realistic rejection rate.
                non_fill_prob = min(0.08 + recent_vol * 3.0, 0.30)
            else:
                non_fill_prob = 0.15
            # Signal-aware adjustment: breakout / momentum entries are rarely filled as
            # maker — price is running away from the limit so the extra penalty models
            # adverse selection more honestly.  Mean-reversion entries provide liquidity
            # and are left at the base rate.
            if is_breakout:
                penalty = getattr(algo, 'breakout_nonfill_penalty', _BREAKOUT_NONFILL_PENALTY_DEFAULT)
                non_fill_prob = min(non_fill_prob + penalty, 0.60)
            if random.random() < non_fill_prob:
                if getattr(algo, 'nonfill_market_fallback_enabled', True):
                    algo.Debug(f"BACKTEST NON-FILL FALLBACK: {symbol.Value} p={non_fill_prob:.1%} -> market")
                    return algo.MarketOrder(symbol, quantity, tag=f"{tag}_NonFillFallback")
                return None
        # Fall through to limit order logic for both live and backtest
        sec = algo.Securities[symbol]
        bid = sec.BidPrice
        ask = sec.AskPrice

        if bid > 0 and ask > 0:
            # Place limit order just above best bid – still maker, improves fill odds
            limit_price = bid * 1.0005  # 0.05% above bid – still below mid, still maker

            # BACKTEST: simulate queue-priority cost when bid/ask data IS available.
            # Offset (0.12%) reflects real queue priority cost on Kraken altcoins (0.10-0.20%).
            if not algo.LiveMode:
                adverse_offset = getattr(algo, 'backtest_entry_adverse_offset', _BACKTEST_ENTRY_ADVERSE_OFFSET_DEFAULT)
                if quantity > 0:
                    limit_price *= (1 + adverse_offset)
                    limit_price = min(limit_price, ask)  # buy: never cross the ask
                else:
                    limit_price *= (1 - adverse_offset)
                    limit_price = max(limit_price, bid)  # sell: never go below the bid

                # Probabilistic rejection based on participation rate (>5% threshold).
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto.get('volume', [])) > 0:
                    bar_volume = float(crypto['volume'][-1])
                    bar_dollar_volume = bar_volume * float(limit_price) if limit_price > 0 else 0
                    order_notional = abs(quantity) * float(limit_price)
                    if bar_dollar_volume > 0:
                        participation_rate = order_notional / bar_dollar_volume
                        if participation_rate > _QUEUE_PARTICIPATION_THRESHOLD:
                            excess = participation_rate - _QUEUE_PARTICIPATION_THRESHOLD
                            rejection_prob = min(excess * _QUEUE_REJECTION_SLOPE, _QUEUE_MAX_REJECTION_PROB)
                            if random.random() < rejection_prob:
                                algo.Debug(
                                    f"BACKTEST QUEUE REJECT (bid/ask): {symbol.Value} "
                                    f"part={participation_rate:.1%} rej={rejection_prob:.1%}"
                                )
                                return None
            else:
                limit_price = min(limit_price, ask)  # live: never cross the spread
        else:
            # No bid/ask data: use last price as limit price.
            # In backtest this would fill immediately at the limit price,
            # giving unrealistic instant maker fills without queue priority.
            limit_price = sec.Price
            if limit_price <= 0:
                algo.Debug(f"Price unavailable for {symbol.Value}, using market order")
                return algo.MarketOrder(symbol, quantity, tag=tag)

            # BACKTEST: simulate queue-priority cost when no bid/ask data.
            # Apply adverse offset + participation-rate rejection.
            if not algo.LiveMode:
                # Adverse offset: algo is not first in line at the limit level.
                adverse_offset = getattr(algo, 'backtest_entry_noquote_offset', _BACKTEST_ENTRY_NOQUOTE_OFFSET_DEFAULT)
                if quantity > 0:
                    limit_price *= (1 + adverse_offset)
                else:
                    limit_price *= (1 - adverse_offset)

                # Probabilistic rejection based on participation rate (>5% threshold).
                crypto = algo.crypto_data.get(symbol)
                if crypto and len(crypto.get('volume', [])) > 0:
                    bar_volume = float(crypto['volume'][-1])
                    bar_dollar_volume = bar_volume * float(limit_price) if limit_price > 0 else 0
                    order_notional = abs(quantity) * float(limit_price)
                    if bar_dollar_volume > 0:
                        participation_rate = order_notional / bar_dollar_volume
                        if participation_rate > _QUEUE_PARTICIPATION_THRESHOLD:
                            excess = participation_rate - _QUEUE_PARTICIPATION_THRESHOLD
                            rejection_prob = min(excess * _QUEUE_REJECTION_SLOPE, _QUEUE_MAX_REJECTION_PROB)
                            if random.random() < rejection_prob:
                                algo.Debug(
                                    f"BACKTEST QUEUE REJECT: {symbol.Value} "
                                    f"part={participation_rate:.1%} rej={rejection_prob:.1%}"
                                )
                                return None

        # Place maker limit order
        limit_ticket = algo.LimitOrder(symbol, quantity, limit_price, tag=tag)

        # Track for fallback in VerifyOrderFills
        if hasattr(algo, '_submitted_orders'):
            algo._submitted_orders[symbol] = {
                'order_id': limit_ticket.OrderId,
                'time': algo.Time,
                'quantity': quantity,  # Store signed quantity
                'is_limit_entry': True,
                'timeout_seconds': timeout_seconds,
                'intent': 'entry'
            }

            algo.Debug(f"MAKER LIMIT: {symbol.Value} | qty={quantity} | bid=${bid:.4f} | limit=${limit_price:.4f} | timeout={timeout_seconds}s")
        return limit_ticket

    except Exception as e:
        algo.Debug(f"Error placing limit order for {symbol.Value}: {e}, falling back to market")
        return algo.MarketOrder(symbol, quantity, tag=tag)


def normalize_order_time(order_time):
    """Helper to normalize order time by removing timezone info if present."""
    return order_time.replace(tzinfo=None) if order_time.tzinfo is not None else order_time


def get_hold_bucket(hold_hours):
    """Return a string bucket label based on hold duration in hours."""
    if hold_hours < 0.5:
        return '<30min'
    elif hold_hours < 2.0:
        return '30min-2h'
    elif hold_hours < 6.0:
        return '2h-6h'
    else:
        return '6h+'


def record_exit_pnl(algo, symbol, entry_price, exit_price, exit_tag="Unknown"):
    """Helper to record PnL from an exit trade. Returns None if prices are invalid."""
    if entry_price <= 0 or exit_price <= 0:
        algo.Debug(f"⚠️ Cannot record PnL for {symbol.Value}: invalid prices (entry=${entry_price:.4f}, exit=${exit_price:.4f})")
        return None
    
    # Fee is a direct percentage-point reduction (matches convention in events.py).
    pnl = (exit_price - entry_price) / entry_price - ESTIMATED_ROUND_TRIP_FEE
    algo._rolling_wins.append(1 if pnl > 0 else 0)
    if pnl > 0:
        algo._rolling_win_sizes.append(pnl)
        algo.winning_trades += 1
        algo.consecutive_losses = 0
    else:
        algo._rolling_loss_sizes.append(abs(pnl))
        algo.losing_trades += 1
        algo.consecutive_losses += 1
    algo.total_pnl += pnl
    if not hasattr(algo, 'pnl_by_tag'):
        algo.pnl_by_tag = {}
    if exit_tag not in algo.pnl_by_tag:
        algo.pnl_by_tag[exit_tag] = []
    algo.pnl_by_tag[exit_tag].append(pnl)

    # Regime-level PnL tracking (capped at 200 entries per bucket to prevent
    # unbounded memory growth during long live runs, consistent with pnl_by_tag).
    if not hasattr(algo, 'pnl_by_regime'):
        algo.pnl_by_regime = {}
    regime = getattr(algo, 'market_regime', 'unknown')
    if regime not in algo.pnl_by_regime:
        algo.pnl_by_regime[regime] = []
    algo.pnl_by_regime[regime].append(pnl)
    if len(algo.pnl_by_regime[regime]) > 200:
        algo.pnl_by_regime[regime] = algo.pnl_by_regime[regime][-200:]

    # Volatility regime PnL tracking (same 200-entry cap)
    if not hasattr(algo, 'pnl_by_vol_regime'):
        algo.pnl_by_vol_regime = {}
    vol_regime = getattr(algo, 'volatility_regime', 'normal')
    if vol_regime not in algo.pnl_by_vol_regime:
        algo.pnl_by_vol_regime[vol_regime] = []
    algo.pnl_by_vol_regime[vol_regime].append(pnl)
    if len(algo.pnl_by_vol_regime[vol_regime]) > 200:
        algo.pnl_by_vol_regime[vol_regime] = algo.pnl_by_vol_regime[vol_regime][-200:]

    # Signal-combination attribution
    if hasattr(algo, '_entry_signal_combos') and symbol in algo._entry_signal_combos:
        signal_combo = algo._entry_signal_combos.pop(symbol)
        if not hasattr(algo, 'pnl_by_signal_combo'):
            algo.pnl_by_signal_combo = {}
        if signal_combo not in algo.pnl_by_signal_combo:
            algo.pnl_by_signal_combo[signal_combo] = []
        algo.pnl_by_signal_combo[signal_combo].append(pnl)

    # Hold-time attribution
    entry_time = algo.entry_times.get(symbol) if hasattr(algo, 'entry_times') else None
    if entry_time is not None:
        hold_hours = (algo.Time - entry_time).total_seconds() / 3600
        hold_bucket = get_hold_bucket(hold_hours)
    else:
        hold_bucket = 'unknown'
    if not hasattr(algo, 'pnl_by_hold_time'):
        algo.pnl_by_hold_time = {}
    if hold_bucket not in algo.pnl_by_hold_time:
        algo.pnl_by_hold_time[hold_bucket] = []
    algo.pnl_by_hold_time[hold_bucket].append(pnl)

    return pnl


def get_open_buy_orders_value(algo):
    """Calculate total value reserved by open buy orders."""
    total_reserved = 0
    for o in algo.Transactions.GetOpenOrders():
        if o.Direction == OrderDirection.Buy:
            if o.Price > 0:
                order_price = o.Price
            elif o.Symbol in algo.Securities:
                order_price = algo.Securities[o.Symbol].Price
                if order_price <= 0:
                    continue
            else:
                continue
            total_reserved += abs(o.Quantity) * order_price
    return total_reserved




def execute_buy(algo, symbol, quantity):
    """Place a market buy order for immediate fill."""
    algo.MarketOrder(symbol, quantity, tag="MG Market Buy")


def execute_sell(algo, symbol, quantity):
    """Place a market sell order for immediate fill."""
    algo.MarketOrder(symbol, -quantity, tag="MG Market Sell")
