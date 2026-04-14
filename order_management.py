# region imports
from AlgorithmImports import *
from execution import (
    is_invested_not_dust, cleanup_position, record_exit_pnl,
    normalize_order_time, effective_stale_timeout, has_non_stale_open_orders,
    persist_state, debug_limited,
)
from datetime import timedelta
# endregion


def cancel_stale_new_orders(algo):
    # Allow cancel gate when venue is online or unknown (fallback handled elsewhere)
    try:
        open_orders = algo.Transactions.GetOpenOrders()
        timeout_seconds = effective_stale_timeout(algo)
        for order in open_orders:
            sym_val = order.Symbol.Value
            if sym_val in algo._session_blacklist:
                continue
            order_time = order.Time
            if order_time.tzinfo is not None:
                order_time = order_time.replace(tzinfo=None)
            order_age = (algo.Time - order_time).total_seconds()
            if order_age > timeout_seconds:
                algo.Debug(f"Canceling stale: {sym_val} (age: {order_age/60:.1f}m, timeout {timeout_seconds/60:.1f}m)")
                
                # Check if this order actually filled (fill event missed)
                if is_invested_not_dust(algo, order.Symbol):
                    # Position exists — order filled, we just missed the event
                    # Re-track instead of blacklisting
                    algo.Debug(f"STALE ORDER but position exists: {sym_val} — re-tracking")
                    holding = algo.Portfolio[order.Symbol]
                    algo.entry_prices[order.Symbol] = holding.AveragePrice
                    algo.highest_prices[order.Symbol] = holding.AveragePrice
                    algo.entry_times[order.Symbol] = algo.Time
                    algo.Transactions.CancelOrder(order.Id)  # Cancel the stale order
                    continue  # Don't blacklist
                
                algo.Transactions.CancelOrder(order.Id)
                algo._cancel_cooldowns[order.Symbol] = algo.Time + timedelta(minutes=algo.cancel_cooldown_minutes)
                
                # Only blacklist stale ENTRY orders, not EXIT orders
                # Exit orders that are stale just get cooldown to allow retry
                has_position_or_tracked = order.Symbol in algo.entry_prices or (
                    order.Symbol in algo.Portfolio and algo.Portfolio[order.Symbol].Quantity != 0
                )
                
                if has_position_or_tracked:
                    algo.Debug(f"STALE EXIT: {sym_val} - cooldown only, not blacklisted")
                else:
                    algo._symbol_entry_cooldowns[sym_val] = algo.Time + timedelta(minutes=15)
                    algo.Debug(f"⚠️ ZOMBIE ORDER DETECTED: {sym_val} - entry cooldown 15min")
    except Exception as e:
        algo.Debug(f"Error in _cancel_stale_new_orders: {e}")


def health_check(algo):
    """
    Enhanced health check with improved orphan detection and reverse resync.
    Implements Fix 3: check if ALL orders are stale before skipping orphan check.
    Includes reverse resync (phantom position detection) for live mode.
    """
    if algo.IsWarmingUp:
        return
    
    # Live resync to catch any holdings the event stream might have missed
    resync_holdings_full(algo)
    
    issues = []
    if algo.Portfolio.Cash < 5:
        issues.append(f"Low cash: ${algo.Portfolio.Cash:.2f}")
    
    for symbol in list(algo.entry_prices.keys()):
        open_orders = algo.Transactions.GetOpenOrders(symbol)
        if len(open_orders) > 0:
            # Check if all orders are stuck (older than timeout)
            all_stale = True
            for o in open_orders:
                order_time = normalize_order_time(o.Time)
                if (algo.Time - order_time).total_seconds() <= algo.live_stale_order_timeout_seconds:
                    all_stale = False
                    break
            if not all_stale:
                continue  # Has non-stale orders, skip this symbol
            
            # All orders are stale - cancel them
            for o in open_orders:
                try:
                    algo.Transactions.CancelOrder(o.Id)
                    issues.append(f"Canceled stale order: {symbol.Value} (order {o.Id})")
                except Exception as e:
                    algo.Debug(f"Error canceling stale order for {symbol.Value}: {e}")
        
        if not is_invested_not_dust(algo, symbol):
            issues.append(f"Orphan tracking: {symbol.Value}")
            cleanup_position(algo, symbol)
    
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key) and kvp.Key not in algo.entry_prices:
            issues.append(f"Untracked position: {kvp.Key.Value}")
    
    if len(algo._session_blacklist) > 50:
        issues.append(f"Large session blacklist: {len(algo._session_blacklist)}")
    
    open_orders = algo.Transactions.GetOpenOrders()
    if len(open_orders) > 0:
        issues.append(f"Open orders: {len(open_orders)}")
    
    if issues:
        algo.Debug("=== HEALTH CHECK ===")
        for issue in issues:
            algo.Debug(f"  ⚠️ {issue}")
    else:
        debug_limited(algo, "Health check: OK")


def resync_holdings_full(algo):
    """Live-only: backfills tracking for holdings missed via OnOrderEvent; detects phantom positions."""
    if algo.IsWarmingUp or not algo.LiveMode:
        return
    
    if not hasattr(algo, '_last_resync_log') or (algo.Time - algo._last_resync_log).total_seconds() > algo.resync_log_interval_seconds:
        algo.Debug(f"RESYNC CHECK: keys={len(list(algo.Portfolio.Keys))} tracked={len(algo.entry_prices)}")
        algo._last_resync_log = algo.Time
    
    # Forward resync: find holdings we're not tracking
    missing = []
    for symbol in algo.Portfolio.Keys:
        if not is_invested_not_dust(algo, symbol):
            continue
        if symbol in algo.entry_prices:
            continue
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        if has_non_stale_open_orders(algo, symbol):
            continue
        missing.append(symbol)
    
    if missing:
        algo.Debug(f"RESYNC: {len(missing)} untracked holdings; backfilling.")
        for symbol in missing:
            try:
                if symbol not in algo.Securities:
                    algo.AddCrypto(symbol.Value, Resolution.Minute5, Market.Kraken)
                holding = algo.Portfolio[symbol]
                entry = holding.AveragePrice
                algo.entry_prices[symbol] = entry
                algo.highest_prices[symbol] = entry
                algo.entry_times[symbol] = algo.Time
                cur = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                pnl_pct = (cur - entry) / entry if entry > 0 else 0
                algo.Debug(f"RESYNCED: {symbol.Value} entry=${entry:.4f} now=${cur:.4f} {pnl_pct:+.2%}")
            except Exception as e:
                algo.Debug(f"Resync error {symbol.Value}: {e}")
    
    # Reverse resync: detect phantom positions (tracked but broker has none)
    phantoms = []
    for symbol in list(algo.entry_prices.keys()):
        if symbol in algo._submitted_orders:
            continue
        if symbol in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[symbol]:
            continue
        holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
        if holding is None or not holding.Invested or holding.Quantity == 0:
            open_orders = algo.Transactions.GetOpenOrders(symbol)
            if len(open_orders) > 0:
                all_stuck = all(
                    (algo.Time - normalize_order_time(o.Time)).total_seconds() > algo.live_stale_order_timeout_seconds
                    for o in open_orders
                )
                if not all_stuck:
                    continue
                for o in open_orders:
                    try:
                        algo.Transactions.CancelOrder(o.Id)
                    except Exception as e:
                        algo.Debug(f"Error canceling stuck order {symbol.Value}: {e}")
            phantoms.append(symbol)
    
    if phantoms:
        algo.Debug(f"⚠️ REVERSE RESYNC: {len(phantoms)} phantom positions")
        for symbol in phantoms:
            algo.Debug(f"⚠️ PHANTOM: {symbol.Value} tracked but qty=0, cleaning up")
            cleanup_position(algo, symbol, record_pnl=True)
            if hasattr(algo, 'exit_cooldown_hours') and hasattr(algo, '_exit_cooldowns'):
                algo._exit_cooldowns[symbol] = algo.Time + timedelta(hours=algo.exit_cooldown_hours)
        persist_state(algo)


def verify_order_fills(algo):
    """Verify submitted orders filled/timed-out; retry once before blacklisting."""
    if algo.IsWarmingUp:
        return
    
    current_time = algo.Time
    symbols_to_remove = []
    
    for symbol in list(algo._retry_pending.keys()):
        cancel_time = algo._retry_pending[symbol]
        if (current_time - cancel_time).total_seconds() >= algo.retry_pending_cooldown_seconds:
            if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                holding = algo.Portfolio[symbol]
                if symbol not in algo.entry_prices:
                    algo.entry_prices[symbol] = holding.AveragePrice
                    algo.highest_prices[symbol] = holding.AveragePrice
                    algo.entry_times[symbol] = current_time
                    algo.Debug(f"RETRY SKIPPED (position exists): {symbol.Value}")
                del algo._retry_pending[symbol]
            else:
                algo.Debug(f"RETRY NOW: {symbol.Value}")
                del algo._retry_pending[symbol]
    
    for symbol, order_info in list(algo._submitted_orders.items()):
        order_age_seconds = (current_time - order_info['time']).total_seconds()
        order_id = order_info['order_id']
        
        if order_info.get('is_limit_entry', False):
            timeout = order_info.get('timeout_seconds', 60)
        elif order_info.get('is_limit_exit', False):
            timeout = 90
        else:
            timeout = algo.order_timeout_seconds
        
        if order_age_seconds > algo.order_fill_check_threshold_seconds:
            intent = order_info.get('intent', 'entry')
            try:
                order = algo.Transactions.GetOrderById(order_id)
                if order is not None and order.Status == OrderStatus.Filled:
                    if intent == 'exit':
                        entry = algo.entry_prices.get(symbol, None)
                        if entry:
                            cur = algo.Securities[symbol].Price if symbol in algo.Securities else None
                            if cur is not None and cur > 0:
                                pnl = record_exit_pnl(algo, symbol, entry, cur)
                                algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} pnl={pnl:+.2%}" if pnl is not None else f"⚠️ MISSED EXIT FILL: {symbol.Value} invalid price")
                            else:
                                algo.Debug(f"⚠️ MISSED EXIT FILL: {symbol.Value} no price")
                            cleanup_position(algo, symbol)
                        symbols_to_remove.append(symbol)
                        algo._order_retries.pop(order_id, None)
                        continue
                    else:
                        if symbol in algo.Portfolio and algo.Portfolio[symbol].Invested:
                            holding = algo.Portfolio[symbol]
                            entry_price = holding.AveragePrice
                            cur = algo.Securities[symbol].Price if symbol in algo.Securities else holding.Price
                            if symbol not in algo.entry_prices:
                                algo.entry_prices[symbol] = entry_price
                                algo.highest_prices[symbol] = max(cur, entry_price)
                                algo.entry_times[symbol] = order_info['time']
                                algo.daily_trade_count += 1
                                algo.Debug(f"FILL VERIFIED: {symbol.Value} entry=${entry_price:.4f}")
                            symbols_to_remove.append(symbol)
                            algo._order_retries.pop(order_id, None)
                            continue
            except Exception as e:
                algo.Debug(f"Error checking order {symbol.Value}: {e}")
            
            if intent == 'exit':
                holding = algo.Portfolio[symbol] if symbol in algo.Portfolio else None
                if holding is None or not holding.Invested or holding.Quantity == 0:
                    entry = algo.entry_prices.get(symbol, None)
                    if entry:
                        cur = algo.Securities[symbol].Price if symbol in algo.Securities else None
                        if cur is not None and cur > 0:
                            pnl = record_exit_pnl(algo, symbol, entry, cur)
                            algo.Debug(f"⚠️ PHANTOM EXIT: {symbol.Value} pnl={pnl:+.2%}" if pnl is not None else f"⚠️ PHANTOM EXIT: {symbol.Value} invalid price")
                        else:
                            algo.Debug(f"⚠️ PHANTOM EXIT: {symbol.Value} no price")
                        cleanup_position(algo, symbol)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    continue
        
        if order_age_seconds > timeout:
            retry_count = algo._order_retries.get(order_id, 0)
            if retry_count == 0:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo.Debug(f"ORDER TIMEOUT attempt 1: {symbol.Value}")
                    algo._retry_pending[symbol] = current_time
                    algo._order_retries[order_id] = 1
                    symbols_to_remove.append(symbol)
                except Exception as e:
                    algo.Debug(f"Error canceling {symbol.Value}: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
            else:
                try:
                    algo.Transactions.CancelOrder(order_id)
                    algo._session_blacklist.add(symbol.Value)
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
                    algo.Debug(f"ORDER TIMEOUT attempt 2: {symbol.Value} blacklisted")
                except Exception as e:
                    algo.Debug(f"Error canceling {order_id}: {e}")
                    symbols_to_remove.append(symbol)
                    algo._order_retries.pop(order_id, None)
    
    for symbol in symbols_to_remove:
        algo._submitted_orders.pop(symbol, None)


def portfolio_sanity_check(algo):
    """
    Check for portfolio value mismatches between QC and tracked positions.
    Fixed: Use CashBook["USD"].Amount for actual USD cash, not Portfolio.Cash
    (which in Cash account mode includes crypto holdings value).
    Fix 6: Trigger resync_holdings_full on mismatch and log detailed breakdown.
    """
    if algo.IsWarmingUp:
        return
    
    total_qc = algo.Portfolio.TotalPortfolioValue
    
    # Use actual USD cash, NOT Portfolio.Cash (which double-counts in Cash accounts)
    try:
        usd_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        usd_cash = algo.Portfolio.Cash
    
    tracked_value = 0.0
    tracked_pos_count = 0
    
    for sym in list(algo.entry_prices.keys()):
        if sym in algo.Securities:
            price = algo.Securities[sym].Price
            if sym in algo.Portfolio:
                qty = algo.Portfolio[sym].Quantity
                tracked_value += abs(qty) * price
                tracked_pos_count += 1
    
    expected = usd_cash + tracked_value
    
    # Use both percentage and minimum absolute threshold to avoid spam; 1-hour cooldown.
    abs_diff = abs(total_qc - expected)
    if total_qc > 1.0:
        pct_diff = abs_diff / total_qc
        should_warn = pct_diff > algo.portfolio_mismatch_threshold and abs_diff > algo.portfolio_mismatch_min_dollars
        if should_warn:
            if algo._last_mismatch_warning is None or (algo.Time - algo._last_mismatch_warning).total_seconds() >= algo.portfolio_mismatch_cooldown_seconds:
                algo.Debug(f"⚠️ PORTFOLIO MISMATCH: QC=${total_qc:.2f} tracked=${expected:.2f} diff=${abs_diff:.2f} ({pct_diff:.2%})")
                algo.Debug(f"  cash=${usd_cash:.2f} tracked_pos={tracked_pos_count} val=${tracked_value:.2f}")
                resync_holdings_full(algo)
                algo._last_mismatch_warning = algo.Time
