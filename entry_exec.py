"""
entry_exec.py — Entry Execution Coordinators

Contains the order-placement logic for both the trend engine and the chop engine.
These functions are extracted from main.py to keep the algorithm file within the
64 KB editor size limit without removing any functionality.

Functions
---------
execute_trend_trades(algo, candidates, threshold_now, effective_max_position_pct)
    Screen and place orders for the trend (MicroScalpEngine) regime.

run_chop_rebalance(algo)
    Screen and place orders for the chop (ChopEngine) regime.
"""

import itertools
from execution import (
    get_actual_position_count, has_open_orders, is_invested_not_dust,
    spread_ok, get_min_quantity, get_min_notional_usd, round_quantity,
    get_open_buy_orders_value, get_slippage_penalty, get_spread_pct,
    intraday_volume_ok, place_limit_or_market, debug_limited,
    SYMBOL_BLACKLIST, KRAKEN_SELL_FEE_BUFFER, get_hold_bucket,
)
from order_management import cancel_stale_new_orders
from trade_quality import (get_session_quality, adverse_selection_filter,
                            record_trade_metadata_on_entry)


def execute_trend_trades(algo, candidates, threshold_now, effective_max_position_pct):
    """
    Screen ranked candidates and place buy orders for the trend (MicroScalpEngine) regime.

    This function was formerly the ``_execute_trades`` method of SimplifiedCryptoStrategy.
    It is called from ``Rebalance()`` when the regime router routes to 'trend'.
    """
    if not algo._positions_synced:
        return
    if algo.LiveMode and algo.kraken_status in ("maintenance", "cancel_only"):
        return
    cancel_stale_new_orders(algo)
    if len(algo.Transactions.GetOpenOrders()) >= algo.max_concurrent_open_orders:
        return
    if algo._compute_portfolio_risk_estimate() > algo.portfolio_vol_cap:
        return

    try:
        available_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        available_cash = algo.Portfolio.Cash

    open_buy_orders_value = get_open_buy_orders_value(algo)

    if available_cash <= 0:
        debug_limited(algo, f"SKIP TRADES: No cash available (${available_cash:.2f})")
        return
    if open_buy_orders_value > available_cash * algo.open_orders_cash_threshold:
        debug_limited(algo, f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved (>{algo.open_orders_cash_threshold:.0%} of ${available_cash:.2f})")
        return

    reject_pending_orders = 0
    reject_open_orders = 0
    reject_already_invested = 0
    reject_spread = 0
    reject_exit_cooldown = 0
    reject_loss_cooldown = 0
    reject_correlation = 0
    reject_price_invalid = 0
    reject_price_too_low = 0
    reject_cash_reserve = 0
    reject_min_qty_too_large = 0
    reject_dollar_volume = 0
    reject_notional = 0
    success_count = 0

    in_post_warmup_grace = (
        getattr(algo, '_post_warmup_bars', 0) < getattr(algo, '_post_warmup_grace_bars', 0)
    )
    if in_post_warmup_grace:
        threshold_now += 0.05

    for cand in candidates:
        if algo.daily_trade_count >= algo.max_daily_trades:
            break
        if get_actual_position_count(algo) >= algo.max_positions:
            break
        sym = cand['symbol']
        net_score = cand.get('net_score', 0.5)
        if net_score < threshold_now:
            continue
        if sym in algo._pending_orders and algo._pending_orders[sym] > 0:
            reject_pending_orders += 1
            continue
        if has_open_orders(algo, sym):
            reject_open_orders += 1
            continue
        if is_invested_not_dust(algo, sym):
            reject_already_invested += 1
            continue
        if not spread_ok(algo, sym):
            reject_spread += 1
            continue
        if sym in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[sym]:
            reject_exit_cooldown += 1
            continue
        if sym.Value in algo._symbol_entry_cooldowns and algo.Time < algo._symbol_entry_cooldowns[sym.Value]:
            reject_loss_cooldown += 1
            continue
        if sym in algo._symbol_loss_cooldowns and algo.Time < algo._symbol_loss_cooldowns[sym]:
            reject_loss_cooldown += 1
            continue
        if not algo._check_correlation(sym):
            reject_correlation += 1
            continue
        sec = algo.Securities[sym]
        price = sec.Price
        if price is None or price <= 0:
            reject_price_invalid += 1
            continue
        if price < algo.min_price_usd:
            reject_price_too_low += 1
            continue

        try:
            available_cash = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = algo.Portfolio.Cash

        available_cash = max(0, available_cash - open_buy_orders_value)
        total_value = algo.Portfolio.TotalPortfolioValue
        # Minimal fee reserve only
        fee_reserve = max(total_value * algo.cash_reserve_pct, 0.50)
        reserved_cash = available_cash - fee_reserve
        if reserved_cash <= 0:
            reject_cash_reserve += 1
            continue

        min_qty = get_min_quantity(algo, sym)
        min_notional_usd = get_min_notional_usd(algo, sym)
        if min_qty * price > reserved_cash * 0.90:
            reject_min_qty_too_large += 1
            continue

        crypto = algo.crypto_data.get(sym)
        if not crypto:
            continue

        if crypto.get('trade_count_today', 0) >= algo.max_symbol_trades_per_day:
            continue

        atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
        if atr_val and price > 0:
            expected_move_pct = (atr_val * algo.atr_tp_mult) / price
            min_profit_gate = algo.min_expected_profit_pct
            min_required = algo.expected_round_trip_fees + algo.fee_slippage_buffer + min_profit_gate
            if expected_move_pct < min_required:
                continue

        if len(crypto['dollar_volume']) >= 3:
            dv_window = min(len(crypto['dollar_volume']), 12)
            # Compute recent_dv once and reuse for both the liquidity filter and
            # the liquidity cap below — avoids a duplicate list copy + numpy call.
            recent_dv = sum(itertools.islice(reversed(crypto['dollar_volume']), dv_window)) / dv_window
            dv_threshold = algo.min_dollar_volume_usd
            if recent_dv < dv_threshold:
                reject_dollar_volume += 1
                continue
        else:
            recent_dv = None

        vol = algo._annualized_vol(crypto)
        size = algo._scoring_engine.calculate_position_size(net_score, threshold_now, vol)

        # Session layer: apply size multiplier for current time-of-day quality.
        _s_thresh_adj, _s_size_mult, _s_spread_cap_mult = get_session_quality(
            algo, algo.Time.hour)
        size *= _s_size_mult
        if in_post_warmup_grace:
            size *= 0.50

        if algo._consecutive_loss_halve_remaining > 0:
            size *= 0.50

        if algo.volatility_regime == "high":
            size = min(size * 1.1, algo.position_size_pct)

        # Per-symbol penalty: halve size after consecutive losses
        sym_val = sym.Value
        if sym_val in algo._symbol_performance:
            recent_pnls = algo._symbol_performance[sym_val]  # deque, no list copy needed
            if len(recent_pnls) >= algo.symbol_penalty_threshold:
                recent_losses = sum(1 for p in itertools.islice(reversed(recent_pnls), algo.symbol_penalty_threshold) if p < 0)
                if recent_losses == algo.symbol_penalty_threshold:
                    size *= algo.symbol_penalty_size_mult
                    algo.Debug(f"PENALTY: {sym_val} size halved ({algo.symbol_penalty_threshold} consecutive losses)")

        slippage_penalty = get_slippage_penalty(algo, sym)
        size *= slippage_penalty

        # Apply BTC weakness size reduction
        size *= algo._btc_dump_size_mult

        # Spread penalty: linear reduction for spreads above 0.2%
        current_spread = get_spread_pct(algo, sym)
        if current_spread is not None and current_spread > 0.002:  # > 0.2% spread
            # 1% penalty per 0.05% excess spread, floored at 50%
            spread_penalty = max(0.5, 1.0 - (current_spread - 0.002) * 20)
            size *= spread_penalty

        val = reserved_cash * size

        # Liquidity cap: max 1% of estimated daily dollar volume (288 5-min bars/day)
        FIVE_MIN_BARS_PER_DAY = 288
        if recent_dv is not None:
            estimated_daily_dv = recent_dv * FIVE_MIN_BARS_PER_DAY
            liquidity_cap = estimated_daily_dv * algo.max_participation_rate
        else:
            liquidity_cap = float('inf')
        val = min(val, liquidity_cap)

        val = max(val, algo.min_notional)
        val = min(val, algo.Portfolio.TotalPortfolioValue * effective_max_position_pct)

        qty = round_quantity(algo, sym, val / price)
        if qty < min_qty:
            qty = round_quantity(algo, sym, min_qty)
            val = qty * price
        total_cost_with_fee = val * 1.006
        if total_cost_with_fee > available_cash:
            reject_cash_reserve += 1
            continue
        if val < min_notional_usd * algo.min_notional_fee_buffer or val < algo.min_notional or val > reserved_cash:
            reject_notional += 1
            continue

        try:
            sec = algo.Securities[sym]
            min_order_size = float(sec.SymbolProperties.MinimumOrderSize or 0)
            lot_size = float(sec.SymbolProperties.LotSize or 0)
            actual_min = max(min_order_size, lot_size)
            if actual_min > 0 and qty < actual_min:
                algo.Debug(f"REJECT ENTRY {sym.Value}: qty={qty} < min_order_size={actual_min} (unsellable)")
                reject_notional += 1
                continue
            if min_order_size > 0:
                post_fee_qty = qty * (1.0 - KRAKEN_SELL_FEE_BUFFER)
                if post_fee_qty < min_order_size:
                    required_qty = round_quantity(algo, sym, min_order_size / (1.0 - KRAKEN_SELL_FEE_BUFFER))
                    if required_qty * price <= available_cash * 0.99:  # 1% safety margin
                        qty = required_qty
                        val = qty * price
                    else:
                        algo.Debug(f"REJECT ENTRY {sym.Value}: post-fee qty={post_fee_qty:.6f} < min_order_size={min_order_size} and can't upsize")
                        reject_notional += 1
                        continue
        except Exception as e:
            algo.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

        if not intraday_volume_ok(algo, sym, val):
            continue

        # Adverse-selection filter: reject overextended / late / thin-market entries.
        _asel_pass, _asel_reason = adverse_selection_filter(
            algo, sym, crypto, cand.get('factors', {}), price)
        if not _asel_pass:
            debug_limited(algo, f"ASEL REJECT {sym.Value}: {_asel_reason}")
            continue

        try:
            components = cand.get('factors', {})
            # Breakout-like entry: strong vol ignition without mean-reversion support.
            # These rarely fill passively as maker — apply a higher non-fill penalty.
            is_breakout = (components.get('vol_ignition', 0) >= 0.20
                           and components.get('mean_reversion', 0) < 0.10)
            # Stress mode: temporarily inflate breakout penalty for robustness testing.
            _stress_nonfill = getattr(algo, 'stress_nonfill_penalty', 0.0)
            if _stress_nonfill > 0:
                # Store original, add stress penalty, restore after call
                _orig_penalty = getattr(algo, 'breakout_nonfill_penalty', 0.08)
                algo.breakout_nonfill_penalty = _orig_penalty + _stress_nonfill
            ticket = place_limit_or_market(algo, sym, qty, timeout_seconds=30, tag="Entry",
                                           is_breakout=is_breakout)
            if _stress_nonfill > 0:
                algo.breakout_nonfill_penalty = _orig_penalty
            if ticket is not None:
                algo._recent_tickets.append(ticket)
                sig_str = (f"vol={components.get('vol_ignition', 0):.2f} "
                           f"mean_rev={components.get('mean_reversion', 0):.2f} "
                           f"vwap={components.get('vwap_signal', 0):.2f}")
                algo.Debug(f"SCALP ENTRY: {sym.Value} | score={net_score:.2f} | ${val:.2f} | {sig_str}")
                # Track signal combination for attribution
                combo_parts = []
                if components.get('vol_ignition', 0) >= 0.10:
                    combo_parts.append('vol')
                if components.get('mean_reversion', 0) >= 0.10:
                    combo_parts.append('mean_rev')
                if components.get('vwap_signal', 0) >= 0.10:
                    combo_parts.append('vwap')
                signal_combo = '+'.join(combo_parts) if combo_parts else 'none'
                algo._entry_signal_combos[sym] = signal_combo
                # Tag this position as opened by the trend engine.
                algo._entry_engine[sym] = 'trend'
                # Record rich per-trade metadata for attribution analytics.
                record_trade_metadata_on_entry(
                    algo, sym, components, crypto,
                    spread_pct=get_spread_pct(algo, sym),
                    recent_dv=recent_dv,
                )
                success_count += 1
                algo.trade_count += 1
                crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                adx_ind = crypto.get('adx')
                is_choppy = (adx_ind is not None and adx_ind.IsReady
                             and adx_ind.Current.Value < 25)
                algo._choppy_regime_entries[sym] = is_choppy
                if algo._consecutive_loss_halve_remaining > 0:
                    algo._consecutive_loss_halve_remaining -= 1
                if algo.LiveMode:
                    algo._last_live_trade_time = algo.Time
        except Exception as e:
            algo.Debug(f"ORDER FAILED: {sym.Value} - {e}")
            algo._session_blacklist.add(sym.Value)
            continue
        if algo.LiveMode and success_count >= 3:
            break

    if success_count > 0 or (reject_exit_cooldown + reject_loss_cooldown) > 3:
        debug_limited(algo, f"EXECUTE: {success_count}/{len(candidates)} | rejects: cd={reject_exit_cooldown} loss={reject_loss_cooldown} corr={reject_correlation} dv={reject_dollar_volume}")


def run_chop_rebalance(algo):
    """
    Execute chop-engine signal screening and order placement.

    Called by Rebalance() when RegimeRouter.route() == 'chop'.
    Uses ChopEngine.calculate_score() and ChopEngine risk parameters
    instead of MicroScalpEngine, keeping the two engines fully isolated.

    This function was formerly the ``_run_chop_rebalance`` method of
    SimplifiedCryptoStrategy.
    """
    if not algo._positions_synced:
        return
    if algo.LiveMode and algo.kraken_status in ("maintenance", "cancel_only"):
        return

    pos_count = get_actual_position_count(algo)
    if pos_count >= algo.max_positions:
        return

    try:
        available_cash = algo.Portfolio.CashBook["USD"].Amount
    except (KeyError, AttributeError):
        available_cash = algo.Portfolio.Cash

    if available_cash <= 0:
        return

    _sess_thresh_adj, _sess_size_mult, _sess_spread_cap_mult = get_session_quality(
        algo, algo.Time.hour)
    chop_threshold = (algo._chop_engine.CHOP_ENTRY_THRESHOLD
                      + max(0.0, _sess_thresh_adj))
    in_post_warmup_grace = (
        getattr(algo, '_post_warmup_bars', 0) < getattr(algo, '_post_warmup_grace_bars', 0)
    )
    if in_post_warmup_grace:
        chop_threshold += 0.05

    chop_candidates = []
    for symbol in list(algo.crypto_data.keys()):
        if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in algo._session_blacklist:
            continue
        if symbol.Value in algo._symbol_entry_cooldowns and algo.Time < algo._symbol_entry_cooldowns[symbol.Value]:
            continue
        if has_open_orders(algo, symbol):
            continue
        if is_invested_not_dust(algo, symbol):
            continue
        if not spread_ok(algo, symbol):
            continue

        crypto = algo.crypto_data[symbol]
        if not algo._is_ready(crypto):
            continue

        # Engine coordination: skip symbols with an active chop cooldown.
        if algo._chop_engine.is_in_fail_cooldown(symbol):
            continue

        # Per-symbol daily cap for chop trades.
        if algo._chop_engine.daily_trade_count(symbol) >= algo._chop_engine.CHOP_MAX_TRADES_PER_SYMBOL_DAY:
            continue

        score, components = algo._chop_engine.calculate_score(crypto)
        if score < chop_threshold:
            continue

        chop_candidates.append({
            'symbol':    symbol,
            'score':     score,
            'components': components,
            'crypto':    crypto,
        })

    if not chop_candidates:
        return

    # Sort by score descending; take best candidate(s).
    chop_candidates.sort(key=lambda x: x['score'], reverse=True)

    cancel_stale_new_orders(algo)

    success_count = 0
    for cand in chop_candidates:
        if get_actual_position_count(algo) >= algo.max_positions:
            break
        if algo.daily_trade_count >= algo.max_daily_trades:
            break

        sym      = cand['symbol']
        score    = cand['score']
        crypto   = cand['crypto']
        comps    = cand['components']

        # Re-check cooldowns / invested status after prior iterations.
        if is_invested_not_dust(algo, sym):
            continue
        if has_open_orders(algo, sym):
            continue
        if algo._chop_engine.is_in_fail_cooldown(sym):
            continue
        if sym in algo._exit_cooldowns and algo.Time < algo._exit_cooldowns[sym]:
            continue
        if sym in algo._symbol_loss_cooldowns and algo.Time < algo._symbol_loss_cooldowns[sym]:
            continue
        if not algo._check_correlation(sym):
            continue

        sec   = algo.Securities[sym]
        price = sec.Price
        if price is None or price <= 0 or price < algo.min_price_usd:
            continue

        try:
            available_cash = algo.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = algo.Portfolio.Cash

        # Chop position sizing: own scale (15–25 %).
        size_frac = algo._chop_engine.calculate_position_size(score)
        size_frac *= _sess_size_mult
        if in_post_warmup_grace:
            size_frac *= 0.50
        if algo._consecutive_loss_halve_remaining > 0:
            size_frac *= 0.50

        val = available_cash * size_frac
        val = max(val, algo.min_notional)
        val = min(val, algo.Portfolio.TotalPortfolioValue * algo.max_position_pct)

        min_qty          = get_min_quantity(algo, sym)
        min_notional_usd = get_min_notional_usd(algo, sym)
        qty              = round_quantity(algo, sym, val / price)
        if qty < min_qty:
            qty = round_quantity(algo, sym, min_qty)
            val = qty * price

        total_cost = val * 1.006
        if total_cost > available_cash:
            continue
        if val < min_notional_usd * algo.min_notional_fee_buffer or val < algo.min_notional:
            continue

        try:
            ticket = place_limit_or_market(algo, sym, qty, timeout_seconds=30,
                                           tag="Chop Entry")
            if ticket is not None:
                algo._recent_tickets.append(ticket)
                sig_str = ' '.join(f"{k[:6]}={v:.2f}" for k, v in comps.items() if v > 0)
                algo.Debug(
                    f"CHOP ENTRY: {sym.Value} | score={score:.2f} | "
                    f"${val:.2f} | {sig_str}"
                )
                # Engine attribution and tracking.
                algo._entry_engine[sym] = 'chop'
                algo._chop_engine.register_entry(sym)
                # Mark as NOT a choppy trend entry (chop engine manages its own exit).
                algo._choppy_regime_entries[sym] = False
                # Signal-combo attribution (chop engine signals).
                combo_parts = [k for k, v in comps.items() if v >= 0.10]
                signal_combo = '+'.join(combo_parts) if combo_parts else 'none'
                algo._entry_signal_combos[sym] = signal_combo
                # Record metadata for attribution.
                record_trade_metadata_on_entry(
                    algo, sym, comps, crypto,
                    spread_pct=get_spread_pct(algo, sym),
                    recent_dv=None,
                )
                success_count += 1
                algo.trade_count += 1
                crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                if algo._consecutive_loss_halve_remaining > 0:
                    algo._consecutive_loss_halve_remaining -= 1
                algo.daily_trade_count += 1
                if algo.LiveMode:
                    algo._last_live_trade_time = algo.Time
        except Exception as e:
            algo.Debug(f"CHOP ORDER FAILED: {sym.Value} - {e}")
            algo._session_blacklist.add(sym.Value)
            continue

        if algo.LiveMode and success_count >= 2:
            # Limit live chop entries per bar to reduce overtrading.
            break

    if success_count > 0:
        debug_limited(algo, f"CHOP EXECUTE: {success_count} entries placed | regime=chop")
