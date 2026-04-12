from AlgorithmImports import *
from collections import deque
import numpy as np
import math
import itertools
from execution import get_spread_pct, _estimate_backtest_spread


def initialize_symbol(algo, symbol):
    algo.crypto_data[symbol] = {
        'prices': deque(maxlen=algo.lookback),
        'returns': deque(maxlen=algo.lookback),
        'volume': deque(maxlen=algo.lookback),
        'volume_ma': deque(maxlen=algo.medium_period),
        'dollar_volume': deque(maxlen=algo.lookback),
        'ema_ultra_short': ExponentialMovingAverage(algo.ultra_short_period),
        'ema_short': ExponentialMovingAverage(algo.short_period),
        'ema_medium': ExponentialMovingAverage(algo.medium_period),
        'ema_5': ExponentialMovingAverage(5),
        'atr': AverageTrueRange(14),
        'adx': AverageDirectionalIndex(algo.adx_min_period),
        'volatility': deque(maxlen=algo.medium_period),
        'rsi': RelativeStrengthIndex(7),
        'rs_vs_btc': deque(maxlen=algo.medium_period),
        'last_price': 0,
        'recent_net_scores': deque(maxlen=3),
        'spreads': deque(maxlen=algo.spread_median_window),
        'trail_stop': None,
        'highs': deque(maxlen=algo.lookback),
        'lows': deque(maxlen=algo.lookback),
        'bb_upper': deque(maxlen=algo.short_period),
        'bb_lower': deque(maxlen=algo.short_period),
        'bb_width': deque(maxlen=algo.medium_period),
        'trade_count_today': 0,
        'last_loss_time': None,
        'bid_size': 0.0,
        'ask_size': 0.0,
        # True intraday VWAP: accumulates all bars since midnight UTC, resets daily.
        # Using plain lists (no maxlen) so we always hold the full day's bars.
        'vwap_pv': [],
        'vwap_v': [],
        'vwap': 0.0,
        'volume_long': deque(maxlen=1440),
        'vwap_sd': 0.0,
        'vwap_sd2_lower': 0.0,
        'vwap_sd3_lower': 0.0,
        '_vwap_date': None,           # tracks the UTC date of the last VWAP reset
        'consolidator': None,
        # Incremental rolling-stat accumulators (avoid deque→list→numpy each bar)
        '_vma_window':   deque(maxlen=algo.short_period),   # 6-bar rolling volume mean
        '_vma_sum':      0.0,
        '_vol_window':   deque(maxlen=10),                  # 10-bar rolling return std
        '_vol_sum':      0.0,
        '_vol_sum_sq':   0.0,
        '_bb_window':    deque(maxlen=algo.medium_period),  # 12-bar rolling price stats (BB)
        '_bb_sum':       0.0,
        '_bb_sum_sq':    0.0,
        '_vol_long_sum': 0.0,                               # running sum of volume_long
        '_vol20_window': deque(maxlen=20),                  # 20-bar rolling volume for scoring
        '_vol20_sum':    0.0,
    }
    # Create 5-minute consolidator for this symbol
    try:
        consolidator = TradeBarConsolidator(timedelta(minutes=5))
        consolidator.DataConsolidated += algo._on_five_minute_bar
        algo.SubscriptionManager.AddConsolidator(symbol, consolidator)
        algo.crypto_data[symbol]['consolidator'] = consolidator
    except Exception as e:
        algo.Debug(f"Warning: Could not add consolidator for {symbol.Value} - {e}")


def update_symbol_data(algo, symbol, bar, quote_bar=None):
    crypto = algo.crypto_data[symbol]
    price = float(bar.Close)
    high = float(bar.High)
    low = float(bar.Low)
    volume = float(bar.Volume)
    crypto['prices'].append(price)
    # Maintain BB incremental window in sync with prices (O(1) update)
    bb_w = crypto['_bb_window']
    if len(bb_w) == bb_w.maxlen:
        old_price = bb_w[0]
        crypto['_bb_sum']    -= old_price
        crypto['_bb_sum_sq'] -= old_price * old_price
    bb_w.append(price)
    crypto['_bb_sum']    += price
    crypto['_bb_sum_sq'] += price * price
    crypto['highs'].append(high)
    crypto['lows'].append(low)
    if crypto['last_price'] > 0:
        ret = (price - crypto['last_price']) / crypto['last_price']
        crypto['returns'].append(ret)
        # Incremental rolling volatility — rolling sum-of-squares (O(1), no list copy)
        vol_w = crypto['_vol_window']
        if len(vol_w) == vol_w.maxlen:
            old_return = vol_w[0]
            crypto['_vol_sum']    -= old_return
            crypto['_vol_sum_sq'] -= old_return * old_return
        vol_w.append(ret)
        crypto['_vol_sum']    += ret
        crypto['_vol_sum_sq'] += ret * ret
        if len(vol_w) >= 10:
            n = len(vol_w)
            vol_mean = crypto['_vol_sum'] / n
            # Sample variance (Bessel's correction: divide by n-1, not n) to avoid
            # systematically underestimating volatility by ~sqrt(n/(n-1)).
            vol_var  = (crypto['_vol_sum_sq'] - n * vol_mean * vol_mean) / (n - 1)
            crypto['volatility'].append(math.sqrt(max(vol_var, 0.0)))
    crypto['last_price'] = price
    crypto['volume'].append(volume)
    crypto['dollar_volume'].append(price * volume)
    # Incremental 6-bar rolling volume mean (O(1), avoids np.mean(list(volume)[-6:]))
    vma_w = crypto['_vma_window']
    if len(vma_w) == vma_w.maxlen:
        crypto['_vma_sum'] -= vma_w[0]
    vma_w.append(volume)
    crypto['_vma_sum'] += volume
    if len(vma_w) >= algo.short_period:
        crypto['volume_ma'].append(crypto['_vma_sum'] / len(vma_w))
    # Maintain 20-bar volume window and volume_long running sums for scoring (O(1))
    v20_w = crypto['_vol20_window']
    if len(v20_w) == v20_w.maxlen:
        crypto['_vol20_sum'] -= v20_w[0]
    v20_w.append(volume)
    crypto['_vol20_sum'] += volume
    vol_long_w = crypto['volume_long']
    if len(vol_long_w) == vol_long_w.maxlen:
        crypto['_vol_long_sum'] -= vol_long_w[0]
    vol_long_w.append(volume)
    crypto['_vol_long_sum'] += volume
    crypto['ema_ultra_short'].Update(bar.EndTime, price)
    crypto['ema_short'].Update(bar.EndTime, price)
    crypto['ema_medium'].Update(bar.EndTime, price)
    crypto['ema_5'].Update(bar.EndTime, price)
    crypto['atr'].Update(bar)
    crypto['adx'].Update(bar)
    # True intraday VWAP: reset accumulator lists at each new UTC day so the
    # VWAP level reflects today's actual traded price rather than an arbitrary
    # rolling 20-bar window that has no institutional significance.
    # _VWAP_MAX_BARS is a safety cap: 5-min bars × 24 h × 2 = 576 covers a full
    # day with margin; guarding against list runaway if the date-reset ever
    # fails (e.g. data replayed without time advancing).
    _VWAP_MAX_BARS = 576
    bar_date = bar.EndTime.date()
    if crypto.get('_vwap_date') != bar_date:
        crypto['_vwap_date'] = bar_date
        crypto['vwap_pv'] = []
        crypto['vwap_v'] = []
        crypto['vwap'] = 0.0
        crypto['vwap_sd'] = 0.0
        crypto['vwap_sd2_lower'] = 0.0
        crypto['vwap_sd3_lower'] = 0.0
    crypto['vwap_pv'].append(price * volume)
    crypto['vwap_v'].append(volume)
    # Safety trim: keep at most _VWAP_MAX_BARS entries so the lists stay
    # bounded even if the date-reset is somehow skipped.
    if len(crypto['vwap_pv']) > _VWAP_MAX_BARS:
        crypto['vwap_pv'] = crypto['vwap_pv'][-_VWAP_MAX_BARS:]
        crypto['vwap_v']  = crypto['vwap_v'][-_VWAP_MAX_BARS:]
    total_v = sum(crypto['vwap_v'])
    if total_v > 0:
        crypto['vwap'] = sum(crypto['vwap_pv']) / total_v
    if len(crypto['vwap_v']) >= 5 and crypto['vwap'] > 0:
        vwap_val = crypto['vwap']
        total_v_sd = sum(crypto['vwap_v'])
        if total_v_sd > 0:
            # True volume-weighted variance: Σ(v_i × (p_i − VWAP)²) / Σ(v_i)
            bar_prices = [pv / v for pv, v in zip(crypto['vwap_pv'], crypto['vwap_v']) if v > 0]
            bar_vols   = [v for v in crypto['vwap_v'] if v > 0]
            if len(bar_prices) >= 5:
                vw_var = sum(bv * (bp - vwap_val) ** 2 for bp, bv in zip(bar_prices, bar_vols)) / total_v_sd
                sd = math.sqrt(max(vw_var, 0.0))
                crypto['vwap_sd'] = sd
                crypto['vwap_sd2_lower'] = vwap_val - 2.0 * sd
                crypto['vwap_sd3_lower'] = vwap_val - 3.0 * sd
    crypto['rsi'].Update(bar.EndTime, price)
    if len(crypto['returns']) >= algo.short_period and len(algo.btc_returns) >= algo.short_period:
        # sum last short_period elements without creating a full list copy
        coin_ret = sum(itertools.islice(reversed(crypto['returns']), algo.short_period))
        btc_ret  = sum(itertools.islice(reversed(algo.btc_returns),  algo.short_period))
        crypto['rs_vs_btc'].append(coin_ret - btc_ret)
    # Bollinger Bands via incremental stats (O(1), replaces np.array(list(prices)[-12:]))
    if len(bb_w) >= algo.medium_period:
        n    = len(bb_w)
        mean = crypto['_bb_sum'] / n
        # Sample variance (Bessel's correction) for unbiased BB band width.
        var  = (crypto['_bb_sum_sq'] - n * mean * mean) / (n - 1)
        std  = math.sqrt(max(var, 0.0))
        if std > 0:
            crypto['bb_upper'].append(mean + 2 * std)
            crypto['bb_lower'].append(mean - 2 * std)
            crypto['bb_width'].append(4 * std / mean if mean > 0 else 0)
    sp = get_spread_pct(algo, symbol)
    if sp is None and not algo.LiveMode:
        # Backtest: estimate spread from dollar volume. Appending happens here
        # (once per bar, in update_symbol_data) rather than inside
        # _estimate_backtest_spread(), which is also called by spread_ok() —
        # twice per bar per symbol (once during candidate screening, once during
        # execution).  Keeping the append here guarantees exactly one entry per
        # 5-minute bar regardless of how many times spread_ok() is invoked.
        sp = _estimate_backtest_spread(algo, symbol)
    if sp is not None:
        crypto['spreads'].append(sp)
    if quote_bar is not None:
        try:
            bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
            ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
            if bid_sz > 0 or ask_sz > 0:
                crypto['bid_size'] = bid_sz
                crypto['ask_size'] = ask_sz
        except Exception:
            pass


def update_market_context(algo):
    if len(algo.btc_prices) >= 48:
        # Use incremental accumulators instead of full deque→list→numpy each bar
        current_btc = algo.btc_prices[-1]
        btc_sma = (algo._btc_sma48_sum / len(algo._btc_sma48_window)
                   if len(algo._btc_sma48_window) > 0 else 0.0)
        btc_mom_12 = (algo._btc_mom12_sum / len(algo._btc_mom12_window)
                      if len(algo._btc_mom12_window) >= 12 else 0.0)
        if current_btc > btc_sma * 1.02:
            new_regime = "bull"
        elif current_btc < btc_sma * 0.98:
            new_regime = "bear"
        else:
            new_regime = "sideways"
        if new_regime == "sideways" and len(algo.btc_returns) >= 12:
            if btc_mom_12 > 0.0001:
                new_regime = "bull"
            elif btc_mom_12 < -0.0001:
                new_regime = "bear"
        # Hysteresis: only change if held for 3+ bars
        if new_regime != algo.market_regime:
            algo._regime_hold_count += 1
            if algo._regime_hold_count >= 3:
                algo.market_regime = new_regime
                algo._regime_hold_count = 0
        else:
            algo._regime_hold_count = 0
    if len(algo.btc_volatility) >= 5:
        current_vol = algo.btc_volatility[-1]
        # Use incremental running sum instead of np.mean(list(btc_volatility))
        avg_vol = (algo._btc_vol_avg_sum / len(algo.btc_volatility)
                   if len(algo.btc_volatility) > 0 else 0.0)
        if current_vol > avg_vol * 1.5:
            algo.volatility_regime = "high"
        elif current_vol < avg_vol * 0.5:
            algo.volatility_regime = "low"
        else:
            algo.volatility_regime = "normal"
    uptrend_count = 0
    total_ready = 0
    for crypto in algo.crypto_data.values():
        if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
            total_ready += 1
            if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                uptrend_count += 1
    if total_ready > 5:
        algo.market_breadth = uptrend_count / total_ready


def compute_ranking_overlay(algo, cand):
    """Small bounded attribution-aware ranking adjustment; net_score remains dominant."""
    adj = 0.0
    factors = cand.get('factors', {})

    if hasattr(algo, 'pnl_by_signal_combo') and len(algo.pnl_by_signal_combo) >= 2:
        combo_parts = []
        if factors.get('vol_ignition',   0) >= 0.10: combo_parts.append('vol')
        if factors.get('mean_reversion', 0) >= 0.10: combo_parts.append('mean_rev')
        if factors.get('vwap_signal',    0) >= 0.10: combo_parts.append('vwap')
        combo = '+'.join(combo_parts) if combo_parts else 'none'
        combo_avgs = {c: np.mean(v) for c, v in algo.pnl_by_signal_combo.items() if len(v) >= 3}
        if combo in combo_avgs and len(combo_avgs) >= 2:
            all_avgs = list(combo_avgs.values())
            overall = np.mean(all_avgs)
            std = max(np.std(all_avgs), 1e-6)
            z = (combo_avgs[combo] - overall) / std
            adj += float(np.clip(z * 0.02, -algo.ranking_combo_bonus_cap, algo.ranking_combo_bonus_cap))

    sym_val = cand['symbol'].Value
    if sym_val in algo._symbol_performance:
        recent = algo._symbol_performance[sym_val]  # deque, no list copy needed
        if len(recent) >= 3:
            sym_avg = np.mean(recent)
            adj += float(np.clip(sym_avg * 2.0, -algo.ranking_symbol_bonus_cap, algo.ranking_symbol_bonus_cap))

    return adj
