# region imports
from AlgorithmImports import *
from execution import get_actual_position_count, is_invested_not_dust, persist_state
import numpy as np
# endregion


def review_performance(algo):
    """Review recent performance and adjust max_positions accordingly."""
    if getattr(algo, 'disable_performance_review_max_position_adjustments', False):
        return
    if algo.IsWarmingUp or len(algo.trade_log) < 10:
        return
    
    recent_trades = list(algo.trade_log)[-15:]
    if len(recent_trades) == 0:
        return
    
    recent_win_rate = sum(1 for t in recent_trades if t['pnl_pct'] > 0) / len(recent_trades)
    recent_avg_pnl = np.mean([t['pnl_pct'] for t in recent_trades])
    old_max = algo.max_positions
    
    if recent_win_rate < 0.2 and recent_avg_pnl < -0.05:
        algo.max_positions = 1
        if old_max != 1:
            algo.Debug(f"PERFORMANCE DECAY: max_pos=1 (WR:{recent_win_rate:.0%}, PnL:{recent_avg_pnl:+.2%})")
    elif recent_win_rate > 0.35 and recent_avg_pnl > -0.01:
        algo.max_positions = algo.base_max_positions
        if old_max != algo.base_max_positions:
            algo.Debug(f"PERFORMANCE RECOVERY: max_pos={algo.base_max_positions}")


def daily_report(algo):
    """Generate daily report with portfolio status and position details."""
    if algo.IsWarmingUp:
        return

    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    avg = algo.total_pnl / total if total > 0 else 0
    algo.Debug(f"=== DAILY {algo.Time.date()} ===")
    algo.Debug(f"Portfolio: ${algo.Portfolio.TotalPortfolioValue:.2f} | Cash: ${algo.Portfolio.Cash:.2f}")
    if getattr(algo, 'comparison_mode', False):
        algo.Debug("Mode: comparison (path-dependent adaptive features reduced)")
    # Include regime router state in the daily header.
    router_regime = getattr(getattr(algo, '_regime_router', None), 'current_regime', 'n/a')
    algo.Debug(f"Pos: {get_actual_position_count(algo)}/{algo.base_max_positions} | {algo.market_regime} {algo.volatility_regime} {algo.market_breadth:.0%} | router:{router_regime}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    if algo._session_blacklist:
        algo.Debug(f"Blacklist: {len(algo._session_blacklist)}")
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key):
            s = kvp.Key
            entry = algo.entry_prices.get(s, kvp.Value.AveragePrice)
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            pnl = (cur - entry) / entry if entry > 0 else 0
            engine_tag = algo._entry_engine.get(s, '?') if hasattr(algo, '_entry_engine') else '?'
            algo.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%}) [{engine_tag}]")
    # Engine PnL summary (trend vs chop).
    if hasattr(algo, 'pnl_by_engine') and algo.pnl_by_engine:
        algo.Debug("  --- PnL by engine ---")
        for engine in ('trend', 'chop'):
            pnls = algo.pnl_by_engine.get(engine, [])
            if not pnls:
                continue
            avg_e = sum(pnls) / len(pnls)
            wr_e  = sum(1 for p in pnls if p > 0) / len(pnls)
            tot_e = sum(pnls)
            algo.Debug(f"  {engine}: {len(pnls)} trades | WR:{wr_e:.0%} | Avg:{avg_e:+.3%} | Total:{tot_e:+.3%}")
    # Regime PnL summary
    if hasattr(algo, 'pnl_by_regime') and algo.pnl_by_regime:
        algo.Debug("  --- PnL by regime ---")
        for regime, pnls in algo.pnl_by_regime.items():
            avg = np.mean(pnls) if pnls else 0
            wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            algo.Debug(f"  {regime}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
    if hasattr(algo, 'pnl_by_vol_regime') and algo.pnl_by_vol_regime:
        algo.Debug("  --- PnL by vol regime ---")
        for vol_regime, pnls in algo.pnl_by_vol_regime.items():
            avg = np.mean(pnls) if pnls else 0
            wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            algo.Debug(f"  {vol_regime}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
    # Exit-tag performance summary
    if hasattr(algo, 'pnl_by_tag') and algo.pnl_by_tag:
        algo.Debug("  --- PnL by exit tag ---")
        for tag, pnls in sorted(algo.pnl_by_tag.items()):
            if len(pnls) == 0:
                continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            total = sum(pnls)
            algo.Debug(f"  {tag}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
    # Signal-combination performance summary
    if hasattr(algo, 'pnl_by_signal_combo') and algo.pnl_by_signal_combo:
        algo.Debug("  --- PnL by signal combo ---")
        for combo, pnls in sorted(algo.pnl_by_signal_combo.items()):
            if len(pnls) == 0:
                continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            total = sum(pnls)
            algo.Debug(f"  {combo}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
    # Hold-time PnL summary
    if hasattr(algo, 'pnl_by_hold_time') and algo.pnl_by_hold_time:
        algo.Debug("  --- PnL by hold time ---")
        for bucket in ['<30min', '30min-2h', '2h-6h', '6h+', 'unknown']:
            pnls = algo.pnl_by_hold_time.get(bucket)
            if not pnls:
                continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            total = sum(pnls)
            algo.Debug(f"  {bucket}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
    # Session PnL summary
    if hasattr(algo, 'pnl_by_session') and algo.pnl_by_session:
        algo.Debug("  --- PnL by session ---")
        for bucket in ['asia_dead', 'asia', 'eu_open', 'eu_main',
                       'us_open', 'us_main', 'us_eve']:
            pnls = algo.pnl_by_session.get(bucket)
            if not pnls:
                continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            total = sum(pnls)
            algo.Debug(f"  {bucket}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
    # ADX bucket PnL (compact)
    if hasattr(algo, 'pnl_by_adx_bucket') and algo.pnl_by_adx_bucket:
        algo.Debug("  --- PnL by ADX bucket ---")
        for bucket in ['flat', 'mild', 'moderate', 'strong', 'unknown']:
            pnls = algo.pnl_by_adx_bucket.get(bucket)
            if not pnls: continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            algo.Debug(f"  {bucket}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
    # Spread bucket PnL (compact)
    if hasattr(algo, 'pnl_by_spread_bucket') and algo.pnl_by_spread_bucket:
        algo.Debug("  --- PnL by spread bucket ---")
        for bucket in ['<0.1%', '0.1-0.3%', '0.3-0.5%', '0.5-1.0%', '>1.0%', 'unknown']:
            pnls = algo.pnl_by_spread_bucket.get(bucket)
            if not pnls: continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            algo.Debug(f"  {bucket}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
    # RS bucket PnL (compact)
    if hasattr(algo, 'pnl_by_rs_bucket') and algo.pnl_by_rs_bucket:
        algo.Debug("  --- PnL by RS-vs-BTC bucket ---")
        for bucket in ['strong_neg', 'mild_neg', 'neutral', 'mild_pos', 'strong_pos', 'unknown']:
            pnls = algo.pnl_by_rs_bucket.get(bucket)
            if not pnls: continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            algo.Debug(f"  {bucket}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
    # Setup archetype PnL summary
    if hasattr(algo, 'pnl_by_archetype') and algo.pnl_by_archetype:
        algo.Debug("  --- PnL by setup archetype ---")
        for arch, pnls in sorted(algo.pnl_by_archetype.items()):
            if not pnls:
                continue
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls)
            total = sum(pnls)
            algo.Debug(f"  {arch}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
    # MFE/MAE by archetype (compact)
    mfe_arch = getattr(algo, 'mfe_by_archetype', {})
    mae_arch = getattr(algo, 'mae_by_archetype', {})
    if mfe_arch:
        algo.Debug("  --- MFE/MAE by archetype ---")
        for arch in sorted(mfe_arch.keys()):
            mfes = mfe_arch.get(arch, [])
            maes = mae_arch.get(arch, [])
            if not mfes:
                continue
            avg_mfe = sum(mfes) / len(mfes)
            avg_mae = sum(maes) / len(maes) if maes else 0
            algo.Debug(f"  {arch}: n={len(mfes)} | AvgMFE:{avg_mfe:+.3%} | AvgMAE:{avg_mae:+.3%}")
    persist_state(algo)


def final_report(algo):
    """Print end-of-algorithm summary: headline stats, per-category PnL breakdowns, MFE/MAE."""
    total = algo.winning_trades + algo.losing_trades
    wr = algo.winning_trades / total if total > 0 else 0
    algo.Debug("=== FINAL ===")
    algo.Debug(f"Trades: {algo.trade_count} | WR: {wr:.1%}")
    algo.Debug(f"Final: ${algo.Portfolio.TotalPortfolioValue:.2f}")
    algo.Debug(f"PnL: {algo.total_pnl:+.2%}")
    persist_state(algo)

    # ── Dual-engine comparison (trend vs chop) ────────────────────────────
    # This section makes it easy to compare the two engines side-by-side and
    # verify that trend-mode behavior was preserved after the architecture change.
    pnl_by_engine = getattr(algo, 'pnl_by_engine', {})
    if pnl_by_engine:
        algo.Debug("=== PnL BY ENGINE (trend vs chop) ===")
        for engine in ('trend', 'chop'):
            v = pnl_by_engine.get(engine, [])
            if not v:
                algo.Debug(f"  {engine}: 0 trades")
                continue
            n    = len(v)
            avg  = sum(v) / n
            wr_e = sum(1 for p in v if p > 0) / n
            tot  = sum(v)
            exp  = avg * wr_e - abs(avg) * (1 - wr_e)   # simplified expectancy
            algo.Debug(
                f"  {engine}: {n} trades | WR:{wr_e:.0%} | "
                f"Avg:{avg:+.3%} | Total:{tot:+.3%} | "
                f"Expectancy:{exp:+.4f} "
                f"{'✅' if avg > 0 else '❌'}"
            )
        # Exit-tag breakdown per engine for chop attribution.
        pnl_by_tag = getattr(algo, 'pnl_by_tag', {})
        chop_tags  = [t for t in pnl_by_tag if t.startswith('Chop')]
        if chop_tags:
            algo.Debug("  --- Chop exit tags ---")
            for tag in sorted(chop_tags):
                v = pnl_by_tag.get(tag, [])
                if not v: continue
                n = len(v); avg = sum(v)/n; wr_t = sum(1 for p in v if p>0)/n
                algo.Debug(f"  {tag}: {n} | WR:{wr_t:.0%} | Avg:{avg:+.3%}")

    _HOLD_ORDER = ['<30min', '30min-2h', '2h-6h', '6h+', 'unknown']
    _SESSION_ORDER = ['asia_dead', 'asia', 'eu_open', 'eu_main', 'us_open', 'us_main', 'us_eve']
    for label, d, order in [
        ("REGIME",         getattr(algo, 'pnl_by_regime', {}),        None),
        ("VOL REGIME",     getattr(algo, 'pnl_by_vol_regime', {}),     None),
        ("EXIT TAG",       getattr(algo, 'pnl_by_tag', {}),            None),
        ("SIGNAL COMBO",   getattr(algo, 'pnl_by_signal_combo', {}),   None),
        ("HOLD TIME",      getattr(algo, 'pnl_by_hold_time', {}),      _HOLD_ORDER),
        ("SESSION",        getattr(algo, 'pnl_by_session', {}),        _SESSION_ORDER),
        ("SETUP ARCHETYPE",getattr(algo, 'pnl_by_archetype', {}),      None),
        ("ADX BUCKET",     getattr(algo, 'pnl_by_adx_bucket', {}),     None),
        ("SPREAD BUCKET",  getattr(algo, 'pnl_by_spread_bucket', {}),  None),
        ("DV BUCKET",      getattr(algo, 'pnl_by_dv_bucket', {}),      None),
        ("RS BUCKET",      getattr(algo, 'pnl_by_rs_bucket', {}),      None),
    ]:
        if not d: continue
        algo.Debug(f"=== PnL BY {label} ===")
        items = [(b, d.get(b)) for b in order] if order else sorted(d.items())
        for k, v in items:
            if not v: continue
            n = len(v); avg = sum(v)/n; wr = sum(1 for p in v if p>0)/n; tot = sum(v)
            algo.Debug(f"  {k}: {n} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{tot:+.3%} {'✅' if avg > 0 else '❌'}")
    algo.Debug("=== SESSION SIZE MULTIPLIERS (current) ===")
    for _sname in ['asia_dead', 'asia', 'eu_open', 'eu_main', 'us_open', 'us_main', 'us_eve']:
        algo.Debug(f"  {_sname}: size_mult={getattr(algo, f'_session_size_{_sname}', 1.0):.2f}")
    mfe_arch = getattr(algo, 'mfe_by_archetype', {})
    mae_arch = getattr(algo, 'mae_by_archetype', {})
    if mfe_arch:
        algo.Debug("=== MFE/MAE BY ARCHETYPE ===")
        for arch in sorted(mfe_arch.keys()):
            mfes = mfe_arch.get(arch, [])
            maes = mae_arch.get(arch, [])
            if not mfes: continue
            algo.Debug(f"  {arch}: n={len(mfes)} | AvgMFE:{sum(mfes)/len(mfes):+.3%} | AvgMAE:{sum(maes)/len(maes) if maes else 0:+.3%}")
