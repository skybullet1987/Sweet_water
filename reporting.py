# region imports
from AlgorithmImports import *
from execution import get_actual_position_count, is_invested_not_dust, persist_state
import numpy as np
# endregion


def review_performance(algo):
    """Review recent performance and adjust max_positions accordingly."""
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
    algo.Debug(f"Pos: {get_actual_position_count(algo)}/{algo.base_max_positions} | {algo.market_regime} {algo.volatility_regime} {algo.market_breadth:.0%}")
    algo.Debug(f"Trades: {total} | WR: {wr:.1%} | Avg: {avg:+.2%}")
    if algo._session_blacklist:
        algo.Debug(f"Blacklist: {len(algo._session_blacklist)}")
    for kvp in algo.Portfolio:
        if is_invested_not_dust(algo, kvp.Key):
            s = kvp.Key
            entry = algo.entry_prices.get(s, kvp.Value.AveragePrice)
            cur = algo.Securities[s].Price if s in algo.Securities else kvp.Value.Price
            pnl = (cur - entry) / entry if entry > 0 else 0
            algo.Debug(f"  {s.Value}: ${entry:.4f}→${cur:.4f} ({pnl:+.2%})")
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
    persist_state(algo)
