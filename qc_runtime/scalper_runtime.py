"""Scalper runtime helpers extracted from main algorithm for size and readability."""

from __future__ import annotations

import math
import statistics
from collections import deque
from datetime import timedelta

import pandas as pd
from config import CONFIG

try:  # pragma: no cover
    from AlgorithmImports import Slice
except Exception:  # pragma: no cover
    class Slice:  # type: ignore
        pass

from execution import PositionState, can_afford, free_cash_usd, get_min_notional_usd, place_entry, place_limit_or_market, round_quantity, smart_liquidate
from risk import equity_kill_switch_active, should_auto_reset_latches
from scalper import corr_hits_from_state, effective_position_size_pct, evaluate_entry, evaluate_exit, regime_for, vol_target_qty

INFINITE_HELD_HOURS = 10**9

def _scalper_sleeve_allocations(self) -> dict[str, float]:
    def _sharpe(values):
        if len(values) < 5:
            return 0.0
        mu = statistics.fmean(values)
        sd = statistics.pstdev(values)
        return 0.0 if sd <= 1e-9 else mu / sd

    sleeve_pnls = getattr(self, "_scalper_sleeve_pnls", {"meanrev": [], "momentum": []})
    mr = _sharpe(sleeve_pnls.get("meanrev", []))
    mom = _sharpe(sleeve_pnls.get("momentum", []))
    mr_w = max(0.05, mr + 1.0)
    mom_w = max(0.05, mom + 1.0)
    total = mr_w + mom_w
    return {"meanrev": mr_w / total, "momentum": mom_w / total}

def _scalper_corr_hits(self, symbol, held: list) -> int:
    threshold = float(getattr(self.config, "scalper_corr_threshold", 0.70) or 0.70)
    state = getattr(self.feature_engine, "_state", {})
    return corr_hits_from_state(feature_state=state, symbol=symbol, held=held, threshold=threshold)

def _hourly_log_returns(self, symbol) -> list[float]:
    state = getattr(self, "crypto_data", {}).get(symbol, {})
    closes = list(state.get("prices", []))
    out = []
    for i in range(1, len(closes)):
        prev = float(closes[i - 1] or 0.0)
        cur = float(closes[i] or 0.0)
        if prev > 0 and cur > 0:
            out.append(float(math.log(cur / prev)))
    lookback = int(getattr(self.config, "scalper_beta_lookback_hours", 24 * 7) or (24 * 7))
    return out[-lookback:]

def _beta_to_btc(self, symbol, btc_symbol) -> float:
    if btc_symbol is None:
        return 0.0
    sym_ret = self._hourly_log_returns(symbol)
    btc_ret = self._hourly_log_returns(btc_symbol)
    n = min(len(sym_ret), len(btc_ret))
    if n < 24:
        return 0.0
    s = pd.Series(sym_ret[-n:], dtype=float)
    b = pd.Series(btc_ret[-n:], dtype=float)
    var_b = float(b.var())
    if not math.isfinite(var_b) or var_b <= 1e-12:
        return 0.0
    beta = float(s.cov(b) / var_b)
    return beta if math.isfinite(beta) else 0.0

def _portfolio_beta_sum_with_candidate(self, candidate, held: list, btc_symbol) -> float:
    equity = float(getattr(self.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
    if equity <= 0:
        return 0.0
    beta_sum = 0.0
    for sym in held:
        beta = self._beta_to_btc(sym, btc_symbol)
        sec = self.Securities.get(sym)
        px = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
        qty = float(getattr(self.Portfolio[sym], "Quantity", 0.0) or 0.0)
        weight = abs(qty * px) / max(equity, 1.0)
        beta_sum += beta * weight
    candidate_beta = self._beta_to_btc(candidate, btc_symbol)
    slot_weight = float(getattr(self.config, "scalper_max_gross_exposure_pct", CONFIG.scalper_max_gross_exposure_pct) or CONFIG.scalper_max_gross_exposure_pct) / max(
        1.0, float(getattr(self.config, "scalper_max_concurrent", CONFIG.scalper_max_concurrent) or CONFIG.scalper_max_concurrent)
    )
    beta_sum += candidate_beta * slot_weight
    return abs(beta_sum)

def _log_scalper_day(self):
    stats = getattr(self, "_scalper_day_stats", None)
    if not stats:
        return
    trades = int(stats.get("trades", 0) or 0)
    wins = int(stats.get("wins", 0) or 0)
    losses = int(stats.get("losses", 0) or 0)
    win_rate = (wins / trades) if trades > 0 else 0.0
    avg_r = (float(stats.get("sum_r", 0.0) or 0.0) / trades) if trades > 0 else 0.0
    gross_pnl = float(stats.get("gross_pnl", 0.0) or 0.0)
    self.Debug(
        f"SCALPER_DAY trades={trades} wins={wins} losses={losses} "
        f"win_rate={win_rate:.2%} avg_R={avg_r:.2f} gross_pnl={gross_pnl:.2f}%"
    )

def _scalper_on_data(
    self,
    data: Slice,
    *,
    place_entry_fn=place_entry,
    round_quantity_fn=round_quantity,
    place_limit_or_market_fn=place_limit_or_market,
    smart_liquidate_fn=smart_liquidate,
):
    try:
        from scalper import evaluate_entry, evaluate_exit
    except ModuleNotFoundError:  # pragma: no cover
        from .scalper import evaluate_entry, evaluate_exit  # type: ignore
    if not hasattr(self, "_scalper_sleeve_pnls"):
        self._scalper_sleeve_pnls = {"meanrev": deque(maxlen=180), "momentum": deque(maxlen=180)}
    if not hasattr(self, "_scalper_symbol_cooldown_until"):
        self._scalper_symbol_cooldown_until = {}
    if not hasattr(self, "_scalper_entry_sleeve"):
        self._scalper_entry_sleeve = {}
    if not hasattr(self, "_scalper_last_exit_by_sleeve"):
        self._scalper_last_exit_by_sleeve = {}
    if not hasattr(self, "_scalper_day_stats"):
        self._scalper_day_stats = {"trades": 0, "wins": 0, "losses": 0, "sum_r": 0.0, "gross_pnl": 0.0}

    equity_now = float(getattr(self.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
    peak = max(float(getattr(self, "_scalper_equity_peak", equity_now) or equity_now), equity_now)
    self._scalper_equity_peak = peak
    today = self.Time.date()
    if should_auto_reset_latches(
        current_day=today,
        last_reset_day=getattr(self, "_scalper_latch_reset_day", today),
        kill_switch_active=equity_kill_switch_active(equity=equity_now, equity_peak=peak, max_drawdown=0.15),
    ):
        self._scalper_session_brake_until = None
        self._scalper_daily_breaker_until = None
        self._failed_escalations = {}
        self._scalper_symbol_cooldown_until = {}
        self._scalper_latch_reset_day = today
        self.Debug("SCALPER_LATCH_RESET day_roll=true")

    if self._scalper_daily_anchor_date != today:
        if self._scalper_daily_anchor_date is not None:
            self._log_scalper_day()
        self._scalper_day_stats = {"trades": 0, "wins": 0, "losses": 0, "sum_r": 0.0, "gross_pnl": 0.0}
        self._scalper_daily_anchor_date = today
        self._scalper_daily_anchor_equity = float(self.Portfolio.TotalPortfolioValue or 0.0)
        self._scalper_daily_pnl = 0.0
        decayed = {}
        for sym, n in self._scalper_consec_losses.items():
            last_t = self._scalper_last_trade_time.get(sym)
            hours_since = ((self.Time - last_t).total_seconds() / 3600.0) if last_t else INFINITE_HELD_HOURS
            if hours_since >= 24.0:
                new_n = max(0, int(n) - 1)
                if new_n > 0:
                    decayed[sym] = new_n
            else:
                decayed[sym] = n
        if self._scalper_consec_losses != decayed:
            readable_before = {getattr(s, "Value", str(s)): v for s, v in dict(self._scalper_consec_losses).items()}
            readable_after = {getattr(s, "Value", str(s)): v for s, v in decayed.items()}
            self.Debug(f"SCALPER_DECAY consec_losses_before={readable_before} after={readable_after}")
        self._scalper_consec_losses = decayed
        self._scalper_symbol_cooldown_until = {
            s: t for s, t in self._scalper_symbol_cooldown_until.items() if t is not None and self.Time < t
        }
    equity = float(self.Portfolio.TotalPortfolioValue or 0.0)
    anchor = self._scalper_daily_anchor_equity or equity
    self._scalper_daily_pnl = (equity / anchor - 1.0) if anchor > 0 else 0.0
    held = self._current_holdings()
    cash = free_cash_usd(self)
    alloc = self._scalper_sleeve_allocations()
    brake_until = getattr(self, "_scalper_session_brake_until", None)
    brake_active = brake_until is not None and self.Time < brake_until
    breaker_state = getattr(self, "_breaker_liquidated", False)
    failed = getattr(self, "_failed_escalations", {}) or {}
    cooldown_hours = float(getattr(self.config, "failed_esc_cooldown_hours", 6.0) or 6.0)
    failed_esc_active = 0
    for last_fail in failed.values():
        try:
            failed_esc_active += int((self.Time - last_fail).total_seconds() < cooldown_hours * 3600.0)
        except Exception:
            pass
    cash_pct = (cash / equity * 100.0) if equity > 0 else 0.0
    size_pct_eff = effective_position_size_pct((cash / equity) if equity > 0 else 0.0, config=self.config)
    self.Debug(
        f"SCALPER_HB t={self.Time} held={len(held)}/{int(self.config.scalper_max_concurrent)} "
        f"cash={cash:.2f} daily_pnl={self._scalper_daily_pnl*100:.2f}% "
        f"brake_active={brake_active} breaker={breaker_state} "
        f"consec_losses_syms={len(self._scalper_consec_losses)} "
        f"failed_esc_active={failed_esc_active} "
        f"cash_pct={cash_pct:.1f} size_pct_eff={size_pct_eff*100.0:.1f} "
        f"eq_peak={peak:.2f} alloc_mr={alloc['meanrev']:.2f} alloc_mom={alloc['momentum']:.2f}"
    )

    btc_sym = self.symbol_by_ticker.get("BTCUSD")
    btc_feats = self.feature_engine.current_features("BTCUSD") if btc_sym is not None else {}
    btc_ret_1h = float(btc_feats.get("ret_1h", 0.0) or 0.0)
    btc_ret_6h = float(btc_feats.get("ret_6h", 0.0) or 0.0)

    for sym in list(self._current_holdings()):
        feats = self.feature_engine.current_features(getattr(sym, "Value", str(sym))) or {}
        state = self.position_state.get(sym)
        if state is None:
            avg_px = float(self.Portfolio[sym].AveragePrice or 0.0)
            if avg_px <= 0:
                sec = self.Securities.get(sym)
                avg_px = float(getattr(sec, "Price", 0.0) or 0.0)
            if avg_px <= 0:
                continue
            atr = float(feats.get("atr", 0.0) or 0.0)
            if atr <= 0:
                atr = avg_px * 0.05
            state = PositionState(
                entry_price=avg_px,
                highest_close=avg_px,
                entry_atr=atr,
                entry_time=self.Time - timedelta(hours=1),
                strategy_owner="scalper",
                initial_risk_distance=max(float(getattr(self.config, "scalper_stop_atr_mult", 1.5) or 1.5) * atr, 1e-9),
                stop_price=max(0.0, avg_px - max(float(getattr(self.config, "scalper_stop_atr_mult", 1.5) or 1.5) * atr, 1e-9)),
                # scalper_tp1_atr is a legacy fallback when scalper_tp_atr_mult is absent.
                take_profit_price=avg_px
                + atr
                * float(
                    getattr(
                        self.config,
                        "scalper_tp_atr_mult",
                        getattr(self.config, "scalper_tp1_atr", 2.0),
                    )
                    or 2.0
                ),
                partial_tp_price=avg_px + atr * float(getattr(self.config, "scalper_partial_tp_atr_mult", 1.0) or 1.0),
                trail_anchor_price=avg_px,
            )
            self.position_state[sym] = state
            if not hasattr(self, "_lazy_seed_warned"):
                self._lazy_seed_warned = set()
            if sym not in self._lazy_seed_warned:
                self.Debug(
                    f"WARN LAZY_SEED sym={getattr(sym,'Value',sym)} avg_px={avg_px:.6f} atr={atr:.6f} reason=missing_state"
                )
                self._lazy_seed_warned.add(sym)
        sec = self.Securities.get(sym)
        px = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
        prev_high = float(getattr(state, "highest_close", state.entry_price) or state.entry_price)
        prev_anchor = getattr(state, "trail_anchor_price", None)
        if prev_anchor is None:
            prev_anchor = prev_high
        trail_anchor = max(float(prev_anchor), px)
        state.trail_anchor_price = trail_anchor
        state.highest_close = max(prev_high, trail_anchor)
        qty_now = float(getattr(self.Portfolio[sym], "Quantity", 0.0) or 0.0)
        side = -1 if qty_now < 0 else 1
        sleeve = "momentum" if "momentum" in str(getattr(state, "strategy_owner", "meanrev")) else "meanrev"
        if sleeve == "momentum" and side < 0:
            sleeve = "momentum_short"
        should_exit, tag = evaluate_exit(
            symbol=sym,
            feats=feats,
            entry_price=state.entry_price,
            entry_time=state.entry_time,
            current_time=self.Time,
            current_price=px,
            btc_ret_1h=btc_ret_1h,
            highest_close=state.highest_close,
            entry_atr=state.entry_atr,
            initial_risk_distance=getattr(state, "initial_risk_distance", 0.0),
            stop_price=getattr(state, "stop_price", None),
            take_profit_price=getattr(state, "take_profit_price", None),
            partial_tp_price=getattr(state, "partial_tp_price", None),
            trail_anchor_price=getattr(state, "trail_anchor_price", None),
            partial_tp_done=bool(getattr(state, "partial_tp_done", False)),
            tight_trail_armed=bool(getattr(state, "tight_trail_armed", False)),
            sleeve=sleeve,
            position_side=side,
            config=self.config,
        )
        if tag == "TightTrailArmed":
            state.tight_trail_armed = True
            continue
        pnl_frac = (side * (px / state.entry_price - 1.0)) if state.entry_price > 0 else 0.0
        if should_exit and tag == "TP1":
            qty_half = round_quantity_fn(
                self,
                sym,
                abs(qty_now) * float(getattr(self.config, "scalper_tp1_partial_pct", CONFIG.scalper_tp1_partial_pct) or CONFIG.scalper_tp1_partial_pct),
            )
            if qty_half > 0:
                ticket = place_limit_or_market_fn(self, sym, (-qty_half if side > 0 else qty_half), tag="TP1")
                if ticket is not None:
                    state.partial_tp_done = True
                    bars_held = int(
                        max(
                            0.0,
                            ((self.Time - state.entry_time).total_seconds() / 3600.0) if state.entry_time else 0.0,
                        )
                    )
                    self.Debug(f"SCALPER_EXIT sym={sym.Value} tag=TP1 pnl={pnl_frac*100:.2f}% bars_held={bars_held} r_multiple=1.00")
            continue
        if should_exit:
            risk_dist = max(float(getattr(state, "initial_risk_distance", 0.0) or 0.0), 1e-9)
            r_multiple = (side * (px - state.entry_price)) / risk_dist if state.entry_price > 0 else 0.0
            bars_held = int(
                max(
                    0.0,
                    ((self.Time - state.entry_time).total_seconds() / 3600.0) if state.entry_time else 0.0,
                )
            )
            if smart_liquidate_fn(self, sym, tag=tag):
                self.Debug(
                    f"SCALPER_EXIT sym={sym.Value} tag={tag} pnl={pnl_frac*100:.2f}% "
                    f"bars_held={bars_held} r_multiple={r_multiple:.2f}"
                )
                self.position_state.pop(sym, None)
                self._scalper_entry_sleeve.pop(sym, None)
                self._scalper_sleeve_pnls.setdefault(sleeve, deque(maxlen=180)).append(pnl_frac)
                self._scalper_recent_pnls.append(pnl_frac)
                self._scalper_last_exit_by_sleeve[(sym, sleeve)] = self.Time
                stats = self._scalper_day_stats
                stats["trades"] = int(stats.get("trades", 0) or 0) + 1
                stats["wins"] = int(stats.get("wins", 0) or 0) + (1 if pnl_frac > 0 else 0)
                stats["losses"] = int(stats.get("losses", 0) or 0) + (1 if pnl_frac <= 0 else 0)
                stats["sum_r"] = float(stats.get("sum_r", 0.0) or 0.0) + float(r_multiple)
                stats["gross_pnl"] = float(stats.get("gross_pnl", 0.0) or 0.0) + float(pnl_frac * 100.0)
                if len(self._scalper_recent_pnls) >= 3 and all(p < 0 for p in list(self._scalper_recent_pnls)[-3:]):
                    self._scalper_session_brake_until = self.Time + timedelta(hours=6)
                    self.Debug(f"SCALPER_BRAKE session=3consec_losses until={self._scalper_session_brake_until}")
                if pnl_frac < 0:
                    streak = self._scalper_consec_losses.get(sym, 0) + 1
                    self._scalper_consec_losses[sym] = streak
                    until = self.Time + timedelta(hours=float(getattr(self.config, "scalper_loss_cooldown_hours", 6.0) or 6.0))
                    self._scalper_symbol_cooldown_until[sym] = until
                    self.Debug(f"COOLDOWN_BLOCK sym={sym.Value} until={until} streak={streak}")
                else:
                    self._scalper_consec_losses[sym] = 0
                self._scalper_last_trade_time[sym] = self.Time

    if self._scalper_session_brake_until is not None and self.Time < self._scalper_session_brake_until:
        return
    elif self._scalper_session_brake_until is not None:
        self._scalper_session_brake_until = None

    held = self._current_holdings()
    breaker_until = getattr(self, "_scalper_daily_breaker_until", None)
    if breaker_until is not None and self.Time < breaker_until:
        return
    if self._scalper_daily_pnl <= float(getattr(self.config, "scalper_daily_loss_brake", -0.01) or -0.01):
        if breaker_until is None or self.Time >= breaker_until:
            next_day = (self.Time + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            self._scalper_daily_breaker_until = next_day
            self.Debug(f"BREAKER_DAILY pnl={self._scalper_daily_pnl*100:.2f}% until={next_day}")
        for sym in held:
            smart_liquidate_fn(self, sym, tag="BREAKER_DAILY")
        return

    if len(held) >= int(self.config.scalper_max_concurrent):
        return

    available_cash = free_cash_usd(self)
    available_cash_pct = (available_cash / equity) if equity > 0 else 0.0

    reg_engine = getattr(self, "regime_engine", None)
    btc_gate_open = reg_engine.btc_above_ema30d() if reg_engine is not None else True
    if bool(getattr(self.config, "scalper_use_btc_ema_gate", False)) and not btc_gate_open:
        return

    candidates = {"meanrev": [], "momentum": [], "momentum_short": []}
    rejection_counts = {}
    for symbol in self.symbols[: int(self.config.scalper_universe_size)]:
        feats = dict(self.feature_engine.current_features(symbol.Value) or {})
        vol_cone = getattr(getattr(self, "signal_features", None), "vol_cone", None)
        feats["vol_cone_pct"] = float(vol_cone.percentile_rank(symbol)) if vol_cone is not None else 1.0
        if not feats:
            reason = "missing_feats"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        cool_until = self._scalper_symbol_cooldown_until.get(symbol)
        if cool_until is not None and self.Time < cool_until:
            rejection_counts["COOLDOWN_BLOCK"] = rejection_counts.get("COOLDOWN_BLOCK", 0) + 1
            continue
        beta_cap = float(getattr(self.config, "scalper_beta_cap", 1.5) or 1.5)
        beta_sum = self._portfolio_beta_sum_with_candidate(symbol, held, btc_sym)
        if beta_sum > beta_cap:
            self.Debug(f"BETA_BLOCK sym={symbol.Value} beta_sum={beta_sum:.2f} cap={beta_cap:.2f}")
            rejection_counts["BETA_BLOCK"] = rejection_counts.get("BETA_BLOCK", 0) + 1
            continue
        last_t = self._scalper_last_trade_time.get(symbol)
        last_trade_hours_ago = (
            (self.Time - last_t).total_seconds() / 3600.0 if last_t else INFINITE_HELD_HOURS
        )
        for sleeve in ("meanrev", "momentum", "momentum_short"):
            last_exit = self._scalper_last_exit_by_sleeve.get((symbol, sleeve))
            if last_exit is not None:
                since_exit = (self.Time - last_exit).total_seconds() / 3600.0
                if since_exit < float(getattr(self.config, "scalper_anti_churn_hours", 2.0) or 2.0):
                    rejection_counts["ANTI_CHURN"] = rejection_counts.get("ANTI_CHURN", 0) + 1
                    continue
            ok, reason = evaluate_entry(
                symbol=symbol,
                feats=feats,
                btc_ret_1h=btc_ret_1h,
                btc_ret_6h=btc_ret_6h,
                has_position=(symbol in held),
                last_trade_hours_ago=last_trade_hours_ago,
                available_cash_pct=available_cash_pct,
                daily_pnl_pct=self._scalper_daily_pnl,
                consecutive_losses_for_symbol=self._scalper_consec_losses.get(symbol, 0),
                equity=equity,
                sleeve=sleeve,
                config=self.config,
            )
            if ok:
                z = float(feats.get("z_20h", 0.0) or 0.0)
                regime = regime_for(feats, config=self.config)
                candidates[sleeve].append((symbol, z, regime))
            else:
                if reason.startswith("REGIME_BLOCK"):
                    self.Debug(f"REGIME_BLOCK sym={symbol.Value} sleeve={sleeve} reason={reason}")
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    if rejection_counts:
        top = sorted(rejection_counts.items(), key=lambda x: -x[1])[:5]
        rej_distinct = len(rejection_counts)
        if getattr(self, "_scalper_rej_distinct_prev", None) != rej_distinct:
            self.Debug(f"SCALPER_REJ top={top}")
            self._scalper_rej_distinct_prev = rej_distinct

    slots_remaining = int(self.config.scalper_max_concurrent) - len(held)
    if slots_remaining <= 0:
        return
    alloc = self._scalper_sleeve_allocations()
    by_sleeve_slots = {
        "meanrev": max(0, int(round(slots_remaining * alloc["meanrev"]))),
        "momentum": max(0, int(round(slots_remaining * alloc["momentum"]))),
        "momentum_short": 0,
    }
    by_sleeve_slots["momentum_short"] = max(0, by_sleeve_slots["momentum"] // 2)
    if by_sleeve_slots["meanrev"] + by_sleeve_slots["momentum"] + by_sleeve_slots["momentum_short"] < slots_remaining:
        by_sleeve_slots["meanrev"] += slots_remaining - (
            by_sleeve_slots["meanrev"] + by_sleeve_slots["momentum"] + by_sleeve_slots["momentum_short"]
        )
    ordered = {
        "meanrev": sorted(candidates["meanrev"], key=lambda x: x[1]),
        "momentum": sorted(candidates["momentum"], key=lambda x: x[1], reverse=True),
        "momentum_short": sorted(candidates["momentum_short"], key=lambda x: x[1]),
    }
    placed_symbols = set()
    gross_now = float(getattr(self.Portfolio, "TotalHoldingsValue", 0.0) or 0.0) / max(equity, 1.0)
    for sleeve in ("meanrev", "momentum", "momentum_short"):
        for symbol, z, regime in ordered[sleeve]:
            if by_sleeve_slots[sleeve] <= 0 or len(placed_symbols) >= slots_remaining:
                break
            if symbol in placed_symbols:
                continue
            sec = self.Securities.get(symbol)
            price = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
            if price <= 0:
                continue
            feats = dict(self.feature_engine.current_features(symbol.Value) or {})
            vol_cone = getattr(getattr(self, "signal_features", None), "vol_cone", None)
            feats["vol_cone_pct"] = float(vol_cone.percentile_rank(symbol)) if vol_cone is not None else 1.0
            atr_pct = float(feats.get("atr_pct", 0.0) or 0.0)
            size_pct_eff = effective_position_size_pct((available_cash / equity) if equity > 0 else 0.0, config=self.config)
            qty_raw, notional = vol_target_qty(
                equity=equity,
                price=price,
                atr_pct=atr_pct,
                available_cash=available_cash,
                current_gross_exposure_pct=gross_now,
                risk_per_trade_pct=float(getattr(self.config, "scalper_risk_per_trade_pct", 0.005) or 0.005),
                max_symbol_exposure_pct=float(getattr(self.config, "scalper_max_symbol_exposure_pct", 0.25) or 0.25),
                max_gross_exposure_pct=float(getattr(self.config, "scalper_max_gross_exposure_pct", 0.95) or 0.95),
                position_size_pct=size_pct_eff,
                atr_stop_mult=float(getattr(self.config, "scalper_stop_atr_mult", 1.5) or 1.5),
                max_concurrent_positions=int(getattr(self.config, "scalper_max_concurrent", 4) or 4),
            )
            qty = round_quantity_fn(self, symbol, qty_raw)
            if qty <= 0 or notional < float(getattr(self.config, "min_notional_usd", 5.0)):
                continue
            order_qty = -qty if sleeve == "momentum_short" else qty
            ok, required, afford = can_afford(self, symbol, abs(order_qty), price)
            if not ok:
                self.Debug(f"INSUFF_FUNDS sym={symbol.Value} req={required:.2f} avail={afford:.2f} tag=Scalper:{sleeve}")
                continue
            tag = "ScalperMomShort:entry" if sleeve == "momentum_short" else ("ScalperMom:entry" if sleeve == "momentum" else "Scalper:entry")
            ticket = (
                place_limit_or_market_fn(self, symbol, order_qty, tag=tag, signal_score=float(abs(z)))
                if sleeve == "momentum_short"
                else place_entry_fn(self, symbol, qty, tag=tag, signal_score=float(abs(z)))
            )
            if ticket is not None:
                atr_abs = float(feats.get("atr", 0.0) or 0.0)
                if atr_abs <= 0:
                    atr_abs = max(atr_pct * price, price * 0.01)
                risk_dist = max(float(getattr(self.config, "scalper_stop_atr_mult", 1.5) or 1.5) * atr_abs, 1e-9)
                if sleeve == "momentum_short":
                    stop_px = price + risk_dist
                    tp1_px = price - atr_abs * float(getattr(self.config, "scalper_tp1_atr", 1.5) or 1.5)
                    tp2_px = price - risk_dist * float(getattr(self.config, "scalper_tp2_r", 2.5) or 2.5)
                else:
                    stop_px = max(0.0, price - risk_dist)
                    tp1_px = price + atr_abs * float(getattr(self.config, "scalper_tp1_atr", 1.5) or 1.5)
                    tp2_px = price + risk_dist * float(getattr(self.config, "scalper_tp2_r", 2.5) or 2.5)
                risk_dollars = qty * risk_dist
                if sleeve != "momentum_short":
                    available_cash -= qty * price
                gross_now += (qty * price) / max(equity, 1.0)
                placed_symbols.add(symbol)
                by_sleeve_slots[sleeve] -= 1
                self._scalper_entry_sleeve[symbol] = sleeve
                self.Debug(
                    f"SCALPER_ENTRY sym={symbol.Value} sleeve={sleeve} regime={regime} "
                    f"z={z:.2f} qty={qty:.6f} px={price:.4f} stop={stop_px:.4f} "
                    f"tp1={tp1_px:.4f} tp2={tp2_px:.4f} risk_${risk_dollars:.2f}"
                )
