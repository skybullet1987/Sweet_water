from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from datetime import timedelta

import pandas as pd

try:  # pragma: no cover
    from AlgorithmImports import AccountType, BrokerageName, Market, QCAlgorithm, Resolution, Slice
    from QuantConnect.Orders.Slippage import ConstantSlippageModel  # type: ignore
except ImportError:  # pragma: no cover
    LEAN_IMPORT_ERROR = RuntimeError("AlgorithmImports is required when running in QuantConnect LEAN.")

    class QCAlgorithm:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise LEAN_IMPORT_ERROR

    class Resolution:
        Hour = "Hour"
        Minute = "Minute"
        Tick = "Tick"

    class BrokerageName:
        Kraken = "Kraken"

    class AccountType:
        Cash = "Cash"

    class Market:
        Kraken = "Kraken"

    class Slice:  # type: ignore
        pass

    class ConstantSlippageModel:  # type: ignore
        def __init__(self, value: float):
            self.value = float(value)

from config import CONFIG
from execution import (
    KrakenTieredFeeModel,
    PositionState,
    RealisticCryptoSlippage,
    debug_limited,
    execute_regime_entries,
    escalate_stale_orders,
    get_min_notional_usd,
    is_invested_not_dust,
    liquidate_all_positions,
    manage_open_positions,
    place_entry,
    place_limit_or_market,
    position_status,
    round_quantity,
    smart_liquidate,
)
from features import FeatureEngine, SignalFeatureStack
from regime import RegimeEngine
from reporting import Reporter
from risk import DrawdownCircuitBreaker, RiskManager
from scoring import Scorer
from sizing import Sizer
from universe import KRAKEN_SAFE_LIST, REFERENCE_SYMBOLS, select_universe

INFINITE_HELD_HOURS = 10**9
DEFAULT_MISSING_SCORE = -1e9
HARD_RISK_OFF_VOL_STRESS = 0.9
NEW_ENTRANT_MIN_DELTA_MULTIPLIER = 0.5
NO_TRADE_HEARTBEAT_THRESHOLD_BARS = 168
NO_TRADE_HEARTBEAT_LOG_CADENCE_BARS = 24
PRIME_MIN_READY_SYMBOLS = 5
NO_CHASE_MOM24_THRESHOLD = 0.15
MIN_VOLUME_RATIO_24H_7D = 1.2


class SweetWaterPhase1(QCAlgorithm):
    def Initialize(self):  # pragma: no cover
        self.config = CONFIG
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
        self.long_only = True
        self.SetCash(self.config.starting_cash)
        self.SetStartDate(self.config.start_year, self.config.start_month, self.config.start_day)
        self.SetEndDate(self.config.end_year, self.config.end_month, self.config.end_day)
        self.SetWarmup(self.config.warmup_bars, Resolution.Hour)

        self.feature_engine = FeatureEngine(signal_mode=getattr(self.config, "signal_mode", "cross_sectional_momentum"))
        self.regime_engine = RegimeEngine(self.config)
        self.scorer = Scorer(self.config)
        self.signal_features = SignalFeatureStack(self)
        self.sizer = Sizer(self.config)
        self.risk = RiskManager(self.config)
        self.reporter = Reporter(self.config)

        self.max_participation_rate = 0.15
        self.spread_limit_pct = 0.025

        self._drawdown_breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.12, max_triggered_bars=168)

        self.symbols = []
        self.reference_symbols = []
        self.symbol_by_ticker = {}
        self.open_snapshots = defaultdict(dict)
        self.trade_log = deque(maxlen=500)
        self.crypto_data = {}
        self._last_rebalance_month = None
        self._init_execution_attrs()

        for ticker in KRAKEN_SAFE_LIST:
            self._subscribe_symbol(ticker)
        for ticker in REFERENCE_SYMBOLS:
            self._subscribe_symbol(ticker)
        self.reference_symbols = [self.symbol_by_ticker[t] for t in REFERENCE_SYMBOLS if t in self.symbol_by_ticker]

        initial_universe = select_universe(self._history_provider, pd.Timestamp.now(tz="UTC"))
        self.symbols = [self.symbol_by_ticker[t] for t in initial_universe if t in self.symbol_by_ticker]
        self.signal_features.set_tracked_symbols(
            [s.Value for s in [*self.reference_symbols, *self.symbols] if hasattr(s, "Value")]
        )
        self._prime_features_from_history()
        now = self.Time if hasattr(self, "Time") else pd.Timestamp.now(tz="UTC")
        self._last_rebalance_month = (int(now.year), int(now.month))
        status = self.signal_features.init_status()
        self.Debug(
            "SIG init "
            + " ".join([f"{k}={v}" for k, v in status.items()])
            + f" mode={getattr(self.config, 'signal_mode', 'cross_sectional_momentum')}"
        )

    def _subscribe_symbol(self, ticker: str):  # pragma: no cover
        if ticker in self.symbol_by_ticker:
            return self.symbol_by_ticker[ticker]
        sec_obj = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
        if (
            getattr(self.config, "signal_mode", "cross_sectional_momentum") == "microstructure"
            and bool(getattr(self, "LiveMode", False))
        ):
            try:
                tick_obj = self.AddCrypto(ticker, Resolution.Tick, Market.Kraken)
                self._configure_security_models(tick_obj)
            except Exception as exc:
                self.Debug(f"SIG tick_subscribe_fallback symbol={ticker} err={type(exc).__name__}:{exc!r}")
        self._configure_security_models(sec_obj)
        self.symbol_by_ticker[ticker] = sec_obj.Symbol
        return sec_obj.Symbol

    def _configure_security_models(self, security):  # pragma: no cover
        fee = KrakenTieredFeeModel()
        slippage = RealisticCryptoSlippage(self)
        try:
            security.SetFeeModel(fee)
        except Exception:
            security.FeeModel = fee
        try:
            security.SetSlippageModel(slippage)
        except Exception:
            try:
                security.set_slippage_model(slippage)
            except Exception:
                security.SetSlippageModel(ConstantSlippageModel(0.0005))

    def _init_execution_attrs(self):  # pragma: no cover
        self.min_notional = 1.0
        self.volatility_regime = "normal"
        self.market_regime = "trending"
        self.log_budget = 200
        self.expected_round_trip_fees = 0.0065
        self._cancel_cooldowns = {}
        self._pending_orders = {}
        self._submitted_orders = {}
        self._failed_escalations = {}
        self._order_retries = {}
        self._failed_exit_counts = {}
        self._partial_sell_symbols = set()
        self._session_blacklist = set()
        self._rolling_wins = deque(maxlen=200)
        self._rolling_win_sizes = deque(maxlen=200)
        self._rolling_loss_sizes = deque(maxlen=200)
        self._recent_trade_outcomes = deque(maxlen=20)
        self._entry_signal_combos = {}
        self._last_cash_gate_log = None
        self._bar_count = 0
        self._last_trade_bar = 0
        self._last_no_trade_log_bar = 0
        self._last_sig_hold_log_bar = {}
        self._last_daily_summary_date = None
        self._last_rebalance_time = None
        self._last_regime_state = None
        self._breaker_liquidated = False
        self._abandoned_dust = set()
        self._dispersion_history = deque(maxlen=60)
        self._last_dispersion_date = None
        self._last_dispersion_log_date = None
        self._pending_rotation_entries = []
        self._pending_rotation_entry_time = None
        self._breaker_disengaged_at = None
        self._last_scored = []
        self._last_force_exit_date = None
        self._last_btc_gate_log_date = None
        self._scalper_last_trade_time = {}
        self._scalper_consec_losses = {}
        self._scalper_daily_pnl = 0.0
        self._scalper_daily_anchor_equity = 0.0
        self._scalper_daily_anchor_date = None
        self._scalper_session_brake_until = None
        self._scalper_recent_pnls = deque(maxlen=6)

        self.position_state: dict[object, PositionState] = {}
        self.entry_volumes = {}
        self.peak_value = None
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.daily_trade_count = 0
        self.trade_count = 0
        self.pnl_by_tag = {}
        self.pnl_by_regime = {}
        self.stale_limit_escalations = 0
        self.stale_limit_escalation_fills = 0
        self.stale_limit_cancels = 0

    def _history_provider(self, symbol: str, start, end):  # pragma: no cover
        sec = self.symbol_by_ticker.get(symbol)
        if sec is None:
            return pd.DataFrame()
        hist = self.History(sec, start, end, Resolution.Hour)
        if hist is None or hist.empty:
            return pd.DataFrame()
        return hist[["open", "high", "low", "close", "volume"]]

    def _prime_features_from_history(self):
        class _PrimeBar:
            __slots__ = ("Close",)

            def __init__(self, c):
                self.Close = c

        prime_days = 140
        end = self.Time if hasattr(self, "Time") else pd.Timestamp.now(tz="UTC")
        start = end - timedelta(days=prime_days)
        primed = 0
        btc_sym = self.symbol_by_ticker.get("BTCUSD")
        for ticker, sym in list(self.symbol_by_ticker.items()):
            try:
                hist = self.History(sym, start, end, Resolution.Hour)
            except Exception:
                continue
            if hist is None or hist.empty:
                continue
            for ts, row in hist.iterrows():
                ts_clean = ts[1] if isinstance(ts, tuple) else ts
                close = float(row["close"])
                self.feature_engine.update(
                    {
                        "symbol": ticker,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": close,
                        "volume": float(row["volume"]),
                        "time": ts_clean,
                    }
                )
                bar_proxy = _PrimeBar(close)
                self.regime_engine.hurst.update(sym, bar_proxy)
                self.regime_engine.vr.update(sym, bar_proxy)
                if btc_sym is not None and sym == btc_sym and close > 0:
                    self.regime_engine.update_btc_close(close)
            primed += 1
        ready = 0
        symbols = list(getattr(self, "symbols", []))
        for sym in symbols:
            feats = self.feature_engine.current_features(sym.Value) or {}
            if abs(float(feats.get("mom_90d", 0.0) or 0.0)) > 1e-9:
                ready += 1
        if ready < PRIME_MIN_READY_SYMBOLS:
            self.Debug(f"CRITICAL prime_features mom_90d_ready={ready}/{len(symbols)} -- priming may have failed!")
        else:
            self.Debug(f"PRIME features symbols={primed} mom_90d_ready={ready}")
        z_ready = 0
        n_total = len(symbols)
        f_state = getattr(self.feature_engine, "_state", {})
        for sym in symbols:
            state = f_state.get(getattr(sym, "Value", str(sym)), {})
            closes = state.get("close_history_24h", [])
            if len(closes) >= 20:
                z_ready += 1
        self.Debug(f"PRIME scalper z_ready={z_ready}/{n_total}")
        disp_ready = len(getattr(self, "_dispersion_history", []))
        if disp_ready < 20:
            self.Debug(f"CRITICAL prime_features dispersion_ready={disp_ready} (<20) by Jan 1")

    def _ensure_monthly_universe(self):  # pragma: no cover
        current_month = (int(self.Time.year), int(self.Time.month))
        if current_month == self._last_rebalance_month:
            return
        for ticker in KRAKEN_SAFE_LIST:
            self._subscribe_symbol(ticker)
        self.symbols = [self.symbol_by_ticker[t] for t in select_universe(self._history_provider, self.Time) if t in self.symbol_by_ticker]
        self.signal_features.set_tracked_symbols(
            [s.Value for s in [*self.reference_symbols, *self.symbols] if hasattr(s, "Value")]
        )
        self._last_rebalance_month = current_month

    def _update_symbol_series(self, symbol, bar):
        state = self.crypto_data.setdefault(symbol, {"prices": deque(maxlen=96), "volume": deque(maxlen=96)})
        state["prices"].append(float(bar.Close))
        state["volume"].append(float(bar.Volume))

    def _portfolio_state(self):
        equity = float(self.Portfolio.TotalPortfolioValue)
        gross = float(self.Portfolio.TotalHoldingsValue) / max(equity, 1.0)
        return equity, gross

    def _collect_scores(self, regime_state: str, btc_ret: float):
        rows = []
        for symbol in self.symbols:
            feats = self.feature_engine.current_features(symbol.Value)
            if not feats:
                continue
            rv = max(float(feats.get("rv_21d", 0.0) or 0.0), 1e-4)
            m21 = float(feats.get("mom_21d_skip", 0.0) or 0.0) / rv
            m63 = float(feats.get("mom_63d_skip", 0.0) or 0.0) / rv
            m90 = float(feats.get("mom_90d_skip", 0.0) or 0.0) / rv
            feats["hurst"] = self.regime_engine.hurst.hurst(symbol)
            feats["dhurst_30d"] = self.regime_engine.hurst.hurst_change_30d(symbol)
            feats["vr_regime"] = self.regime_engine.vr.regime(symbol)
            rows.append((symbol, feats, m21, m63, m90))
        if len(rows) < 4:
            self._last_scored = []
            return []

        def _z(values):
            mu = statistics.fmean(values)
            sd = statistics.pstdev(values) if len(values) > 1 else 0.0
            sd = max(sd, 1e-6)
            return [(v - mu) / sd for v in values]

        m21_vals = [r[2] for r in rows]
        z21 = _z(m21_vals)
        z63 = _z([r[3] for r in rows])
        z90 = _z([r[4] for r in rows])
        if len(m21_vals) > 1:
            disp = statistics.pstdev(m21_vals)
            day_key = self.Time.date() if hasattr(self, "Time") else None
            if day_key is not None and day_key != self._last_dispersion_date:
                self._dispersion_history.append(float(disp))
                self._last_dispersion_date = day_key
        if getattr(self, "_bar_count", 0) % 24 == 0 and rows:
            avg_hurst = statistics.fmean([float(r[1].get("hurst", 0.5) or 0.5) for r in rows])
            self.Debug(f"HURST avg={avg_hurst:.4f} n={len(rows)}")
        scored = []
        for i, (symbol, feats, *_rest) in enumerate(rows):
            z_avg = (z21[i] + z63[i] + z90[i]) / 3.0
            snap = self.scorer.score(
                symbol=symbol,
                features=feats,
                regime_state=regime_state,
                btc_context={"btc_trend": btc_ret},
                signal_stack=self.signal_features,
                regime_engine=self.regime_engine,
                cross_section_zscore=z_avg,
            )
            scored.append((symbol, float(snap.get("final", 0.0)), feats))
        scored.sort(key=lambda x: x[1], reverse=True)
        self._last_scored = list(scored)
        return scored

    def _force_exit_losers(self, scored):
        loser_floor_z = -0.5
        exited = set()
        holdings = self._current_holdings()
        for sym in list(holdings):
            score = next((s for sy, s, _ in scored if sy == sym), 0.0)
            if score < loser_floor_z:
                if smart_liquidate(self, sym, tag="ZScoreLoser"):
                    self.Debug(f"FORCE_EXIT sym={sym.Value} z={score:.3f}")
                    exited.add(sym)
        return exited

    def _process_pending_entries(self, scored_lookup=None):
        _ = scored_lookup
        if not getattr(self, "_pending_rotation_entries", None):
            return
        pending_at = getattr(self, "_pending_rotation_entry_time", None)
        if pending_at is None:
            self._pending_rotation_entries = []
            return
        if (self.Time - pending_at).total_seconds() < 3600:
            return
        try:
            available_cash = float(self.Portfolio.CashBook["USD"].Amount)
        except Exception:
            available_cash = float(getattr(self.Portfolio, "Cash", 0.0) or 0.0)
        equity = float(getattr(self.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        submitted = 0
        for entry in self._pending_rotation_entries:
            sym = entry["symbol"]
            if available_cash < float(self.config.min_position_floor_usd):
                break
            sec = self.Securities.get(sym)
            price = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
            if price <= 0:
                continue
            feat_engine = getattr(self, "feature_engine", None)
            feats = feat_engine.current_features(getattr(sym, "Value", str(sym))) if feat_engine is not None else {}
            feats = feats or {}
            mom_24 = float(feats.get("mom_24", 0.0) or 0.0)
            if mom_24 > NO_CHASE_MOM24_THRESHOLD:
                self.Debug(f"NO_CHASE sym={getattr(sym,'Value',sym)} mom_24={mom_24:.3f}")
                continue
            vol_ratio = float(feats.get("vol_ratio_24h_7d", 1.0) or 1.0)
            if vol_ratio < MIN_VOLUME_RATIO_24H_7D:
                self.Debug(f"NO_VOLUME sym={getattr(sym,'Value',sym)} vol_ratio={vol_ratio:.2f}")
                continue
            desired = equity * float(entry["target_weight"])
            notional = min(desired, available_cash * 0.95)
            if notional < float(self.config.min_position_floor_usd):
                continue
            qty = round_quantity(self, sym, notional / max(price, 1e-9))
            if qty <= 0:
                continue
            ticket = place_entry(self, sym, qty, tag="Rebalance:entry-deferred", signal_score=float(entry["score"]))
            if ticket is not None:
                available_cash -= qty * price
                submitted += 1
        self.Debug(f"REBAL deferred_entries submitted={submitted} of {len(self._pending_rotation_entries)}")
        self._pending_rotation_entries = []
        self._pending_rotation_entry_time = None

    def _dispersion_regime(self):
        history = getattr(self, "_dispersion_history", [])
        if len(history) < 20:
            return "full"
        sorted_disp = sorted(history)
        p15 = sorted_disp[max(0, int(0.15 * len(sorted_disp)) - 1)]
        p30 = sorted_disp[max(0, int(0.30 * len(sorted_disp)) - 1)]
        current = history[-1]
        if current < p15:
            return "flat"
        if current < p30:
            return "half"
        return "full"

    def _current_holdings(self):
        out = []
        for kv in self.Portfolio:
            if hasattr(kv, "Key"):
                sym = kv.Key
                holding = getattr(kv, "Value", None)
            else:
                sym = kv
                holding = None
            try:
                if holding is None or not hasattr(holding, "Quantity"):
                    holding = self.Portfolio[sym]
                qty = float(getattr(holding, "Quantity", 0.0) or 0.0)
            except Exception:
                qty = 0.0
            if qty > 0:
                out.append(sym)
        return out

    def _market_regime_state(self, btc_ret: float, breadth: float):
        btc = self.symbol_by_ticker.get("BTCUSD")
        feats = self.feature_engine.current_features("BTCUSD") if btc is not None else {}
        btc_mom_90d = float(feats.get("mom_90d", 0.0) or 0.0)
        btc_vol_stress = float(feats.get("vol_stress_21d", 1.0) or 1.0)
        risk_on = btc_mom_90d > 0.0 or btc_vol_stress < float(self.config.vol_stress_threshold)
        self.regime_engine.update(btc_ret, abs(btc_ret), breadth, btc_above_ema200=(btc_mom_90d > 0.0))
        self.regime_engine.vol_stress = btc_vol_stress
        if btc_mom_90d <= 0.0 and btc_vol_stress >= HARD_RISK_OFF_VOL_STRESS:
            state = "risk_off"
        elif risk_on:
            state = "risk_on"
        else:
            state = "risk_reduce"
        self.market_regime = state
        if state != self._last_regime_state:
            self.Debug(
                f"REGIME transition from={self._last_regime_state or 'init'} to={state} "
                f"btc_mom_90d={btc_mom_90d:+.4f} vol_stress={btc_vol_stress:.3f}"
            )
            self._last_regime_state = state
        return state

    def _rebalance_due(self):
        cadence = max(1, int(getattr(self.config, "rebalance_cadence_hours", 24) or 24))
        if self._last_rebalance_time is None:
            return self.Time.hour == 16
        elapsed_h = (self.Time - self._last_rebalance_time).total_seconds() / 3600
        return elapsed_h >= cadence and self.Time.hour == 16

    def _score_candidates(self, data: Slice):
        btc_ret, breadth = self._ingest_data(data)
        state = self._market_regime_state(btc_ret, breadth)
        scored = self._collect_scores(state, btc_ret)
        return state, scored

    def _rebalance_portfolio(self, scored, risk_scale: float = 1.0):
        breaker_disengaged_at = getattr(self, "_breaker_disengaged_at", None)
        if breaker_disengaged_at is not None:
            cooldown_h = float(getattr(self.config, "post_breaker_cooldown_hours", 48) or 48)
            hrs_since = (self.Time - breaker_disengaged_at).total_seconds() / 3600.0
            if hrs_since < cooldown_h:
                self.Debug(f"REBAL skip reason=post_breaker_cooldown hrs={hrs_since:.1f}/{cooldown_h:.1f}")
                return
            self._breaker_disengaged_at = None

        self._last_rebalance_time = self.Time
        top_k = max(1, int(getattr(self.config, "top_k", 8) or 8))
        effective_top_k = max(1, int(round(top_k * max(0.0, float(risk_scale)))))
        max_repl = max(0, int(getattr(self.config, "max_replacements_per_rebalance", 2)))
        min_hold = max(0, int(getattr(self.config, "min_hold_hours", 0)))
        min_delta = max(0.0, float(getattr(self.config, "min_rebalance_weight_delta", 0.03)))
        holdings = self._current_holdings()
        exited = self._force_exit_losers(scored)
        holdings = [h for h in holdings if h not in exited]
        held_set = set(holdings)
        ranked = [(s, score, feats) for s, score, feats in scored if score >= 0.0]
        top = ranked[:effective_top_k]
        top_symbols = [s for s, _, _ in top]
        top_set = set(top_symbols)
        losers = sorted(
            [s for s in holdings if s not in top_set],
            key=lambda sym: next((x[1] for x in ranked if x[0] == sym), DEFAULT_MISSING_SCORE),
        )
        entrants = [s for s in top_symbols if s not in held_set]
        replacements = []
        for loser, entrant in zip(losers[:max_repl], entrants[:max_repl]):
            pstate = self.position_state.get(loser)
            held_hours = INFINITE_HELD_HOURS
            if pstate is not None and getattr(pstate, "entry_time", None) is not None:
                held_hours = int((self.Time - pstate.entry_time).total_seconds() / 3600)
            if held_hours < min_hold:
                continue
            replacements.append((loser, entrant))
        target_set = set(holdings)
        for loser, entrant in replacements:
            target_set.discard(loser)
            target_set.add(entrant)
        for sym in top_symbols:
            if len(target_set) >= effective_top_k:
                break
            target_set.add(sym)
        target_list = sorted(
            target_set,
            key=lambda sym: next((x[1] for x in ranked if x[0] == sym), DEFAULT_MISSING_SCORE),
            reverse=True,
        )[:effective_top_k]
        target_set = set(target_list)

        equity = float(getattr(self.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        if equity <= 0 or not target_set:
            return
        weights_raw = {}
        for sym in target_list:
            feats = next((f for s, _, f in scored if s == sym), {})
            rv = max(float(feats.get("rv_21d", 0.0) or 0.0), 0.05)
            sec = self.Securities.get(sym)
            bid = float(getattr(sec, "BidPrice", 0.0) or 0.0)
            ask = float(getattr(sec, "AskPrice", 0.0) or 0.0)
            mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else float(getattr(sec, "Price", 0.0) or 0.0)
            spread_bps = (
                ((ask - bid) / mid * 1e4)
                if (bid > 0 and ask > 0 and mid > 0)
                else float(getattr(self.config, "assumed_spread_bps", 12.0))
            )
            spread_penalty = 1.0 + max(0.0, spread_bps - 10.0) / 50.0
            weights_raw[sym] = 1.0 / (rv * spread_penalty)
        total = sum(weights_raw.values()) or 1.0
        target_weights = {sym: max(0.0, float(risk_scale)) * (w / total) for sym, w in weights_raw.items()}
        sorted_by_score = sorted(scored, key=lambda x: x[1], reverse=True)
        rank_by_sym = {sym: i for i, (sym, _score, _feats) in enumerate(sorted_by_score)}

        def _conviction_cap(rank, dispersion_state):
            base = 0.20
            if dispersion_state != "full":
                return base
            if rank == 0:
                return 0.35
            if rank == 1:
                return 0.30
            if rank == 2:
                return 0.25
            if rank == 3:
                return 0.22
            return 0.20

        disp_state = self._dispersion_regime()
        floor_cap = float(getattr(self.config, "max_position_pct", 0.20) or 0.20)
        for sym in list(target_weights):
            cap = max(floor_cap, _conviction_cap(rank_by_sym.get(sym, 99), disp_state))
            target_weights[sym] = min(target_weights[sym], cap)
        total_deployment_cap = float(getattr(self.config, "total_deployment_cap", 0.85) or 0.85)
        total_w = sum(target_weights.values())
        if total_w > total_deployment_cap and total_w > 0:
            scale = total_deployment_cap / total_w
            target_weights = {sym: w * scale for sym, w in target_weights.items()}

        self.Debug(
            "REB "
            f"top={[s.Value for s in top_symbols]} "
            f"hold={[s.Value for s in holdings]} "
            f"repl={[f'{a.Value}->{b.Value}' for a, b in replacements]}"
        )

        exits_submitted = False
        for symbol in holdings:
            if symbol in target_set:
                continue
            if smart_liquidate(self, symbol, tag="RebalanceExit"):
                exits_submitted = True
            else:
                self.Debug(f"ORD_FAIL action=exit symbol={symbol.Value}")

        if exits_submitted:
            candidates = [
                {"symbol": sym, "score": score, "target_weight": target_weights[sym]}
                for sym, score, _feats in scored
                if sym in target_set and sym not in held_set
            ]
            existing_syms = {e["symbol"] for e in self._pending_rotation_entries}
            new_entries = [e for e in candidates if e["symbol"] not in existing_syms]
            self._pending_rotation_entries.extend(new_entries)
            self._pending_rotation_entry_time = self.Time
            self.Debug(f"REBAL exits_done pending_entries={len(self._pending_rotation_entries)}")
            return

        try:
            available_cash = float(self.Portfolio.CashBook["USD"].Amount)
        except Exception:
            available_cash = float(getattr(self.Portfolio, "Cash", 0.0) or 0.0)
        notional_cap = max(0.0, available_cash * 0.95)
        cost_skips = []
        for symbol in target_list:
            sec = self.Securities.get(symbol)
            if sec is None:
                continue
            price = float(getattr(sec, "Price", 0.0) or 0.0)
            if price <= 0:
                continue
            current_qty = float(getattr(self.Portfolio[symbol], "Quantity", 0.0) or 0.0)
            current_w = (current_qty * price) / max(equity, 1e-9)
            target_w = target_weights.get(symbol, 0.0)
            delta_w = target_w - current_w
            required_delta = min_delta if symbol in held_set else (min_delta * NEW_ENTRANT_MIN_DELTA_MULTIPLIER)
            if abs(delta_w) < required_delta:
                continue
            score = next((x[1] for x in ranked if x[0] == symbol), 0.0)
            fee_model = getattr(sec, "FeeModel", None)
            if delta_w > 0:
                feat_engine = getattr(self, "feature_engine", None)
                feats = feat_engine.current_features(getattr(symbol, "Value", str(symbol))) if feat_engine is not None else {}
                feats = feats or {}
                mom_24 = float(feats.get("mom_24", 0.0) or 0.0)
                if mom_24 > NO_CHASE_MOM24_THRESHOLD:
                    self.Debug(f"NO_CHASE sym={getattr(symbol,'Value',symbol)} mom_24={mom_24:.3f}")
                    continue
                vol_ratio = float(feats.get("vol_ratio_24h_7d", 1.0) or 1.0)
                if vol_ratio < MIN_VOLUME_RATIO_24H_7D:
                    self.Debug(f"NO_VOLUME sym={getattr(symbol,'Value',symbol)} vol_ratio={vol_ratio:.2f}")
                    continue
                desired_notional = equity * target_w
                notional = min(desired_notional, notional_cap)
                if notional < float(self.config.min_position_floor_usd):
                    self.Debug(
                        f"REBAL skip sym={symbol.Value} reason=cash_floor cash={available_cash:.2f} desired={desired_notional:.2f}"
                    )
                    continue
                qty = round_quantity(self, symbol, notional / max(price, 1e-9))
                if qty <= 0 or qty * price < get_min_notional_usd(self, symbol):
                    continue
                if not self.sizer.passes_cost_gate(symbol, score, notional, fee_model, is_limit=True):
                    cost_skips.append(symbol.Value)
                    continue
                ticket = place_entry(self, symbol, qty, tag="Rebalance:entry", signal_score=score)
                if ticket is not None:
                    available_cash -= qty * price
                    notional_cap = max(0.0, available_cash * 0.95)
                    if available_cash < float(self.config.min_position_floor_usd):
                        self.Debug("REBAL cash exhausted -- defer remaining entries to next rebalance")
                        break
            else:
                notional = abs(delta_w) * equity
                qty = round_quantity(self, symbol, notional / max(price, 1e-9))
                if qty <= 0 or qty * price < get_min_notional_usd(self, symbol):
                    continue
                ticket = place_limit_or_market(self, symbol, -qty, tag="Rebalance")
            if ticket is None:
                self.Debug(f"ORD_FAIL action={'buy' if delta_w > 0 else 'sell'} symbol={symbol.Value}")
        if cost_skips:
            self.Debug(f"REB skip_cost_gate={cost_skips}")

    def _ingest_data(self, data: Slice):
        btc_ret = 0.0
        breadth_votes = []
        feed_symbols = [*self.reference_symbols, *self.symbols]
        seen = set()
        for symbol in feed_symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            bar = data.Bars.get(symbol)
            if bar is None:
                continue
            self._update_symbol_series(symbol, bar)
            self.feature_engine.update(
                {
                    "symbol": symbol.Value,
                    "open": bar.Open,
                    "high": bar.High,
                    "low": bar.Low,
                    "close": bar.Close,
                    "volume": bar.Volume,
                }
            )
            self.regime_engine.hurst.update(symbol, bar)
            self.regime_engine.vr.update(symbol, bar)
            if getattr(self.config, "signal_mode", "cross_sectional_momentum") == "microstructure":
                self.signal_features.update(symbol, bar)
                ticks = getattr(data, "Ticks", {}).get(symbol, []) if hasattr(data, "Ticks") else []
                for tick in ticks:
                    self.signal_features.update(symbol, tick)
            feats = self.feature_engine.current_features(symbol.Value)
            if feats:
                breadth_votes.append(1.0 if feats.get("ema20", 0.0) > feats.get("ema50", 0.0) else 0.0)
            if symbol.Value == "BTCUSD":
                btc_ret = math.log(max(bar.Close, 1e-9) / max(bar.Open, 1e-9))
                btc_close = float(getattr(bar, "Close", 0.0) or 0.0)
                if btc_close > 0:
                    self.regime_engine.update_btc_close(btc_close)
        breadth = sum(breadth_votes) / len(breadth_votes) if breadth_votes else 0.5
        return btc_ret, breadth

    def OnData(self, data: Slice):  # pragma: no cover
        self._ensure_monthly_universe()
        self._bar_count += 1
        day_key = self.Time.date() if hasattr(self, "Time") else None
        if day_key is not None and self._last_daily_summary_date is not None and day_key != self._last_daily_summary_date:
            daily = self.reporter.daily_report()
            self.Debug(f"DAY_SUMMARY prev_date={self._last_daily_summary_date} trades={int(daily.get('daily_trade_count', 0.0))}")
        if day_key is not None:
            self._last_daily_summary_date = day_key
        if self._bar_count % 24 == 0:
            equity = float(getattr(self.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
            self.Debug(
                f"HB t={getattr(self, 'Time', 'n/a')} warmup={bool(getattr(self, 'IsWarmingUp', False))} "
                f"bars={self._bar_count} equity={equity:.2f}"
            )
        if self.IsWarmingUp:
            self._ingest_data(data)
            return

        self._drawdown_breaker.update(self)
        if self._drawdown_breaker.is_triggered():
            if not self._breaker_liquidated:
                liquidate_all_positions(self, tag="Breaker")
                self._breaker_liquidated = True
                self.Debug("BREAKER liquidate_once=true")
            self.reporter.tick("breaker")
            return
        else:
            if self._breaker_liquidated:
                self.Debug("BREAKER disengaged")
                self._breaker_disengaged_at = self.Time
            self._breaker_liquidated = False

        exits = manage_open_positions(self, data)
        for sym, tag in exits:
            self.Debug(f"EXIT sym={sym.Value} tag={tag}")

        cfg = getattr(self, "config", None)
        mode = str(getattr(cfg, "strategy_mode", "scalper"))
        if mode == "scalper":
            self._scalper_on_data(data)
            return
        self._momentum_on_data(data)

    def _momentum_on_data(self, data: Slice):
        # Tier 7: drain deferred entries every bar (cash-aware, idempotent if list empty)
        self._process_pending_entries(scored_lookup=None)

        state, scored = self._score_candidates(data)
        if self.Time.hour == 8 and getattr(self, "_last_scored", None):
            if getattr(self, "_last_force_exit_date", None) != self.Time.date():
                self._force_exit_losers(self._last_scored)
                self._last_force_exit_date = self.Time.date()
        disp = self._dispersion_regime()
        if self._dispersion_history:
            day_key = self.Time.date() if hasattr(self, "Time") else None
            if day_key is not None and day_key != self._last_dispersion_log_date:
                self.Debug(
                    f"DISP regime={disp} current={self._dispersion_history[-1]:.4f} n={len(self._dispersion_history)}"
                )
                self._last_dispersion_log_date = day_key
        reg_engine = getattr(self, "regime_engine", None)
        btc_gate_open = reg_engine.btc_above_ema30d() if reg_engine is not None else True
        if not btc_gate_open:
            gate_day = self.Time.date() if hasattr(self, "Time") else None
            if gate_day is not None and getattr(self, "_last_btc_gate_log_date", None) != gate_day:
                self.Debug("GATE btc_below_ema30d=true -- no new entries")
                self._last_btc_gate_log_date = gate_day
        if state == "risk_off" or disp == "flat":
            liquidate_all_positions(self, tag="RiskOff" if state == "risk_off" else "FlatDispersion")
        elif self._rebalance_due() and btc_gate_open:
            if state == "risk_on" and disp == "full":
                risk_scale = 1.0
            elif state == "risk_on" and disp == "half":
                risk_scale = 0.5
            elif state == "risk_reduce" and disp == "full":
                risk_scale = 0.5
            else:
                risk_scale = 0.25
            self._rebalance_portfolio(scored, risk_scale=risk_scale)

        bars_since_trade = self._bar_count - self._last_trade_bar
        if (
            state == "risk_on"
            and bars_since_trade > NO_TRADE_HEARTBEAT_THRESHOLD_BARS
            and scored
            and (self._bar_count - self._last_no_trade_log_bar >= NO_TRADE_HEARTBEAT_LOG_CADENCE_BARS)
        ):
            top_symbol, top_score, _ = scored[0]
            self.Debug(
                f"NO_TRADE_HB bars_since_trade={bars_since_trade} "
                f"top_candidate={top_symbol.Value} top_score={top_score:.3f} state={state}"
            )
            self._last_no_trade_log_bar = self._bar_count

        for escalated in escalate_stale_orders(self):
            self.reporter.on_order_event({"status": "escalated", "symbol": getattr(escalated, "Value", str(escalated))})

        self.reporter.tick(state)

    def _scalper_on_data(self, data: Slice):
        try:
            from scalper import evaluate_entry, evaluate_exit
        except ModuleNotFoundError:  # pragma: no cover
            from .scalper import evaluate_entry, evaluate_exit  # type: ignore

        self._ingest_data(data)

        today = self.Time.date()
        if self._scalper_daily_anchor_date != today:
            self._scalper_daily_anchor_date = today
            self._scalper_daily_anchor_equity = float(self.Portfolio.TotalPortfolioValue or 0.0)
            self._scalper_daily_pnl = 0.0
        equity = float(self.Portfolio.TotalPortfolioValue or 0.0)
        anchor = self._scalper_daily_anchor_equity or equity
        self._scalper_daily_pnl = (equity / anchor - 1.0) if anchor > 0 else 0.0

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
                    continue
                state = PositionState(entry_price=avg_px, highest_close=avg_px, entry_atr=avg_px * 0.05, entry_time=self.Time)
                self.position_state[sym] = state
            sec = self.Securities.get(sym)
            px = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
            should_exit, tag = evaluate_exit(
                symbol=sym,
                feats=feats,
                entry_price=state.entry_price,
                entry_time=state.entry_time,
                current_time=self.Time,
                current_price=px,
                btc_ret_1h=btc_ret_1h,
                config=self.config,
            )
            if should_exit:
                pnl_frac = (px / state.entry_price - 1.0) if state.entry_price > 0 else 0.0
                if smart_liquidate(self, sym, tag=tag):
                    self.Debug(f"SCALPER_EXIT sym={sym.Value} tag={tag} pnl={pnl_frac*100:.2f}%")
                    self.position_state.pop(sym, None)
                    self._scalper_recent_pnls.append(pnl_frac)
                    if len(self._scalper_recent_pnls) >= 3 and all(p < 0 for p in list(self._scalper_recent_pnls)[-3:]):
                        self._scalper_session_brake_until = self.Time + timedelta(hours=6)
                        self.Debug(f"SCALPER_BRAKE session=3consec_losses until={self._scalper_session_brake_until}")
                    if pnl_frac < 0:
                        self._scalper_consec_losses[sym] = self._scalper_consec_losses.get(sym, 0) + 1
                    else:
                        self._scalper_consec_losses[sym] = 0
                    self._scalper_last_trade_time[sym] = self.Time

        if self._scalper_session_brake_until is not None and self.Time < self._scalper_session_brake_until:
            return
        elif self._scalper_session_brake_until is not None:
            self._scalper_session_brake_until = None

        if self._scalper_daily_pnl < self.config.scalper_daily_loss_brake:
            return

        held = self._current_holdings()
        if len(held) >= int(self.config.scalper_max_concurrent):
            return

        try:
            available_cash = float(self.Portfolio.CashBook["USD"].Amount)
        except Exception:
            available_cash = float(getattr(self.Portfolio, "Cash", 0.0) or 0.0)
        available_cash_pct = (available_cash / equity) if equity > 0 else 0.0

        candidates = []
        for symbol in self.symbols[: int(self.config.scalper_universe_size)]:
            feats = self.feature_engine.current_features(symbol.Value) or {}
            if not feats:
                continue
            last_t = self._scalper_last_trade_time.get(symbol)
            last_trade_hours_ago = ((self.Time - last_t).total_seconds() / 3600.0) if last_t else 1e9
            ok, _reason = evaluate_entry(
                symbol=symbol,
                feats=feats,
                btc_ret_1h=btc_ret_1h,
                btc_ret_6h=btc_ret_6h,
                has_position=(symbol in held),
                last_trade_hours_ago=last_trade_hours_ago,
                available_cash_pct=available_cash_pct,
                daily_pnl_pct=self._scalper_daily_pnl,
                consecutive_losses_for_symbol=self._scalper_consec_losses.get(symbol, 0),
                config=self.config,
            )
            if ok:
                candidates.append((symbol, float(feats.get("z_20h", 0.0) or 0.0)))

        candidates.sort(key=lambda x: x[1])
        slots_remaining = int(self.config.scalper_max_concurrent) - len(held)
        for symbol, z in candidates[:slots_remaining]:
            sec = self.Securities.get(symbol)
            price = float(getattr(sec, "Price", 0.0) or 0.0) if sec is not None else 0.0
            if price <= 0:
                continue
            notional = min(equity * float(self.config.scalper_position_size_pct), available_cash * 0.95)
            if notional < float(self.config.min_position_floor_usd):
                continue
            qty = round_quantity(self, symbol, notional / max(price, 1e-9))
            if qty <= 0:
                continue
            ticket = place_entry(self, symbol, qty, tag="Scalper:entry", signal_score=float(-z))
            if ticket is not None:
                available_cash -= qty * price
                self.Debug(f"SCALPER_ENTRY sym={symbol.Value} z={z:.2f} qty={qty:.6f} px={price:.4f}")
                self._scalper_last_trade_time[symbol] = self.Time

    def OnOrderEvent(self, event):  # pragma: no cover
        self.reporter.on_order_event(self, event)
        status = getattr(event, "Status", None)
        status_str = str(status).lower() if status is not None else ""
        symbol = getattr(event, "Symbol", None)
        is_invalid = "invalid" in status_str
        if is_invalid:
            if symbol is not None:
                getattr(self, "_submitted_orders", {}).pop(symbol, None)
                if not hasattr(self, "_failed_escalations"):
                    self._failed_escalations = {}
                self._failed_escalations[symbol] = self.Time
                self.Debug(f"INVALID sym={symbol.Value} cooldown=24h")
            return
        is_canceled = "canceled" in status_str or "cancelled" in status_str
        if is_canceled:
            if symbol is not None:
                getattr(self, "_submitted_orders", {}).pop(symbol, None)
            return
        is_filled = "filled" in status_str and "partial" not in status_str
        if not is_filled:
            return
        self._last_trade_bar = self._bar_count
        if symbol is None:
            return
        if hasattr(self, "_scalper_last_trade_time"):
            self._scalper_last_trade_time[symbol] = self.Time
        try:
            qty_now = float(getattr(self.Portfolio[symbol], "Quantity", 0.0) or 0.0)
        except Exception:
            qty_now = 0.0
        if qty_now <= 0:
            self.position_state.pop(symbol, None)
            try:
                from execution import is_invested_not_dust

                if not is_invested_not_dust(self, symbol):
                    self._abandoned_dust.add(symbol)
            except Exception:
                pass
        if qty_now > 0 and symbol not in self.position_state:
            feats = self.feature_engine.current_features(symbol.Value)
            atr = float(feats.get("atr", 0.0) or 0.0)
            fill_px = float(getattr(event, "FillPrice", 0.0) or 0.0)
            if fill_px > 0:
                self.position_state[symbol] = PositionState(
                    entry_price=fill_px,
                    highest_close=fill_px,
                    entry_atr=atr if atr > 0 else fill_px * 0.02,
                    entry_time=self.Time,
                )

    def OnEndOfAlgorithm(self):  # pragma: no cover
        self.reporter.final_report()
