from __future__ import annotations

import math
from collections import defaultdict, deque

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


class SweetWaterPhase1(QCAlgorithm):
    def Initialize(self):  # pragma: no cover
        self.config = CONFIG
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
        self.long_only = True
        self.SetCash(self.config.starting_cash)
        self.SetStartDate(self.config.start_year, self.config.start_month, self.config.start_day)
        self.SetEndDate(self.config.end_year, self.config.end_month, self.config.end_day)
        self.SetWarmup(self.config.warmup_bars, Resolution.Hour)

        self.feature_engine = FeatureEngine(signal_mode=getattr(self.config, "signal_mode", "microstructure"))
        self.regime_engine = RegimeEngine(self.config)
        self.scorer = Scorer(self.config)
        self.signal_features = SignalFeatureStack(self)
        self.sizer = Sizer(self.config)
        self.risk = RiskManager(self.config)
        self.reporter = Reporter(self.config)

        self.max_participation_rate = 0.15
        self.spread_limit_pct = 0.025

        self._drawdown_breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.10)

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
        now = self.Time if hasattr(self, "Time") else pd.Timestamp.now(tz="UTC")
        self._last_rebalance_month = (int(now.year), int(now.month))
        status = self.signal_features.init_status()
        self.Debug(
            "SIG init "
            + " ".join([f"{k}={v}" for k, v in status.items()])
            + f" mode={getattr(self.config, 'signal_mode', 'microstructure')}"
        )

    def _subscribe_symbol(self, ticker: str):  # pragma: no cover
        if ticker in self.symbol_by_ticker:
            return self.symbol_by_ticker[ticker]
        sec_obj = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
        if getattr(self.config, "signal_mode", "microstructure") == "microstructure" and bool(getattr(self, "LiveMode", False)):
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
        scored = []
        for symbol in self.symbols:
            feats = self.feature_engine.current_features(symbol.Value)
            if not feats:
                continue
            snap = self.scorer.score(
                symbol=symbol,
                features=feats,
                regime_state=regime_state,
                btc_context={"btc_trend": btc_ret},
                signal_stack=self.signal_features,
                regime_engine=self.regime_engine,
            )
            scored.append((symbol, float(snap.get("final", 0.0)), feats))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _current_holdings(self):
        holdings = []
        for symbol in self.symbols:
            if is_invested_not_dust(self, symbol):
                holdings.append(symbol)
        return holdings

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
        cadence = max(1, int(getattr(self.config, "rebalance_cadence_hours", 6) or 6))
        if self._last_rebalance_time is None:
            return True
        return (self.Time - self._last_rebalance_time).total_seconds() >= cadence * 3600

    def _score_candidates(self, data: Slice):
        btc_ret, breadth = self._ingest_data(data)
        state = self._market_regime_state(btc_ret, breadth)
        scored = self._collect_scores(state, btc_ret)
        return state, scored

    def _rebalance_portfolio(self, scored, risk_scale: float = 1.0):
        self._last_rebalance_time = self.Time
        top_k = max(1, int(getattr(self.config, "top_k", 8) or 8))
        effective_top_k = max(1, int(round(top_k * max(0.0, float(risk_scale)))))
        max_repl = max(0, int(getattr(self.config, "max_replacements_per_rebalance", 2)))
        min_hold = max(0, int(getattr(self.config, "min_hold_hours", 0)))
        min_delta = max(0.0, float(getattr(self.config, "min_rebalance_weight_delta", 0.03)))
        score_threshold = float(getattr(self.config, "score_threshold", 0.0) or 0.0)
        holdings = self._current_holdings()
        held_set = set(holdings)
        ranked = [(s, score, feats) for s, score, feats in scored if score >= score_threshold]
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
        self.Debug(
            "REB "
            f"top={[s.Value for s in top_symbols]} "
            f"hold={[s.Value for s in holdings]} "
            f"repl={[f'{a.Value}->{b.Value}' for a, b in replacements]}"
        )

        for symbol in holdings:
            if symbol in target_set:
                continue
            if not smart_liquidate(self, symbol, tag="RebalanceExit"):
                self.Debug(f"ORD_FAIL action=exit symbol={symbol.Value}")

        equity = float(getattr(self.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        if equity <= 0 or not target_set:
            return
        target_w = max(0.0, float(risk_scale)) / len(target_set)
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
            delta_w = target_w - current_w
            required_delta = min_delta if symbol in held_set else (min_delta * NEW_ENTRANT_MIN_DELTA_MULTIPLIER)
            if abs(delta_w) < required_delta:
                continue
            notional = abs(delta_w) * equity
            qty = round_quantity(self, symbol, notional / max(price, 1e-9))
            if qty <= 0 or qty * price < get_min_notional_usd(self, symbol):
                continue
            score = next((x[1] for x in ranked if x[0] == symbol), 0.0)
            fee_model = getattr(sec, "FeeModel", None)
            if delta_w > 0 and not self.sizer.passes_cost_gate(symbol, score, notional, fee_model, is_limit=True):
                cost_skips.append(symbol.Value)
                continue
            signed_qty = qty if delta_w > 0 else -qty
            ticket = place_limit_or_market(self, symbol, signed_qty, tag="Rebalance")
            if ticket is None:
                self.Debug(f"ORD_FAIL action={'buy' if signed_qty > 0 else 'sell'} symbol={symbol.Value}")
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
            if getattr(self.config, "signal_mode", "microstructure") == "microstructure":
                self.signal_features.update(symbol, bar)
                self.regime_engine.hurst.update(symbol, bar)
                self.regime_engine.vr.update(symbol, bar)
                ticks = getattr(data, "Ticks", {}).get(symbol, []) if hasattr(data, "Ticks") else []
                for tick in ticks:
                    self.signal_features.update(symbol, tick)
            feats = self.feature_engine.current_features(symbol.Value)
            if feats:
                breadth_votes.append(1.0 if feats.get("ema20", 0.0) > feats.get("ema50", 0.0) else 0.0)
            if symbol.Value == "BTCUSD":
                btc_ret = math.log(max(bar.Close, 1e-9) / max(bar.Open, 1e-9))
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
            liquidate_all_positions(self, tag="Breaker")
            self.reporter.tick("breaker")
            return

        state, scored = self._score_candidates(data)
        if state == "risk_off":
            liquidate_all_positions(self, tag="RiskOff")
        elif self._rebalance_due():
            risk_scale = 1.0 if state == "risk_on" else 0.5
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

    def OnOrderEvent(self, event):  # pragma: no cover
        status = getattr(event, "Status", None)
        if status in ("Filled", 3) or str(getattr(event, "Status", "")).lower() == "filled":
            self._last_trade_bar = self._bar_count
        self.reporter.on_order_event(self, event)

    def OnEndOfAlgorithm(self):  # pragma: no cover
        self.reporter.final_report()
