from __future__ import annotations

import math
from collections import defaultdict, deque

import pandas as pd

try:  # pragma: no cover
    from AlgorithmImports import AccountType, BrokerageName, Market, QCAlgorithm, Resolution, Slice
except ImportError:  # pragma: no cover
    LEAN_IMPORT_ERROR = RuntimeError("AlgorithmImports is required when running in QuantConnect LEAN.")

    class QCAlgorithm:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise LEAN_IMPORT_ERROR

    class Resolution:
        Hour = "Hour"
        Minute = "Minute"

    class BrokerageName:
        Kraken = "Kraken"

    class AccountType:
        Cash = "Cash"

    class Market:
        Kraken = "Kraken"

    class Slice:  # type: ignore
        pass

from config import CONFIG
from execution import (
    KrakenTieredFeeModel,
    RealisticCryptoSlippage,
    execute_regime_entries,
    escalate_stale_orders,
    manage_open_positions,
)
from features import FeatureEngine
from regime import RegimeEngine
from reporting import Reporter
from risk import DrawdownCircuitBreaker, RiskManager
from scoring import Scorer
from sizing import Sizer
from universe import KRAKEN_SAFE_LIST, REFERENCE_SYMBOLS, select_universe


class SweetWaterPhase1(QCAlgorithm):
    def Initialize(self):  # pragma: no cover
        self.config = CONFIG
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
        self.SetCash(self.config.starting_cash)
        self.SetStartDate(self.config.start_year, self.config.start_month, self.config.start_day)
        self.SetEndDate(self.config.end_year, self.config.end_month, self.config.end_day)
        self.SetWarmup(self.config.warmup_bars, Resolution.Hour)

        self.feature_engine = FeatureEngine()
        self.regime_engine = RegimeEngine(self.config)
        self.scorer = Scorer()
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
        now = getattr(self, "Time", pd.Timestamp.now(tz="UTC"))
        self._last_rebalance_month = (int(now.year), int(now.month))

    def _subscribe_symbol(self, ticker: str):  # pragma: no cover
        if ticker in self.symbol_by_ticker:
            return self.symbol_by_ticker[ticker]
        sec_obj = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
        sec_obj.FeeModel = KrakenTieredFeeModel()
        sec_obj.SetSlippageModel(RealisticCryptoSlippage(self))
        self.symbol_by_ticker[ticker] = sec_obj.Symbol
        return sec_obj.Symbol

    def _init_execution_attrs(self):  # pragma: no cover
        self.min_notional = 1.0
        self.max_spread_pct = 0.0080
        self.spread_widen_mult = 3.0
        self.volatility_regime = "normal"
        self.market_regime = "trending"
        self.log_budget = 200
        self.live_stale_order_timeout_seconds = 90
        self.stale_order_timeout_seconds = 300
        self.slip_outlier_threshold = 0.005
        self.quick_take_profit = 0.04
        self.tight_stop_loss = 0.025
        self.disable_recent_outcome_cash_mode = True
        self.disable_adaptive_ranking_memory = True
        self.expected_round_trip_fees = 0.0065
        self.stop_loss_pct = 0.025
        self.take_profit_pct = 0.05
        self.max_hold_hours = self.config.time_stop_bars
        self.stale_order_bars = 3
        self.enable_queue_rejection = False
        self.stress_spread_mult = 1.0
        self.breakout_nonfill_penalty = 0.08

        self._spread_warning_times = {}
        self._cancel_cooldowns = {}
        self._pending_orders = {}
        self._submitted_orders = {}
        self._order_retries = {}
        self._failed_exit_attempts = {}
        self._failed_exit_counts = {}
        self._partial_sell_symbols = set()
        self._session_blacklist = set()
        self._rolling_wins = deque(maxlen=200)
        self._rolling_win_sizes = deque(maxlen=200)
        self._rolling_loss_sizes = deque(maxlen=200)
        self._recent_trade_outcomes = deque(maxlen=20)
        self._slip_abs = deque(maxlen=200)
        self._symbol_slippage_history = {}
        self._spike_entries = {}
        self._partial_tp_taken = {}
        self._partial_tp_tier = {}
        self._entry_signal_combos = {}
        self._entry_engine = {}
        self._chop_engine = {}
        self._last_live_trade_time = None
        self._cash_mode_until = None

        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
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
        self.pnl_by_vol_regime = {}
        self.pnl_by_signal_combo = {}
        self.pnl_by_hold_time = {}
        self.pnl_by_engine = {}
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
        self._last_rebalance_month = current_month

    def _update_symbol_series(self, symbol, bar):
        state = self.crypto_data.setdefault(symbol, {"prices": deque(maxlen=96), "volume": deque(maxlen=96)})
        state["prices"].append(float(bar.Close))
        state["volume"].append(float(bar.Volume))

    def _portfolio_state(self):
        equity = float(self.Portfolio.TotalPortfolioValue)
        gross = float(self.Portfolio.TotalHoldingsValue) / max(equity, 1.0)
        return equity, gross

    def _score_candidates(self, data: Slice):
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
            feats = self.feature_engine.current_features(symbol.Value)
            if feats:
                breadth_votes.append(1.0 if feats.get("ema20", 0.0) > feats.get("ema50", 0.0) else 0.0)
            if symbol.Value == "BTCUSD":
                btc_ret = math.log(max(bar.Close, 1e-9) / max(bar.Open, 1e-9))

        btc_vol = abs(btc_ret)
        breadth = sum(breadth_votes) / len(breadth_votes) if breadth_votes else 0.5
        self.regime_engine.update(btc_ret, btc_vol, breadth)
        state = self.regime_engine.current_state()
        self.market_regime = state
        self.sizer.update_returns(btc_ret)

        equity, gross = self._portfolio_state()
        candidates = []
        for symbol in self.symbols:
            feats = self.feature_engine.current_features(symbol.Value)
            if not feats:
                continue
            score = self.scorer.score(symbol.Value, feats, state, {"btc_trend": btc_ret})
            if abs(score) < self.config.score_threshold:
                continue
            target = self.sizer.size_for_trade(symbol.Value, score, {"equity": equity, "gross_exposure": gross})
            decision = self.risk.evaluate(
                {
                    "symbol": symbol.Value,
                    "target_weight": target,
                    "equity": equity,
                    "gross_exposure": gross,
                    "net_exposure": 0.0,
                    "correlation": 0.0,
                }
            )
            if not decision.approved:
                continue
            sec = self.Securities[symbol]
            notional = equity * abs(decision.adjusted_target_weight)
            fee_model = getattr(sec, "FeeModel", None)
            if not self.sizer.passes_cost_gate(symbol, score, notional, fee_model, is_limit=True):
                continue
            candidates.append((symbol, score, decision.adjusted_target_weight))
        return state, candidates

    def OnData(self, data: Slice):  # pragma: no cover
        self._ensure_monthly_universe()

        self._drawdown_breaker.update(self)
        if self._drawdown_breaker.is_triggered():
            self.Liquidate()
            return

        state, candidates = self._score_candidates(data)
        execute_regime_entries(self, candidates, regime_tag=state)

        manage_open_positions(self)

        for escalated in escalate_stale_orders(self):
            self.reporter.on_order_event({"status": "escalated", "symbol": getattr(escalated, "Value", str(escalated))})

        self.reporter.tick(state)

    def OnOrderEvent(self, event):  # pragma: no cover
        self.reporter.on_order_event(self, event)

    def OnEndOfAlgorithm(self):  # pragma: no cover
        self.reporter.final_report()
