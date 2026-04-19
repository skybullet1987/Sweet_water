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
from universe import select_universe


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

        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.06
        self.max_hold_hours = 24
        self.max_participation_rate = 0.15
        self.spread_limit_pct = 0.025
        self.stale_order_bars = self.config.stale_order_bars

        self._drawdown_breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.10)

        self.symbols = []
        self.symbol_by_ticker = {}
        self.open_snapshots = defaultdict(dict)

        # Bookkeeping state expected by restored order-event logic.
        self._pending_orders = {}
        self._submitted_orders = {}
        self._order_retries = {}
        self.entry_prices = {}
        self.highest_prices = {}
        self.entry_times = {}
        self.entry_volumes = {}
        self._rolling_wins = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._recent_trade_outcomes = deque(maxlen=20)
        self._partial_sell_symbols = set()
        self._failed_exit_attempts = {}
        self._failed_exit_counts = {}
        self._session_blacklist = set()
        self._spike_entries = {}
        self.daily_trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.consecutive_losses = 0
        self.peak_value = None
        self._cancel_cooldowns = {}
        self._spread_warning_times = {}
        self._symbol_slippage_history = {}
        self._slip_abs = deque(maxlen=50)
        self.pnl_by_tag = {}
        self.pnl_by_regime = {}
        self.pnl_by_vol_regime = {}
        self.pnl_by_signal_combo = {}
        self.pnl_by_hold_time = {}
        self.pnl_by_engine = {"trend": [], "chop": []}
        self.trade_log = deque(maxlen=500)
        self.crypto_data = {}

        self.expected_round_trip_fees = 0.0065
        self.stale_limit_cancels = 0
        self.stale_limit_escalations = 0
        self.stale_limit_escalation_fills = 0

        initial_universe = select_universe(self._history_provider, pd.Timestamp.utcnow())
        for ticker in initial_universe:
            sec_obj = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
            sec_obj.FeeModel = KrakenTieredFeeModel()
            sec_obj.SlippageModel = RealisticCryptoSlippage(self)
            self.symbol_by_ticker[ticker] = sec_obj.Symbol
        self.symbols = [self.symbol_by_ticker[t] for t in self.symbol_by_ticker]

    def _history_provider(self, symbol: str, start, end):  # pragma: no cover
        sec = self.symbol_by_ticker.get(symbol)
        if sec is None:
            return pd.DataFrame()
        hist = self.History(sec, start, end, Resolution.Hour)
        if hist is None or hist.empty:
            return pd.DataFrame()
        return hist[["open", "high", "low", "close", "volume"]]

    def _ensure_monthly_universe(self):  # pragma: no cover
        if self.Time.day != 1 or self.Time.hour != 0:
            return
        for ticker in select_universe(self._history_provider, self.Time):
            if ticker in self.symbol_by_ticker:
                continue
            sec_obj = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
            sec_obj.FeeModel = KrakenTieredFeeModel()
            sec_obj.SlippageModel = RealisticCryptoSlippage(self)
            self.symbol_by_ticker[ticker] = sec_obj.Symbol
        self.symbols = [self.symbol_by_ticker[t] for t in self.symbol_by_ticker]

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
        for symbol in self.symbols:
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
