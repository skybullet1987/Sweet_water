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
    execute_regime_entries,
    escalate_stale_orders,
    liquidate_all_positions,
    manage_open_positions,
    position_status,
    smart_liquidate,
)
from features import FeatureEngine, SignalFeatureStack
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
        self.long_only = True
        self.SetCash(self.config.starting_cash)
        self.SetStartDate(self.config.start_year, self.config.start_month, self.config.start_day)
        self.SetEndDate(self.config.end_year, self.config.end_month, self.config.end_day)
        self.SetWarmup(self.config.warmup_bars, Resolution.Hour)

        self.feature_engine = FeatureEngine()
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
        if getattr(self.config, "signal_mode", "microstructure") == "microstructure":
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

    def _score_candidates(self, data: Slice):
        btc_ret, breadth = self._ingest_data(data)
        btc_vol = abs(btc_ret)
        btc_above_ema200 = True
        for ref in self.reference_symbols:
            feats = self.feature_engine.current_features(ref.Value)
            if ref.Value == "BTCUSD" and feats:
                btc_above_ema200 = feats.get("ema20", 0.0) >= feats.get("ema200", 0.0)
                break
        self.regime_engine.update(btc_ret, btc_vol, breadth, btc_above_ema200=btc_above_ema200)
        state = self.regime_engine.current_state()
        self.market_regime = state
        self.sizer.update_returns(btc_ret)
        entry_threshold = (
            self.config.score_threshold * (self.config.chop_threshold_multiplier if state == "chop" else 1.0)
            if self.config.signal_mode == "legacy"
            else self.config.micro_entry_threshold
        )
        equity, gross = self._portfolio_state()
        gate_ok = self.regime_engine.gates_pass(breadth)
        candidates = []
        symbol_scores = [(symbol, self.feature_engine.current_features(symbol.Value)) for symbol in self.symbols]
        symbol_scores = [(symbol, feats) for symbol, feats in symbol_scores if feats]
        mom24 = [float(feats.get("mom_24", 0.0)) for _, feats in symbol_scores]
        mom168 = [float(feats.get("mom_168", 0.0)) for _, feats in symbol_scores]

        def _rank(values, idx):
            if not values:
                return 0.5
            v = values[idx]
            return (sum(1 for x in values if x < v) + 0.5 * sum(1 for x in values if x == v)) / len(values)

        for idx, (symbol, feats) in enumerate(symbol_scores):
            snap = self.scorer.score(
                symbol=symbol,
                features=feats,
                regime_state=state,
                btc_context={"btc_trend": btc_ret},
                rank_24h=_rank(mom24, idx),
                rank_168h=_rank(mom168, idx),
                signal_stack=self.signal_features,
                regime_engine=self.regime_engine,
            )
            composite_score = float(snap["final"])
            action = "hold"
            if position_status(self, symbol) == "long" and abs(composite_score) < float(self.config.micro_flatten_threshold):
                if smart_liquidate(self, symbol, tag="Flatten"):
                    action = "flatten"
            elif (
                gate_ok
                and position_status(self, symbol) == "flat"
                and composite_score > 0
                and abs(composite_score) > float(entry_threshold)
            ):
                atr_values = [
                    float(self.feature_engine.current_features(sym.Value).get("atr", 1.0))
                    for sym in self.symbols
                    if position_status(self, sym) == "long"
                ] or [max(float(feats.get("atr", 1.0)), 1e-6)]
                target = self.sizer.size_for_trade(
                    symbol.Value,
                    composite_score,
                    {
                        "equity": equity,
                        "gross_exposure": gross,
                        "realized_vol_annual": float(feats.get("realized_vol_30", self.config.target_annual_vol)),
                        "atr": float(feats.get("atr", 1.0)),
                        "open_position_atrs": atr_values,
                    },
                )
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
                if decision.approved:
                    sec = self.Securities[symbol]
                    notional = equity * abs(decision.adjusted_target_weight)
                    fee_model = getattr(sec, "FeeModel", None)
                    if self.sizer.passes_cost_gate(symbol, composite_score, notional, fee_model, is_limit=True):
                        candidates.append((symbol, composite_score, decision.adjusted_target_weight))
                        action = "enter"
            self.Debug(
                "SIG "
                f"sym={symbol.Value} cvd={float(snap['cvd']):+.2f} ofi={float(snap['ofi']):+.2f} vol={float(snap['volc']):+.2f} "
                f"rot={float(snap['rot']):+.2f} mult={float(snap['mult']):.2f} H={float(snap['hurst']):.2f} "
                f"raw={float(snap['raw']):+.2f} final={composite_score:+.2f} action={action}"
            )
        return state, candidates

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

        self._drawdown_breaker.update(self)
        if self._drawdown_breaker.is_triggered():
            liquidate_all_positions(self, tag="Breaker")
            self.reporter.tick("breaker")
            return

        state, candidates = self._score_candidates(data)
        execute_regime_entries(self, candidates, regime_tag=state)

        manage_open_positions(self, data)

        for escalated in escalate_stale_orders(self):
            self.reporter.on_order_event({"status": "escalated", "symbol": getattr(escalated, "Value", str(escalated))})

        self.reporter.tick(state)

    def OnOrderEvent(self, event):  # pragma: no cover
        self.reporter.on_order_event(self, event)

    def OnEndOfAlgorithm(self):  # pragma: no cover
        self.reporter.final_report()
