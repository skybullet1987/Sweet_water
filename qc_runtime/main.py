from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd

try:  # pragma: no cover
    from AlgorithmImports import AccountType, BrokerageName, Market, QCAlgorithm, Resolution, Slice
except ImportError:  # pragma: no cover
    # Local test fallback only: these placeholders allow module import in non-LEAN
    # environments, but production execution must use QuantConnect's AlgorithmImports.
    LEAN_IMPORT_ERROR = RuntimeError("AlgorithmImports is required when running in QuantConnect LEAN.")

    class QCAlgorithm:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise LEAN_IMPORT_ERROR

    class Resolution:
        Hour = "Hour"

    class BrokerageName:
        Kraken = "Kraken"

    class AccountType:
        Cash = "Cash"

    class Market:
        Kraken = "Kraken"

    class Slice:  # type: ignore
        pass

from config import CONFIG
from execution import Executor
from features import FeatureEngine
from regime import RegimeEngine
from reporting import Reporter
from risk import RiskManager
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
        self.executor = Executor(self.config)
        self.risk = RiskManager(self.config)
        self.reporter = Reporter(self.config)
        self.symbols = []
        self.symbol_by_ticker = {}
        self.open_snapshots = defaultdict(dict)

    def _history_provider(self, symbol: str, start, end):  # pragma: no cover
        sec = self.symbol_by_ticker.get(symbol)
        if sec is None:
            return pd.DataFrame()
        hist = self.History(sec, start, end, Resolution.Hour)
        if hist is None or hist.empty:
            return pd.DataFrame()
        return hist[["open", "high", "low", "close", "volume"]]

    def OnData(self, data: Slice):  # pragma: no cover
        if self.Time.day == 1 and self.Time.hour == 0:
            for ticker in select_universe(self._history_provider, self.Time):
                if ticker not in self.symbol_by_ticker:
                    self.symbol_by_ticker[ticker] = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken).Symbol
            self.symbols = [self.symbol_by_ticker[t] for t in self.symbol_by_ticker]
        btc_ret = 0.0
        breadth_votes = []
        for symbol in self.symbols:
            bar = data.Bars.get(symbol)
            if bar is None:
                continue
            self.feature_engine.update({"symbol": symbol.Value, "open": bar.Open, "high": bar.High, "low": bar.Low, "close": bar.Close, "volume": bar.Volume})
            feats = self.feature_engine.current_features(symbol.Value)
            if feats:
                breadth_votes.append(1.0 if feats.get("ema20", 0.0) > feats.get("ema50", 0.0) else 0.0)
            if symbol.Value == "BTCUSD":
                btc_ret = math.log(max(bar.Close, 1e-9) / max(bar.Open, 1e-9))
        btc_vol = abs(btc_ret)
        breadth = sum(breadth_votes) / len(breadth_votes) if breadth_votes else 0.5
        self.regime_engine.update(btc_ret, btc_vol, breadth)
        state = self.regime_engine.current_state()
        self.sizer.update_returns(btc_ret)
        for symbol in self.symbols:
            feats = self.feature_engine.current_features(symbol.Value)
            if not feats:
                continue
            score = self.scorer.score(symbol.Value, feats, state, {"btc_trend": btc_ret})
            if abs(score) < self.config.score_threshold:
                continue
            target = self.sizer.size_for_trade(symbol.Value, score, {"equity": self.Portfolio.TotalPortfolioValue, "gross_exposure": self.Portfolio.TotalHoldingsValue / max(self.Portfolio.TotalPortfolioValue, 1.0)})
            decision = self.risk.evaluate({"target_weight": target, "equity": self.Portfolio.TotalPortfolioValue, "gross_exposure": self.Portfolio.TotalHoldingsValue / max(self.Portfolio.TotalPortfolioValue, 1.0), "net_exposure": 0.0, "correlation": 0.0})
            if decision.approved:
                self.executor.place_entry(symbol.Value, decision.adjusted_target_weight, score)
        for escalated in self.executor.escalate_stale_orders():
            self.reporter.on_order_event({"status": "escalated", "symbol": escalated})
        self.reporter.tick(state)

    def OnOrderEvent(self, orderEvent):  # pragma: no cover
        self.reporter.on_order_event({"status": "filled" if getattr(orderEvent, "Status", None) else "unknown", "pnl": 0.0})

    def OnEndOfAlgorithm(self):  # pragma: no cover
        self.reporter.final_report()
