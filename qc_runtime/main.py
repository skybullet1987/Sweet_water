from AlgorithmImports import *

import math
from collections import defaultdict
import pandas as pd

from config.strategy_config import DEFAULT_STRATEGY_CONFIG
from circuit_breaker import DrawdownCircuitBreaker
from entry_exec import execute_regime_entries, manage_open_positions
from events import on_order_event
from execution import *
from exits.triple_barrier import build_barrier
from fee_model import KrakenTieredFeeModel
from features.indicators import adx, aroon, atr, bbands, cci, macd, mfi, rsi, ema
from features.microstructure import realized_vol
from order_management import *
from realistic_slippage import RealisticCryptoSlippage
from regime.hmm import HMMRegime, build_hmm_features
from reporting import *
from scoring.rule_scorer import score_symbol
from sizing.kelly import RollingKellyEstimator, fractional_kelly
from sizing.vol_target import VolTargetScaler
from strategy_core import initialize_symbol, update_symbol_data
from universe.dynamic import select_universe
from nextgen.core.models import PortfolioTarget
from nextgen.risk.engine import RiskConfig, UnifiedRiskEngine


class SimplifiedCryptoStrategy(QCAlgorithm):
    ALGO_VERSION = "v10.0.0-phase1"

    def Initialize(self):
        self.cfg = DEFAULT_STRATEGY_CONFIG
        self.SetStartDate(2025, 1, 1)
        self.SetEndDate(2025, 6, 30)
        self.SetCash(500)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        self.max_positions = self.cfg.max_positions
        self.crypto_data = {}
        self.current_universe = []
        self.symbol_by_ticker = {}
        self.open_positions = {}
        self.position_barriers = {}
        self.bar_index = 0

        self.hmm_regime = HMMRegime(self.cfg)
        self.kelly_estimator = RollingKellyEstimator(window=60)
        self.vol_target = VolTargetScaler(self.cfg)
        self._drawdown_entry_breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.10)
        self._risk_engine = UnifiedRiskEngine(
            RiskConfig(
                name="phase1",
                symbols=tuple(),
                target_volatility=self.cfg.target_annual_vol,
                max_gross_exposure=1.5,
                max_net_exposure=1.0,
                max_position_weight=0.35,
                drawdown_throttle=0.10,
                kill_switch_drawdown=0.25,
            )
        )

        self.SetWarmup(200, Resolution.Hour)
        self.Schedule.On(self.DateRules.MonthStart("BTCUSD"), self.TimeRules.At(0, 0), self._rebalance_universe)

        self._rebalance_universe()

    def _history_provider(self, symbol: str, start, end):
        sec = self.symbol_by_ticker.get(symbol)
        if sec is None:
            return None
        history = self.History(sec, start, end, Resolution.Hour)
        if history.empty:
            return None
        if "close" in history.columns:
            return history[["open", "high", "low", "close", "volume"]]
        hist = history.reset_index().rename(
            columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        )
        return hist[["open", "high", "low", "close", "volume"]]

    def _rebalance_universe(self):
        selected = select_universe(self._history_provider, self.Time)
        for ticker in selected:
            if ticker not in self.symbol_by_ticker:
                symbol = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken).Symbol
                self.symbol_by_ticker[ticker] = symbol
                initialize_symbol(self, symbol)
                sec = self.Securities[symbol]
                sec.FeeModel = KrakenTieredFeeModel()
                sec.SlippageModel = RealisticCryptoSlippage(self)
        self.current_universe = [self.symbol_by_ticker[t] for t in selected if t in self.symbol_by_ticker]

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        self.bar_index += 1
        for symbol in self.current_universe:
            bar = data.Bars.get(symbol)
            if bar is None:
                continue
            update_symbol_data(self, symbol, bar)

        btc_symbol = self.symbol_by_ticker.get("BTCUSD")
        if btc_symbol is None or btc_symbol not in self.crypto_data:
            return
        btc_df = self.crypto_data[btc_symbol].to_frame()
        if len(btc_df) < 60:
            return

        btc_close = btc_df["close"].astype(float)
        breadth = self._market_breadth()
        hmm_features = build_hmm_features(btc_close, breadth)
        latest = hmm_features.iloc[-1]
        self.hmm_regime.update(
            btc_log_return=float(latest["btc_log_return"]),
            btc_realized_vol=float(latest["btc_realized_vol"]),
            breadth=float(latest["breadth"]),
        )
        self.vol_target.update_returns(float(latest["btc_log_return"]))
        regime = self.hmm_regime.current_state()

        self._manage_existing_positions(data)
        self._drawdown_entry_breaker.update(self)
        if self._drawdown_entry_breaker.is_triggered():
            return
        if regime == "risk_off":
            return

        candidates = self._score_candidates(regime)
        if not candidates:
            return
        candidates = candidates[: self.cfg.max_positions]

        orders = []
        for candidate in candidates:
            symbol = candidate["symbol"]
            decision = self._risk_engine.evaluate_target(candidate["target"])
            if not decision.approved:
                continue
            p, r = self.kelly_estimator.estimate(regime)
            kelly = fractional_kelly(
                win_prob=p,
                win_loss_ratio=r,
                fraction=self.cfg.kelly_fraction,
                cap=self.cfg.kelly_cap,
            )
            target_weight = kelly * self.vol_target.scale() * decision.adjusted_target_weight
            if target_weight <= 0:
                continue
            price = float(self.Securities[symbol].Price)
            if price <= 0:
                continue
            qty = (self.Portfolio.TotalPortfolioValue * target_weight) / price
            candidate["quantity"] = qty
            orders.append(candidate)

        for c in orders:
            symbol = c["symbol"]
            atr_value = max(float(c.get("atr", 0.0)), 1e-9)
            side = "long" if c["score"] >= 0 else "short"
            barrier = build_barrier(
                entry_price=float(self.Securities[symbol].Price),
                atr=atr_value,
                side=side,
                entry_bar=self.bar_index,
                config=self.cfg,
            )
            self.position_barriers[symbol] = barrier

        execute_regime_entries(self, orders, regime=regime)

    def _score_candidates(self, regime: str):
        scored = []
        for symbol in self.current_universe:
            state = self.crypto_data.get(symbol)
            if state is None:
                continue
            frame = state.to_frame()
            if len(frame) < 60:
                continue

            indicators = self._compute_symbol_features(frame)
            score = score_symbol(indicators, regime=regime, btc_context={"btc_trend": 0.0})
            if abs(score) < self.cfg.score_threshold:
                continue

            target = PortfolioTarget(symbol.Value, target_weight=1.0 / self.cfg.max_positions, source_score=score)
            scored.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "threshold": self.cfg.score_threshold,
                    "atr": indicators.get("atr", 0.0),
                    "target": target,
                }
            )
        return sorted(scored, key=lambda x: abs(x["score"]), reverse=True)

    def _compute_symbol_features(self, frame):
        ema20 = ema(frame, 20).iloc[-1]
        ema50 = ema(frame, 50).iloc[-1]
        macd_df = macd(frame)
        bb = bbands(frame)
        atr14 = atr(frame, 14).iloc[-1]
        upper = float(bb["upper"].iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        close = float(frame["close"].iloc[-1])
        width = max(upper - lower, 1e-9)
        return {
            "adx": float(adx(frame, 14).iloc[-1]),
            "ema20": float(ema20),
            "ema50": float(ema50),
            "macd_hist": float(macd_df["hist"].iloc[-1]),
            "aroon_osc": float(aroon(frame, 14)["oscillator"].iloc[-1]),
            "mfi": float(mfi(frame, 14).iloc[-1]),
            "cci": float(cci(frame, 20).iloc[-1]),
            "rsi": float(rsi(frame, 14).iloc[-1]),
            "bb_pos": float((close - lower) / width),
            "atr": float(atr14),
        }

    def _market_breadth(self):
        values = []
        for symbol in self.current_universe:
            frame = self.crypto_data[symbol].to_frame()
            if len(frame) < 50:
                continue
            ema50_series = ema(frame, 50)
            if math.isnan(float(ema50_series.iloc[-1])):
                continue
            values.append(1.0 if float(frame["close"].iloc[-1]) > float(ema50_series.iloc[-1]) else 0.0)
        breadth = sum(values) / len(values) if values else 0.5
        return pd.Series([breadth], index=[self.Time])

    def _manage_existing_positions(self, data: Slice):
        position_snapshot = {}
        for symbol in list(self.position_barriers.keys()):
            bar = data.Bars.get(symbol)
            if bar is None:
                continue
            holding = self.Portfolio[symbol]
            if not holding.Invested:
                continue
            position_snapshot[symbol] = {
                "quantity": float(holding.Quantity),
                "high": float(bar.High),
                "low": float(bar.Low),
                "close": float(bar.Close),
            }
        manage_open_positions(self, position_snapshot, self.position_barriers, self.bar_index)

    def OnOrderEvent(self, orderEvent):
        on_order_event(self, orderEvent)
