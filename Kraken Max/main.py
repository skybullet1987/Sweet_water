from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import pandas as pd

try:  # pragma: no cover
    from AlgorithmImports import AccountType, BrokerageName, Market, QCAlgorithm, Resolution
except ImportError:  # pragma: no cover
    class QCAlgorithm:  # type: ignore
        pass

    class Resolution:
        Hour = "Hour"

    class BrokerageName:
        Kraken = "Kraken"

    class AccountType:
        Cash = "Cash"

    class Market:
        Kraken = "Kraken"

from config import CONFIG
from ensemble import AlphaEnsemble
from execution import get_min_qty, liquidate_symbol, manage_exits, market_value, place_buy, position_qty
from features import FeatureCache, cross_section_ranks
from ml_scorer import MLScorer
from regime import RegimeEngine
from risk import PortfolioRisk, PositionRisk
from sizing import AggressiveSizer
from universe import KRAKEN_MAX_UNIVERSE, REFERENCE_SYMBOLS, select_universe


class KrakenMaxAlgorithm(QCAlgorithm):
    """
    Kraken Max — aggressive long-only spot strategy for QuantConnect + Kraken.

    Design goal: maximize convexity in bull regimes via concentrated momentum/breakout
    ensemble with ML scoring. Cash account, no margin, Canada-aware universe bias.
    """

    def Initialize(self):  # pragma: no cover
        self.config = CONFIG
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
        self.SetCash(float(self.config.starting_cash))
        self.SetStartDate(
            self.config.start_year,
            self.config.start_month,
            self.config.start_day,
        )
        self.SetEndDate(
            self.config.end_year,
            self.config.end_month,
            self.config.end_day,
        )
        self.SetWarmup(self.config.warmup_bars, Resolution.Hour)

        self.feature_cache = FeatureCache()
        self.regime_engine = RegimeEngine(self.config)
        self.ensemble = AlphaEnsemble(self.config, MLScorer())
        self.sizer = AggressiveSizer(self.config)
        self.portfolio_risk = PortfolioRisk(self.config)
        self.position_risk: dict = {}
        self.symbol_by_ticker: dict[str, object] = {}
        self.active_universe: list[str] = []
        self._last_rebalance = None
        self._bar_count = 0

        for ticker in set(KRAKEN_MAX_UNIVERSE) | set(REFERENCE_SYMBOLS):
            self._subscribe(ticker)

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self._hourly_maintenance,
        )
        self.Debug(
            "KRAKEN_MAX init cash=%s long_only=True margin=False target=aggressive_growth"
            % self.config.starting_cash
        )

    def _subscribe(self, ticker: str) -> None:
        try:
            sec = self.AddCrypto(ticker, Resolution.Hour, Market.Kraken)
            self.symbol_by_ticker[ticker] = sec.Symbol
        except Exception as exc:
            self.Debug(f"KRAKEN_MAX skip subscribe {ticker}: {exc}")

    def OnData(self, slice) -> None:  # pragma: no cover
        self._bar_count += 1
        for ticker, sym in self.symbol_by_ticker.items():
            if sym not in slice.Bars:
                continue
            self.feature_cache.update(ticker, slice.Bars[sym])

        if self.IsWarmingUp:
            return

        now = self.Time
        if self._last_rebalance is None or (now - self._last_rebalance) >= timedelta(
            hours=int(self.config.rebalance_hours)
        ):
            self._rebalance(now)
            self._last_rebalance = now

        self._manage_positions(now)

    def _hourly_maintenance(self) -> None:  # pragma: no cover
        equity = float(self.Portfolio.TotalPortfolioValue)
        dd = self.portfolio_risk.update_peak(equity)
        if self.portfolio_risk.drawdown_halted(self.Time, dd):
            self._flatten_all("drawdown_halt")
            return
        if dd < -0.15 and self._bar_count % 24 == 0:
            self.Debug(f"KRAKEN_MAX drawdown={dd:.2%} equity={equity:.2f}")

    def _history_provider(self, ticker: str, start, end) -> pd.DataFrame:
        sym = self.symbol_by_ticker.get(ticker)
        if sym is None:
            return pd.DataFrame()
        try:
            hist = self.History(sym, start, end, Resolution.Hour)
        except Exception:
            return pd.DataFrame()
        if hist is None or hist.empty:
            return pd.DataFrame()
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.reset_index()
        cols = {c.lower(): c for c in hist.columns}
        out = pd.DataFrame(
            {
                "open": hist[cols.get("open", "open")],
                "high": hist[cols.get("high", "high")],
                "low": hist[cols.get("low", "low")],
                "close": hist[cols.get("close", "close")],
                "volume": hist[cols.get("volume", "volume")],
            }
        )
        return out.dropna()

    def _rebalance(self, now) -> None:  # pragma: no cover
        if self.portfolio_risk.drawdown_halted(now, 0.0):
            return
        if not self.portfolio_risk.can_place_order(now):
            return

        self.active_universe = select_universe(self._history_provider, now)
        feature_map = {
            t: self.feature_cache.features(t)
            for t in self.active_universe
            if t in self.symbol_by_ticker
        }
        feature_map = {k: v for k, v in feature_map.items() if v}
        if len(feature_map) < 3:
            return

        rank_mom = cross_section_ranks(feature_map, "mom_21d")
        rank_bo = cross_section_ranks(feature_map, "breakout_strength")
        rv_vals = [float(f.get("rv_21d", 0.05)) for f in feature_map.values()]
        median_rv = sorted(rv_vals)[len(rv_vals) // 2] if rv_vals else 0.05
        breadth = sum(1 for f in feature_map.values() if float(f.get("mom_7d", 0)) > 0) / len(feature_map)

        btc_feat = feature_map.get("BTCUSD") or self.feature_cache.features("BTCUSD") or {}
        regime = self.regime_engine.classify(
            btc_features=btc_feat,
            breadth=breadth,
            median_rv=median_rv,
        )
        if not regime.allow_new_entries or regime.deployment_cap <= 0:
            self._flatten_all("regime_chaos")
            return

        scores: list[tuple[str, float, dict]] = []
        for ticker, feats in feature_map.items():
            if regime.prefer_symbols and ticker not in regime.prefer_symbols and regime.name == "bear":
                continue
            comp = self.ensemble.score_symbol(
                feats,
                rank_mom_21=rank_mom.get(ticker, 0.5),
                rank_breakout=rank_bo.get(ticker, 0.5),
                breadth=breadth,
            )
            scores.append((ticker, float(comp["final"]), comp))
        scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [s for s, sc, _ in scores if sc >= float(self.config.entry_score_threshold)]
        targets = candidates[: int(self.config.top_k)]

        equity = float(self.Portfolio.TotalPortfolioValue)
        deployable = equity * regime.deployment_cap
        per_slot = deployable / max(len(targets), 1)

        held = {t for t in self.symbol_by_ticker if position_qty(self, self.symbol_by_ticker[t]) > 0}

        for ticker in list(held):
            if ticker not in targets:
                sym = self.symbol_by_ticker[ticker]
                liquidate_symbol(self, sym)
                self.position_risk.pop(ticker, None)

        for ticker in targets:
            sym = self.symbol_by_ticker.get(ticker)
            if sym is None:
                continue
            feats = feature_map[ticker]
            sc = next((x[1] for x in scores if x[0] == ticker), 0.0)
            rank_pct = rank_mom.get(ticker, 0.5)
            weight = self.sizer.weight_for_score(sc, float(feats.get("rv_21d", 0.2)), rank_pct)
            notional = min(per_slot, equity * weight)
            if not self.sizer.passes_cost_gate(sc, notional):
                continue

            qty = position_qty(self, sym)
            if qty > 0:
                self._maybe_pyramid(ticker, sym, feats, sc, equity)
                continue

            if not self.portfolio_risk.can_place_order(now):
                break
            if place_buy(self, sym, notional):
                self.portfolio_risk.record_order()
                close = float(feats.get("close", self.Securities[sym].Price))
                self.position_risk[ticker] = PositionRisk(
                    entry_price=close,
                    entry_time=now,
                    entry_atr=float(feats.get("atr", close * 0.02)),
                    highest_close=close,
                    predicted_score=sc,
                )

        self.Debug(
            f"KRAKEN_MAX rebalance regime={regime.name} targets={targets} breadth={breadth:.2f}"
        )

    def _maybe_pyramid(self, ticker: str, sym, feats: dict, score: float, equity: float) -> None:
        state = self.position_risk.get(ticker)
        if state is None or state.pyramid_count >= 1:
            return
        close = float(feats.get("close", self.Securities[sym].Price))
        pnl = (close / state.entry_price) - 1.0 if state.entry_price > 0 else 0.0
        if pnl < float(self.config.pyramid_min_unrealized_pct):
            return
        if score < float(self.config.entry_score_threshold) + 0.1:
            return
        add_usd = equity * float(self.config.pyramid_add_pct)
        if place_buy(self, sym, add_usd):
            state.pyramid_count += 1
            state.highest_close = max(state.highest_close, close)

    def _manage_positions(self, now) -> None:  # pragma: no cover
        for ticker, sym in self.symbol_by_ticker.items():
            qty = position_qty(self, sym)
            if qty <= 0:
                self.position_risk.pop(ticker, None)
                continue
            state = self.position_risk.get(ticker)
            if state is None:
                px = float(self.Securities[sym].Price)
                state = PositionRisk(
                    entry_price=px,
                    entry_time=now,
                    entry_atr=px * 0.02,
                    highest_close=px,
                )
                self.position_risk[ticker] = state
            close = float(self.Securities[sym].Price)
            if manage_exits(self, sym, state, close, now):
                pnl = (close / state.entry_price) - 1.0 if state.entry_price else 0.0
                self.sizer.record_trade(pnl)
                self.ensemble.ml.online_update(pnl, state.predicted_score)
                self.position_risk.pop(ticker, None)

    def _flatten_all(self, reason: str) -> None:  # pragma: no cover
        for ticker, sym in self.symbol_by_ticker.items():
            if position_qty(self, sym) > 0:
                liquidate_symbol(self, sym)
        self.position_risk.clear()
        self.Debug(f"KRAKEN_MAX flatten reason={reason}")
