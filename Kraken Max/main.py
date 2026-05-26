from __future__ import annotations

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
from correlation import filter_uncorrelated_picks
from ensemble import AlphaEnsemble
from execution import (
    escalate_stale_limits,
    liquidate_symbol,
    manage_exits,
    place_buy_notional,
    position_qty,
)
from features import FeatureCache, btc_beta_vs, cross_section_ranks
from ml_scorer import MLScorer, load_ml_weights
from ml_trainer import MLTrainer
from regime import RegimeEngine
from risk import PortfolioRisk, PositionRisk
from scalper_sleeve import build_scalper_features, evaluate_scalper_entry
from sentiment import compute_sentiment
from sizing import AggressiveSizer
from universe import KRAKEN_MAX_UNIVERSE, REFERENCE_SYMBOLS, select_universe


class KrakenMaxAlgorithm(QCAlgorithm):
    """Kraken Max v2 — limit execution, decorrelation, sentiment regime, ML retrain, scalper."""

    def Initialize(self):  # pragma: no cover
        self.config = CONFIG
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
        self.long_only = True
        self.SetCash(float(self.config.starting_cash))
        self.SetStartDate(self.config.start_year, self.config.start_month, self.config.start_day)
        self.SetEndDate(self.config.end_year, self.config.end_month, self.config.end_day)
        self.SetWarmup(self.config.warmup_bars, Resolution.Hour)

        blob = load_ml_weights()
        self.feature_cache = FeatureCache()
        self.ml_trainer = MLTrainer(self.config)
        self.regime_engine = RegimeEngine(self.config)
        self.ensemble = AlphaEnsemble(self.config, MLScorer(blob))
        self.sizer = AggressiveSizer(self.config)
        self.portfolio_risk = PortfolioRisk(self.config)
        self.position_risk: dict = {}
        self.symbol_by_ticker: dict[str, object] = {}
        self.active_universe: list[str] = []
        self._last_rebalance = None
        self._last_scalper = None
        self._last_trade_hours: dict[str, float] = {}
        self._bar_count = 0
        self._pending_limits = {}

        for ticker in set(KRAKEN_MAX_UNIVERSE) | set(REFERENCE_SYMBOLS):
            self._subscribe(ticker)

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(hours=1)),
            self._hourly_maintenance,
        )
        self.Schedule.On(
            self.DateRules.MonthStart(),
            self.TimeRules.At(0, 15),
            self._scheduled_ml_retrain,
        )
        self.Debug("KRAKEN_MAX v2 init — limits, corr filter, sentiment, ML retrain, scalper")

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

        escalate_stale_limits(self)
        now = self.Time
        if self._last_rebalance is None or (now - self._last_rebalance) >= timedelta(
            hours=int(self.config.rebalance_hours)
        ):
            self._rebalance(now)
            self._last_rebalance = now

        if bool(self.config.enable_scalper) and (
            self._last_scalper is None
            or (now - self._last_scalper) >= timedelta(hours=int(self.config.scalper_cadence_hours))
        ):
            self._scalper_pass(now)
            self._last_scalper = now

        self._manage_positions(now)

    def _hourly_maintenance(self) -> None:  # pragma: no cover
        equity = float(self.Portfolio.TotalPortfolioValue)
        dd = self.portfolio_risk.update_peak(equity)
        if self.portfolio_risk.drawdown_halted(self.Time, dd):
            self._flatten_all("drawdown_halt")
            return
        if self.ml_trainer.should_retrain(self.Time):
            self._run_ml_retrain()

    def _scheduled_ml_retrain(self) -> None:  # pragma: no cover
        self._run_ml_retrain()

    def _run_ml_retrain(self) -> None:  # pragma: no cover
        if len(self.ml_trainer.samples) < int(self.config.ml_min_samples):
            return
        blob = self.ml_trainer.retrain(self.ensemble.ml, self.Time)
        self.ml_trainer.try_persist_object_store(self, blob)
        self.Debug(f"KRAKEN_MAX ml_retrain samples={blob.get('trained_samples', 0)}")

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
        return pd.DataFrame(
            {
                "open": hist[cols.get("open", "open")],
                "high": hist[cols.get("high", "high")],
                "low": hist[cols.get("low", "low")],
                "close": hist[cols.get("close", "close")],
                "volume": hist[cols.get("volume", "volume")],
            }
        ).dropna()

    def _btc_returns(self) -> tuple[float, float]:
        btc = self.feature_cache.features("BTCUSD")
        return float(btc.get("ret_1h", 0.0)), float(btc.get("ret_6h", 0.0))

    def _rebalance(self, now) -> None:  # pragma: no cover
        if self.portfolio_risk.drawdown_halted(now, 0.0):
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
        alt_mom = sorted(float(f.get("mom_7d", 0)) for f in feature_map.values() if f)
        alt_med = alt_mom[len(alt_mom) // 2] if alt_mom else 0.0

        btc_feat = feature_map.get("BTCUSD") or self.feature_cache.features("BTCUSD") or {}
        eth_feat = feature_map.get("ETHUSD") or self.feature_cache.features("ETHUSD") or {}
        sentiment = compute_sentiment(
            btc_features=btc_feat,
            eth_features=eth_feat,
            breadth=breadth,
            median_rv=median_rv,
            alt_median_mom_7d=alt_med,
        )
        regime = self.regime_engine.classify(
            btc_features=btc_feat,
            breadth=breadth,
            median_rv=median_rv,
            sentiment=sentiment,
        )
        if not regime.allow_new_entries or regime.deployment_cap <= 0:
            self._flatten_momentum_only("regime_chaos")
            return

        scores: list[tuple[str, float, dict]] = []
        for ticker, feats in feature_map.items():
            if regime.prefer_symbols and ticker not in regime.prefer_symbols and regime.name == "bear":
                continue
            beta = btc_beta_vs(feats, btc_feat)
            comp = self.ensemble.score_symbol(
                feats,
                rank_mom_21=rank_mom.get(ticker, 0.5),
                rank_breakout=rank_bo.get(ticker, 0.5),
                breadth=breadth,
                btc_beta=beta,
            )
            scores.append((ticker, float(comp["final"]), comp))
        scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [(t, s) for t, s, _ in scores if s >= float(self.config.entry_score_threshold)]
        targets = filter_uncorrelated_picks(candidates, self.feature_cache, top_k=int(self.config.top_k))

        equity = float(self.Portfolio.TotalPortfolioValue)
        deployable = equity * regime.deployment_cap
        per_slot = deployable / max(len(targets), 1)

        for ticker, sym in self.symbol_by_ticker.items():
            st = self.position_risk.get(ticker)
            if position_qty(self, sym) <= 0:
                continue
            if st and st.strategy_owner == "momentum" and ticker not in targets:
                liquidate_symbol(self, sym)
                self._record_exit(ticker, st, sym)

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

            st = self.position_risk.get(ticker)
            if position_qty(self, sym) > 0 and st and st.strategy_owner == "momentum":
                self._maybe_pyramid(ticker, sym, feats, sc, equity)
                continue
            if position_qty(self, sym) > 0:
                continue
            if not self.portfolio_risk.can_place_order(now):
                break
            if place_buy_notional(self, sym, notional, tag="KM:Momentum"):
                self.portfolio_risk.record_order()
                close = float(feats.get("close", self.Securities[sym].Price))
                self.position_risk[ticker] = PositionRisk(
                    entry_price=close,
                    entry_time=now,
                    entry_atr=float(feats.get("atr", close * 0.02)),
                    highest_close=close,
                    predicted_score=sc,
                    strategy_owner="momentum",
                )
                self._last_trade_hours[ticker] = 0.0

        self.Debug(
            f"KRAKEN_MAX rebalance regime={regime.name} fg={sentiment.fear_greed:.2f} "
            f"dom={sentiment.btc_dominance:.2f} targets={targets}"
        )

    def _scalper_pass(self, now) -> None:  # pragma: no cover
        if self.portfolio_risk.drawdown_halted(now, 0.0):
            return
        btc_feat = self.feature_cache.features("BTCUSD")
        eth_feat = self.feature_cache.features("ETHUSD")
        feature_map = {
            t: self.feature_cache.features(t)
            for t in self.active_universe or list(self.symbol_by_ticker.keys())[:15]
        }
        rv_vals = [float(f.get("rv_21d", 0.05)) for f in feature_map.values() if f]
        median_rv = sorted(rv_vals)[len(rv_vals) // 2] if rv_vals else 0.05
        breadth = sum(1 for f in feature_map.values() if f and float(f.get("mom_7d", 0)) > 0) / max(
            len(feature_map), 1
        )
        sentiment = compute_sentiment(
            btc_features=btc_feat,
            eth_features=eth_feat,
            breadth=breadth,
            median_rv=median_rv,
        )
        regime = self.regime_engine.classify(
            btc_features=btc_feat,
            breadth=breadth,
            median_rv=median_rv,
            sentiment=sentiment,
        )
        if not regime.allow_scalper:
            return

        btc_1h, btc_6h = self._btc_returns()
        scalper_open = sum(
            1
            for t, st in self.position_risk.items()
            if st.strategy_owner == "scalper" and position_qty(self, self.symbol_by_ticker.get(t, t)) > 0
        )
        if scalper_open >= int(self.config.scalper_max_positions):
            return

        ranked: list[tuple[str, float]] = []
        for ticker in self.active_universe:
            frame = self.feature_cache.frame(ticker)
            s_feats = build_scalper_features(frame)
            if not s_feats:
                continue
            z = float(s_feats.get("z_20h", 0.0))
            ranked.append((ticker, -z))

        ranked.sort(key=lambda x: x[1], reverse=True)
        equity = float(self.Portfolio.TotalPortfolioValue)
        notional = equity * float(self.config.scalper_position_pct)

        for ticker, _ in ranked[:8]:
            if scalper_open >= int(self.config.scalper_max_positions):
                break
            sym = self.symbol_by_ticker.get(ticker)
            if sym is None or position_qty(self, sym) > 0:
                continue
            frame = self.feature_cache.frame(ticker)
            s_feats = build_scalper_features(frame)
            hours = float(self._last_trade_hours.get(ticker, 999.0))
            ok, reason = evaluate_scalper_entry(
                s_feats,
                btc_ret_1h=btc_1h,
                btc_ret_6h=btc_6h,
                last_trade_hours=hours,
            )
            if not ok:
                continue
            if not self.portfolio_risk.can_place_order(now):
                break
            if place_buy_notional(self, sym, notional, tag="KM:Scalper"):
                self.portfolio_risk.record_order()
                close = float(s_feats.get("close", self.Securities[sym].Price))
                self.position_risk[ticker] = PositionRisk(
                    entry_price=close,
                    entry_time=now,
                    entry_atr=float(s_feats.get("atr", close * 0.02)),
                    highest_close=close,
                    strategy_owner="scalper",
                )
                self._last_trade_hours[ticker] = 0.0
                scalper_open += 1
                self.Debug(f"KRAKEN_MAX scalper entry {ticker} reason={reason} z={s_feats.get('z_20h'):.2f}")

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
        if place_buy_notional(self, sym, add_usd, tag="KM:Pyramid"):
            state.pyramid_count += 1
            state.highest_close = max(state.highest_close, close)

    def _record_exit(self, ticker: str, state: PositionRisk, sym) -> None:
        close = float(self.Securities[sym].Price)
        pnl = (close / state.entry_price) - 1.0 if state.entry_price else 0.0
        self.sizer.record_trade(pnl)
        self.ensemble.ml.online_update(pnl, state.predicted_score)
        feats = self.feature_cache.features(ticker)
        ctx = {"breadth": 0.5, "btc_beta": 0.0}
        self.ml_trainer.add_closed_trade(feats, ctx, pnl)
        self.position_risk.pop(ticker, None)
        self._last_trade_hours[ticker] = 0.0

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
            state.highest_close = max(state.highest_close, close)
            feats = self.feature_cache.frame(ticker)
            scalper_feats = build_scalper_features(feats) if state.strategy_owner == "scalper" else {}
            if manage_exits(self, sym, state, close, now, scalper_feats or None):
                self._record_exit(ticker, state, sym)
            else:
                held_h = (now - state.entry_time).total_seconds() / 3600.0
                self._last_trade_hours[ticker] = held_h

    def _flatten_momentum_only(self, reason: str) -> None:
        for ticker, sym in self.symbol_by_ticker.items():
            st = self.position_risk.get(ticker)
            if position_qty(self, sym) > 0 and (st is None or st.strategy_owner == "momentum"):
                liquidate_symbol(self, sym)
                if st:
                    self._record_exit(ticker, st, sym)
        self.Debug(f"KRAKEN_MAX flatten_momentum reason={reason}")

    def _flatten_all(self, reason: str) -> None:
        for ticker, sym in self.symbol_by_ticker.items():
            if position_qty(self, sym) > 0:
                st = self.position_risk.get(ticker)
                liquidate_symbol(self, sym)
                if st:
                    self._record_exit(ticker, st, sym)
        self.position_risk.clear()
        self.Debug(f"KRAKEN_MAX flatten_all reason={reason}")
