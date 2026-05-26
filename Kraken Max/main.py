from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

try:  # pragma: no cover
    from AlgorithmImports import AccountType, BrokerageName, Market, OrderStatus, QCAlgorithm, Resolution
except ImportError:  # pragma: no cover
    class QCAlgorithm:  # type: ignore
        pass

    class Resolution:
        Hour = "Hour"
        Minute = "Minute"

    class OrderStatus:
        Filled = "Filled"
        Canceled = "Canceled"

    class BrokerageName:
        Kraken = "Kraken"

    class AccountType:
        Cash = "Cash"

    class Market:
        Kraken = "Kraken"

from config import CONFIG
from core import (
    AggressiveSizer,
    AlphaEnsemble,
    FeatureCache,
    KRAKEN_MAX_UNIVERSE,
    REFERENCE_SYMBOLS,
    btc_beta_vs,
    build_scalper_features,
    cross_section_ranks,
    evaluate_scalper_entry,
    filter_uncorrelated_picks,
    load_optimized_ensemble_weights,
    select_universe,
)
from data import CrossVenueLead, SentimentDataHub, compute_sentiment, merge_external_sentiment
from execution import (
    escalate_orders,
    init_execution_state,
    liquidate_symbol,
    manage_position_exit,
    place_buy_notional,
    position_qty,
    set_bracket_prices,
    sync_brackets,
)
from kraken_ml import MLScorer, MLTrainer, load_ml_weights
from kraken_ops import (
    AlertManager,
    BaselineManager,
    CalibratedCostModel,
    DriftMonitor,
    FillTracker,
    PaperTradingScorecard,
    TelemetryDashboard,
    build_html_digest,
    build_text_digest,
    load_bundle,
    persist_digest,
)
from regime import RegimeEngine, UnifiedRegimeEngine, load_regime_weights_from_object_store
from risk import (
    PortfolioRisk,
    PositionRisk,
    allocate_erc_notionals,
    filter_cluster_caps,
    max_cluster_exposure,
)
from workflow import AutoRevalidator, consolidate_minute_ohlcv


class KrakenMaxAlgorithm(QCAlgorithm):
    """Kraken Max v8 — regime WF, auto revalidation, dashboard digest, native 15m export path."""

    def Initialize(self):  # pragma: no cover
        self.config = CONFIG
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)
        self.long_only = True
        self.SetCash(float(self.config.starting_cash))
        self.SetStartDate(self.config.start_year, self.config.start_month, self.config.start_day)
        self.SetEndDate(self.config.end_year, self.config.end_month, self.config.end_day)
        res = Resolution.Minute if bool(self.config.use_sub_hour_bars) else Resolution.Hour
        warmup = (
            int(self.config.warmup_bars_sub_hour)
            if bool(self.config.use_sub_hour_bars)
            else int(self.config.warmup_bars)
        )
        if bool(self.config.fast_qc_backtest) and not getattr(self, "LiveMode", False):
            sy, sm, sd = self.config.fast_qc_start
            ey, em, ed = self.config.fast_qc_end
            self.SetStartDate(int(sy), int(sm), int(sd))
            self.SetEndDate(int(ey), int(em), int(ed))
            warmup = int(self.config.warmup_bars_fast)
            self.Debug(f"KRAKEN_MAX fast_qc_backtest {sy}-{sm}-{sd} → {ey}-{em}-{ed} warmup={warmup}")
        self.SetWarmup(warmup, res)

        init_execution_state(self)
        self.feature_cache = FeatureCache()
        self.sentiment_hub = SentimentDataHub(self.config)
        self.sentiment_hub.initialize_algorithm(self)
        self.ml_trainer = MLTrainer(self.config)
        if bool(self.config.use_advanced_regime):
            self.regime_engine = UnifiedRegimeEngine(self.config)
        else:
            self.regime_engine = RegimeEngine(self.config)
        self.ensemble = AlphaEnsemble(self.config, MLScorer(load_ml_weights()), algo=self)
        self.revalidator = AutoRevalidator(self.config) if bool(self.config.enable_auto_revalidation) else None
        self.sizer = AggressiveSizer(self.config)
        self.alerts = AlertManager(self) if bool(self.config.enable_live_alerts) else None
        self.fill_tracker = FillTracker(self.config) if bool(self.config.enable_fill_tracking) else None
        self.drift_monitor = DriftMonitor(self.config) if bool(self.config.enable_drift_monitor) else None
        self.telemetry = TelemetryDashboard(self.config) if bool(self.config.enable_telemetry) else None
        self.cross_venue = CrossVenueLead(self.config) if bool(self.config.use_cross_venue_lead) else None
        self.baseline_mgr = BaselineManager(self.config)
        self.cost_model = CalibratedCostModel(self.config) if bool(self.config.use_calibrated_costs) else None
        self.scorecard = PaperTradingScorecard(self.config) if bool(self.config.enable_scorecard) else None
        self._last_telemetry = None
        self._last_regime_state = ("", "", 0.0)
        self.portfolio_risk = PortfolioRisk(self.config)
        self.position_risk: dict = {}
        self.symbol_by_ticker: dict[str, object] = {}
        self._sym_to_ticker: dict[object, str] = {}
        self.active_universe: list[str] = []
        self._last_rebalance = None
        self._last_scalper = None
        self._last_trade_hours: dict[str, float] = {}
        self._erc_weights: dict[str, float] = {}
        self._bar_count = 0

        opt = load_optimized_ensemble_weights()
        if opt:
            self.Debug(f"KRAKEN_MAX ensemble_weights loaded {opt}")
        if self.drift_monitor is not None:
            self.drift_monitor.load_baseline_from_object_store(self)
            try:
                import json

                ew = Path(__file__).resolve().parent / "ensemble_weights.json"
                if ew.is_file():
                    blob = json.loads(ew.read_text(encoding="utf-8"))
                    sharpe = float((blob.get("metrics") or {}).get("oos_sharpe", 0.0))
                    if sharpe > 0:
                        self.drift_monitor.baseline_sharpe = sharpe
            except Exception:
                pass

        if bool(self.config.subscribe_all_universe_on_init):
            boot = set(KRAKEN_MAX_UNIVERSE) | set(REFERENCE_SYMBOLS)
        else:
            boot = set(self.config.seed_subscribe_symbols) | set(REFERENCE_SYMBOLS)
        for ticker in sorted(boot):
            self._subscribe(ticker)
        if not self.symbol_by_ticker:
            raise RuntimeError(
                "KRAKEN_MAX: no Kraken symbols subscribed — enable Crypto Price data (Kraken) "
                "and check tickers in config.seed_subscribe_symbols."
            )
        self.Debug(
            f"KRAKEN_MAX subscribed {len(self.symbol_by_ticker)} symbols "
            f"(all_universe={self.config.subscribe_all_universe_on_init}) warmup={warmup} res={res}"
        )

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
        if self.revalidator is not None:
            self.Schedule.On(
                self.DateRules.MonthStart(),
                self.TimeRules.At(1, 0),
                self._scheduled_revalidation,
            )
        if bool(self.config.enable_dashboard_digest):
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.At(int(self.config.dashboard_digest_hour), 0),
                self._scheduled_dashboard_digest,
            )
        if self.telemetry is not None:
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.Every(timedelta(hours=int(self.config.telemetry_cadence_hours))),
                self._persist_telemetry,
            )
        self.Debug(
            f"KRAKEN_MAX v8 init res={self.config.resolution_minutes}m "
            f"reval={self.config.enable_auto_revalidation} dashboard={self.config.enable_dashboard_digest} "
            f"regime_wf={self.config.use_regime_wf_weights}"
        )

    def _ensure_subscribed(self, tickers: list[str]) -> None:
        for ticker in tickers:
            if ticker not in self.symbol_by_ticker:
                self._subscribe(ticker)

    def _subscribe(self, ticker: str) -> None:
        if ticker in self.symbol_by_ticker:
            return
        try:
            res = Resolution.Minute if bool(self.config.use_sub_hour_bars) else Resolution.Hour
            sec = self.AddCrypto(ticker, res, Market.Kraken)
            sym = sec.Symbol
            self.symbol_by_ticker[ticker] = sym
            self._sym_to_ticker[sym] = ticker
            if bool(self.config.use_sub_hour_bars):
                self.Consolidate(
                    sym,
                    timedelta(minutes=int(self.config.resolution_minutes)),
                    lambda bar, s=sym: self._on_consolidated_bar(s, bar),
                )
        except Exception as exc:
            self.Debug(f"KRAKEN_MAX skip subscribe {ticker}: {exc}")

    def _on_consolidated_bar(self, sym, bar) -> None:  # pragma: no cover
        ticker = self._sym_to_ticker.get(sym, getattr(sym, "Value", str(sym)))
        self.feature_cache.update(ticker, bar)
        if ticker == "BTCUSD" and isinstance(self.regime_engine, UnifiedRegimeEngine):
            close = float(bar.Close)
            feats = self.feature_cache.features("BTCUSD")
            ema200 = float(feats.get("ema200", close))
            btc_ret = float(feats.get("ret_1h", 0.0))
            rv = float(feats.get("rv_21d", 0.2))
            self.regime_engine.update_market(
                btc_close=close,
                btc_return=btc_ret,
                btc_vol=rv,
                breadth=0.5,
                btc_above_ema200=close > ema200,
                ema200=ema200,
            )

    def OnData(self, slice) -> None:  # pragma: no cover
        if not self.IsWarmingUp and not getattr(self, "_logged_warmup_done", False):
            self._logged_warmup_done = True
            self.Debug(
                f"KRAKEN_MAX warmup done — trading from {self.Time} "
                f"symbols={len(self.symbol_by_ticker)} equity={self.Portfolio.TotalPortfolioValue:.2f}"
            )

        if not self.IsWarmingUp:
            self._progress_bars = int(getattr(self, "_progress_bars", 0)) + 1
            if self._progress_bars in (1, 24, 168):
                self.Debug(
                    f"KRAKEN_MAX progress bars={self._progress_bars} time={self.Time} "
                    f"equity={self.Portfolio.TotalPortfolioValue:.2f}"
                )

        if not bool(self.config.use_sub_hour_bars):
            self._bar_count += 1
            for ticker, sym in self.symbol_by_ticker.items():
                if sym not in slice.Bars:
                    continue
                self.feature_cache.update(ticker, slice.Bars[sym])
                if ticker == "BTCUSD" and isinstance(self.regime_engine, UnifiedRegimeEngine):
                    close = float(slice.Bars[sym].Close)
                    feats = self.feature_cache.features("BTCUSD")
                    self.regime_engine.update_market(
                        btc_close=close,
                        btc_return=float(feats.get("ret_1h", 0.0)),
                        btc_vol=float(feats.get("rv_21d", 0.2)),
                        breadth=0.5,
                        btc_above_ema200=close > float(feats.get("ema200", close)),
                        ema200=float(feats.get("ema200", close)),
                    )

        if self.IsWarmingUp:
            return

        escalate_orders(self)
        now = self.Time
        if self.drift_monitor is not None:
            self.drift_monitor.record_equity(now, float(self.Portfolio.TotalPortfolioValue))
        if self.scorecard is not None:
            self.scorecard.record_equity(now, float(self.Portfolio.TotalPortfolioValue))
        if self._last_rebalance is None or (now - self._last_rebalance) >= timedelta(
            hours=int(self.config.rebalance_hours)
        ):
            self._rebalance(now, slice)
            self._last_rebalance = now

        if bool(self.config.enable_scalper) and (
            self._last_scalper is None
            or (now - self._last_scalper) >= timedelta(hours=int(self.config.scalper_cadence_hours))
        ):
            self._scalper_pass(now)
            self._last_scalper = now

        self._sync_brackets_all()
        self._manage_positions(now)

    def _sync_brackets_all(self) -> None:
        if not bool(self.config.enable_brackets):
            return
        for ticker, sym in self.symbol_by_ticker.items():
            st = self.position_risk.get(ticker)
            qty = position_qty(self, sym)
            if st and qty > 0 and st.strategy_owner == "momentum":
                sync_brackets(self, sym, st, qty)

    def _hourly_maintenance(self) -> None:  # pragma: no cover
        equity = float(self.Portfolio.TotalPortfolioValue)
        dd = self.portfolio_risk.update_peak(equity)
        if self.portfolio_risk.drawdown_halted(self.Time, dd):
            if self.alerts and bool(self.config.alert_on_drawdown_halt):
                self.alerts.notify(
                    "DRAWDOWN_HALT",
                    f"dd={dd:.2%} equity={equity:.2f}",
                    dedupe_key="drawdown_halt",
                )
            self._flatten_all("drawdown_halt")
            return
        if self.fill_tracker is not None:
            alert, msg = self.fill_tracker.should_alert()
            if alert and self.alerts and bool(self.config.enable_live_alerts):
                self.alerts.notify("FILL_QUALITY", msg, dedupe_key="fill_quality")
        if self.drift_monitor is not None:
            alert, msg = self.drift_monitor.should_alert()
            if alert and self.alerts and bool(self.config.alert_on_drift):
                self.alerts.notify("DRIFT", msg, dedupe_key="drift")
        if self.ml_trainer.should_retrain(self.Time):
            self._run_ml_retrain()
        if self.scorecard is not None:
            snap = self.scorecard.build(self, fill_tracker=self.fill_tracker)
            self.scorecard.persist(self, snap)
            ok, reason = self.scorecard.passes_paper_gate(snap)
            if not ok and self.alerts and bool(self.config.alert_on_paper_gate_fail):
                self.alerts.notify("PAPER_GATE", reason, dedupe_key="paper_gate")
            elif ok and snap.days_tracked >= float(self.config.paper_min_days):
                self.Debug(self.scorecard.summary_line(snap))
        self._maybe_persist_telemetry()

    def _maybe_persist_telemetry(self) -> None:  # pragma: no cover
        if self.telemetry is None:
            return
        cadence = timedelta(hours=int(self.config.telemetry_cadence_hours))
        now = self.Time
        if self._last_telemetry is not None and (now - self._last_telemetry) < cadence:
            return
        self._persist_telemetry()

    def _persist_telemetry(self) -> None:  # pragma: no cover
        if self.telemetry is None:
            return
        reg, micro, cap = self._last_regime_state
        xb = 0.0
        if self.cross_venue is not None and self.active_universe:
            xb = self.cross_venue.aggregate_boost(self.active_universe[:8], self.Time)
        snap = self.telemetry.build(
            self,
            regime_name=reg,
            micro_regime=micro,
            deployment_cap=cap,
            cross_venue_boost=xb,
        )
        self.telemetry.persist(self, snap)
        self._last_telemetry = self.Time
        self.Debug(self.telemetry.summary_line(snap))

    def OnOrderEvent(self, order_event) -> None:  # pragma: no cover
        if self.fill_tracker is None:
            return
        status = str(getattr(order_event, "Status", ""))
        oid = int(getattr(order_event, "OrderId", 0) or 0)
        if "Canceled" in status:
            self.fill_tracker.on_cancel(oid)
        if "Filled" not in status:
            return
        fill_px = float(getattr(order_event, "FillPrice", 0.0) or 0.0)
        tag = str(getattr(order_event, "Message", "") or "")
        is_limit = "Limit" in tag or getattr(order_event, "OrderType", "") == "Limit"
        self.fill_tracker.on_fill(oid, fill_px, is_limit=is_limit, tag=tag)

    def _scheduled_ml_retrain(self) -> None:  # pragma: no cover
        self._run_ml_retrain()

    def _scheduled_revalidation(self) -> None:  # pragma: no cover
        if self.revalidator is None or self.IsWarmingUp:
            return
        if not self.revalidator.should_run(self.Time):
            return
        bars = self._export_history_bars()
        if bars is None or bars.empty:
            self.Debug("KRAKEN_MAX revalidation skipped: no history")
            return
        try:
            result = self.revalidator.run(bars, self, n_folds=int(self.config.auto_revalidate_folds))
            stored = load_regime_weights_from_object_store(self)
            if stored:
                self.ensemble._regime_weights = stored
            status = "PASS" if result.get("validation_passed") else "FAIL"
            self.Debug(f"KRAKEN_MAX revalidation {status} sharpe={result.get('oos_sharpe', 0):.3f}")
            if self.alerts:
                self.alerts.notify(
                    "REVALIDATION",
                    f"{status} sharpe={result.get('oos_sharpe', 0):.3f}",
                    dedupe_key=f"reval-{self.Time.date()}",
                )
        except Exception as exc:
            self.Debug(f"KRAKEN_MAX revalidation error: {exc}")

    def _scheduled_dashboard_digest(self) -> None:  # pragma: no cover
        if self.IsWarmingUp:
            return
        bundle = load_bundle(self, self.config)
        text = build_text_digest(bundle)
        html = build_html_digest(bundle)
        persist_digest(self, text, html, self.config)
        if self.alerts and bool(self.config.alert_on_dashboard):
            self.alerts.notify("DASHBOARD", text[:1800], dedupe_key=f"dash-{self.Time.date()}")

    def _export_history_bars(self) -> pd.DataFrame:  # pragma: no cover
        """Build OHLCV panel from QC History for revalidation (hourly if sub-hour live)."""
        lookback = int(self.config.auto_revalidate_lookback_days)
        end = self.Time
        start = end - timedelta(days=lookback)
        res = Resolution.Hour
        if bool(self.config.use_sub_hour_bars):
            res = Resolution.Minute
        rows: list[dict] = []
        tickers = list(self.active_universe or ["BTCUSD", "ETHUSD", "SOLUSD"])[:12]
        for ticker in tickers:
            sym = self.symbol_by_ticker.get(ticker)
            if sym is None:
                continue
            try:
                hist = self.History(sym, start, end, res)
            except Exception:
                continue
            if hist is None or hist.empty:
                continue
            h = hist.reset_index() if isinstance(hist.index, pd.MultiIndex) else hist
            cols = {str(c).lower(): c for c in h.columns}
            sym_col = cols.get("symbol", None)
            time_col = cols.get("time", cols.get("endtime", h.columns[0]))
            for _, row in h.iterrows():
                rows.append(
                    {
                        "symbol": ticker,
                        "timestamp": pd.Timestamp(row[time_col], tz="UTC"),
                        "open": float(row[cols.get("open", "open")]),
                        "high": float(row[cols.get("high", "high")]),
                        "low": float(row[cols.get("low", "low")]),
                        "close": float(row[cols.get("close", "close")]),
                        "volume": float(row.get(cols.get("volume", "volume"), 1000.0) or 1000.0),
                    }
                )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if bool(self.config.use_sub_hour_bars) and int(self.config.resolution_minutes) < 60:
            return consolidate_minute_ohlcv(df, int(self.config.resolution_minutes))
        return df.sort_values(["symbol", "timestamp"])

    def _run_ml_retrain(self) -> None:  # pragma: no cover
        if len(self.ml_trainer.samples) < int(self.config.ml_min_samples):
            return
        blob = self.ml_trainer.retrain(self.ensemble.ml, self.Time)
        self.ml_trainer.try_persist_object_store(self, blob)
        acc = float(blob.get("train_accuracy", 0.0) or 0.0)
        pseudo_sharpe = max(0.05, (acc - 0.5) * 2.0)
        self.baseline_mgr.refresh_from_sharpe(self, pseudo_sharpe, source="ml_retrain", drift=self.drift_monitor)
        self.Debug(f"KRAKEN_MAX ml_retrain samples={blob.get('trained_samples', 0)} acc={acc:.3f}")

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

    def _build_sentiment(self, feature_map: dict, slice) -> object:
        rv_vals = [float(f.get("rv_21d", 0.05)) for f in feature_map.values()]
        median_rv = sorted(rv_vals)[len(rv_vals) // 2] if rv_vals else 0.05
        breadth = sum(1 for f in feature_map.values() if float(f.get("mom_7d", 0)) > 0) / len(feature_map)
        alt_mom = sorted(float(f.get("mom_7d", 0)) for f in feature_map.values())
        alt_med = alt_mom[len(alt_mom) // 2] if alt_mom else 0.0
        btc_feat = feature_map.get("BTCUSD") or self.feature_cache.features("BTCUSD") or {}
        eth_feat = feature_map.get("ETHUSD") or self.feature_cache.features("ETHUSD") or {}
        proxy = compute_sentiment(
            btc_features=btc_feat,
            eth_features=eth_feat,
            breadth=breadth,
            median_rv=median_rv,
            alt_median_mom_7d=alt_med,
        )
        if not bool(self.config.use_external_sentiment):
            return proxy
        external = self.sentiment_hub.update_from_slice(self, slice, feature_map)
        return merge_external_sentiment(proxy, external)

    def _rebalance(self, now, slice) -> None:  # pragma: no cover
        if self.portfolio_risk.drawdown_halted(now, 0.0):
            return

        self.active_universe = select_universe(
            self._history_provider,
            now,
            candidates=tuple(self.symbol_by_ticker.keys()),
        )
        self._ensure_subscribed(self.active_universe)
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
        sentiment = self._build_sentiment(feature_map, slice)
        ema200 = float(btc_feat.get("ema200", 0.0))
        btc_above = float(btc_feat.get("close", 0.0)) > ema200 if ema200 > 0 else True
        if isinstance(self.regime_engine, UnifiedRegimeEngine):
            regime = self.regime_engine.classify_unified(
                btc_features=btc_feat,
                breadth=breadth,
                median_rv=median_rv,
                sentiment=sentiment,
                btc_return=float(btc_feat.get("ret_1h", 0.0)),
                btc_vol=median_rv,
                btc_above_ema200=btc_above,
            )
        else:
            regime = self.regime_engine.classify(
                btc_features=btc_feat,
                breadth=breadth,
                median_rv=median_rv,
                sentiment=sentiment,
            )
        self._last_regime_state = (regime.name, str(regime.micro_regime), float(regime.deployment_cap))

        if not regime.allow_new_entries or regime.deployment_cap <= 0:
            self._flatten_momentum_only("regime_chaos")
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
                btc_beta=btc_beta_vs(feats, btc_feat),
                regime_name=regime.name,
            )
            final = float(comp["final"])
            if self.cross_venue is not None:
                final += self.cross_venue.score_adjustment(ticker, now)
            scores.append((ticker, final, comp))
        scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [(t, s) for t, s, _ in scores if s >= float(self.config.entry_score_threshold)]
        decorrelated = filter_uncorrelated_picks(
            candidates, self.feature_cache, top_k=int(self.config.top_k) * 2
        )
        if bool(self.config.enable_cluster_risk):
            holdings = [
                t
                for t, st in self.position_risk.items()
                if st.strategy_owner == "momentum"
                and (sym := self.symbol_by_ticker.get(t)) is not None
                and position_qty(self, sym) > 0
            ]
            score_by_ticker = {t: s for t, s, _ in scores}
            ranked = [(t, score_by_ticker.get(t, 0.0)) for t in decorrelated]
            targets = filter_cluster_caps(
                ranked,
                current_holdings=holdings,
                config=self.config,
            )[: int(self.config.top_k)]
        else:
            targets = decorrelated[: int(self.config.top_k)]

        equity = float(self.Portfolio.TotalPortfolioValue)
        if bool(self.config.use_erc_sizing) and targets:
            slot_notionals = allocate_erc_notionals(
                targets,
                self.feature_cache,
                equity,
                regime.deployment_cap,
                config=self.config,
                previous_weights=self._erc_weights,
            )
            total = sum(slot_notionals.values()) or 1.0
            self._erc_weights = {t: slot_notionals.get(t, 0.0) / total for t in targets}
        else:
            deployable = equity * regime.deployment_cap
            per = deployable / max(len(targets), 1)
            slot_notionals = {t: per for t in targets}

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
            weight = self.sizer.weight_for_score(sc, float(feats.get("rv_21d", 0.2)), rank_mom.get(ticker, 0.5))
            notional = min(slot_notionals.get(ticker, 0.0), equity * weight)
            if not self.sizer.passes_cost_gate(sc, notional, self):
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
                atr = float(feats.get("atr", close * 0.02))
                state = PositionRisk(
                    entry_price=close,
                    entry_time=now,
                    entry_atr=atr,
                    highest_close=close,
                    predicted_score=sc,
                    strategy_owner="momentum",
                )
                set_bracket_prices(state, close, atr)
                self.position_risk[ticker] = state
                qty = position_qty(self, sym)
                if qty > 0:
                    sync_brackets(self, sym, state, qty)
                self._last_trade_hours[ticker] = 0.0

        msg = (
            f"KRAKEN_MAX rebalance regime={regime.name} micro={regime.micro_regime} "
            f"fg={sentiment.fear_greed:.2f} targets={targets}"
        )
        self.Debug(msg)
        if self.alerts and bool(self.config.alert_on_rebalance):
            self.alerts.notify("REBALANCE", msg, dedupe_key=f"rebal-{now.date()}")

    def _scalper_pass(self, now) -> None:  # pragma: no cover
        if self.portfolio_risk.drawdown_halted(now, 0.0):
            return
        feature_map = {
            t: self.feature_cache.features(t)
            for t in (self.active_universe or list(self.symbol_by_ticker)[:15])
        }
        feature_map = {k: v for k, v in feature_map.items() if v}
        btc_feat = feature_map.get("BTCUSD") or {}
        rv_vals = [float(f.get("rv_21d", 0.05)) for f in feature_map.values()]
        median_rv = sorted(rv_vals)[len(rv_vals) // 2] if rv_vals else 0.05
        breadth = sum(1 for f in feature_map.values() if float(f.get("mom_7d", 0)) > 0) / max(len(feature_map), 1)
        sentiment = compute_sentiment(
            btc_features=btc_feat,
            eth_features=feature_map.get("ETHUSD"),
            breadth=breadth,
            median_rv=median_rv,
        )
        ema200 = float(btc_feat.get("ema200", 0.0))
        btc_above = float(btc_feat.get("close", 0.0)) > ema200 if ema200 > 0 else True
        if isinstance(self.regime_engine, UnifiedRegimeEngine):
            regime = self.regime_engine.classify_unified(
                btc_features=btc_feat,
                breadth=breadth,
                median_rv=median_rv,
                sentiment=sentiment,
                btc_return=float(btc_feat.get("ret_1h", 0.0)),
                btc_vol=median_rv,
                btc_above_ema200=btc_above,
            )
        else:
            regime = self.regime_engine.classify(
                btc_features=btc_feat, breadth=breadth, median_rv=median_rv, sentiment=sentiment
            )
        if not regime.allow_scalper:
            return

        btc_1h, btc_6h = self._btc_returns()
        scalper_open = sum(
            1
            for t, st in self.position_risk.items()
            if st.strategy_owner == "scalper"
            and (sym := self.symbol_by_ticker.get(t)) is not None
            and position_qty(self, sym) > 0
        )
        if scalper_open >= int(self.config.scalper_max_positions):
            return

        ranked: list[tuple[str, float]] = []
        for ticker in self.active_universe or []:
            s_feats = build_scalper_features(self.feature_cache.frame(ticker))
            if s_feats:
                ranked.append((ticker, -float(s_feats.get("z_20h", 0.0))))
        ranked.sort(key=lambda x: x[1], reverse=True)
        equity = float(self.Portfolio.TotalPortfolioValue)
        notional = equity * float(self.config.scalper_position_pct)

        for ticker, _ in ranked[:8]:
            if scalper_open >= int(self.config.scalper_max_positions):
                break
            sym = self.symbol_by_ticker.get(ticker)
            if sym is None or position_qty(self, sym) > 0:
                continue
            s_feats = build_scalper_features(self.feature_cache.frame(ticker))
            ok, reason = evaluate_scalper_entry(
                s_feats,
                btc_ret_1h=btc_1h,
                btc_ret_6h=btc_6h,
                last_trade_hours=float(self._last_trade_hours.get(ticker, 999.0)),
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
                scalper_open += 1
                self.Debug(f"KRAKEN_MAX scalper {ticker} {reason}")

    def _maybe_pyramid(self, ticker: str, sym, feats: dict, score: float, equity: float) -> None:
        state = self.position_risk.get(ticker)
        if state is None or state.pyramid_count >= 1:
            return
        close = float(feats.get("close", self.Securities[sym].Price))
        pnl = (close / state.entry_price) - 1.0 if state.entry_price > 0 else 0.0
        if pnl < float(self.config.pyramid_min_unrealized_pct) or score < float(self.config.entry_score_threshold) + 0.1:
            return
        if place_buy_notional(self, sym, equity * float(self.config.pyramid_add_pct), tag="KM:Pyramid"):
            state.pyramid_count += 1
            state.highest_close = max(state.highest_close, close)

    def _record_exit(self, ticker: str, state: PositionRisk, sym) -> None:
        close = float(self.Securities[sym].Price)
        pnl = (close / state.entry_price) - 1.0 if state.entry_price else 0.0
        self.sizer.record_trade(pnl)
        if self.scorecard is not None:
            self.scorecard.record_trade(
                ticker, pnl, strategy=str(state.strategy_owner), when=self.Time
            )
        self.ensemble.ml.online_update(pnl, state.predicted_score)
        feats = self.feature_cache.features(ticker)
        ctx = self.sentiment_hub.to_context() if hasattr(self, "sentiment_hub") else {}
        self.ml_trainer.add_closed_trade(feats, ctx, pnl)
        self.position_risk.pop(ticker, None)

    def _manage_positions(self, now) -> None:  # pragma: no cover
        for ticker, sym in self.symbol_by_ticker.items():
            if position_qty(self, sym) <= 0:
                self.position_risk.pop(ticker, None)
                continue
            state = self.position_risk.get(ticker)
            if state is None:
                px = float(self.Securities[sym].Price)
                state = PositionRisk(entry_price=px, entry_time=now, entry_atr=px * 0.02, highest_close=px)
                self.position_risk[ticker] = state
            close = float(self.Securities[sym].Price)
            state.highest_close = max(state.highest_close, close)
            feats = build_scalper_features(self.feature_cache.frame(ticker)) if state.strategy_owner == "scalper" else {}
            if manage_position_exit(self, sym, state, close, now, feats or None):
                self._record_exit(ticker, state, sym)

    def _flatten_momentum_only(self, reason: str) -> None:
        for ticker, sym in self.symbol_by_ticker.items():
            st = self.position_risk.get(ticker)
            if position_qty(self, sym) > 0 and (st is None or st.strategy_owner == "momentum"):
                liquidate_symbol(self, sym)
                if st:
                    self._record_exit(ticker, st, sym)
        self.Debug(f"KRAKEN_MAX flatten_momentum {reason}")

    def _flatten_all(self, reason: str) -> None:
        for ticker, sym in self.symbol_by_ticker.items():
            if position_qty(self, sym) > 0:
                st = self.position_risk.get(ticker)
                liquidate_symbol(self, sym)
                if st:
                    self._record_exit(ticker, st, sym)
        self.position_risk.clear()
        self.Debug(f"KRAKEN_MAX flatten_all {reason}")
