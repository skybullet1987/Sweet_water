from AlgorithmImports import *
from execution import *
from reporting import *
from order_management import *
from realistic_slippage import RealisticCryptoSlippage
from events import on_order_event
from scoring import MicroScalpEngine
from strategy_core import (initialize_symbol, update_symbol_data,
                            update_market_context, compute_ranking_overlay)
from trade_quality import (get_session_quality, adverse_selection_filter,
                            record_trade_metadata_on_entry, update_trade_excursion,
                            get_bb_compression_state)
from fee_model import KrakenTieredFeeModel
from regime_router import RegimeRouter
from chop_engine import ChopEngine
from entry_exec import execute_trend_trades, run_chop_rebalance
from collections import deque
import numpy as np
import math
import itertools
# endregion


class SimplifiedCryptoStrategy(QCAlgorithm):

    ALGO_VERSION = "v8.5.0"  # dual-engine regime architecture

    def Initialize(self):
        self.SetStartDate(2025, 1, 1)
        self.SetEndDate(2026, 12, 1)
        # ⚠️ WALK-FORWARD WARNING: backtest runs Jan 2025–Dec 2026 with no held-out OOS window.
        # All parameter tuning should be validated on a separate out-of-sample period before live.
        # Recommended: tune on Jan–Jun 2025, validate on Jul–Dec 2025, deploy Jan 2026+.
        self.SetCash(5000)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        self.entry_threshold = 0.40
        self.high_conviction_threshold = 0.50
        self.min_signal_count = int(self._get_param("min_signal_count", 1))
        self.quick_take_profit = self._get_param("quick_take_profit", 0.015)  # 1.5% for 5-min scalping (was 15%)
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.080)  # 8%: 5% was too tight for 5-min bars
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  4.0)
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  2.0)
        self.trail_activation  = self._get_param("trail_activation",  0.018)
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.012)
        self.time_stop_hours   = self._get_param("time_stop_hours",   8.0)
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.005)
        self.atr_trail_mult             = self._get_param("atr_trail_mult",             4.0)   # was 6.0 — was too loose, gave back too much
        self.post_partial_tp_trail_mult = self._get_param("post_partial_tp_trail_mult", 2.5)   # was 3.5 — tighter after locking profits
        self.min_trail_hold_minutes     = int(self._get_param("min_trail_hold_minutes", 60))

        self.position_size_pct  = 1.0
        self.max_positions      = 6
        self.min_notional       = 5.5
        # NOTE: $5.50 minimum allows many tiny pyramid positions. Each pays full fee overhead.
        # Monitor position count carefully when pyramiding is enabled.
        self.max_position_pct   = self._get_param("max_position_pct", 0.25)  # Lowered: prevents 50% single-position concentration
        # NOTE: At >$5,000 capital, max_position_pct=0.50 creates $2,500+ positions.
        # Market impact becomes significant above ~$2,000 notional on low-cap alts.
        # Consider reducing max_position_pct to 0.25 at higher capital levels.
        self.min_price_usd      = 0.001
        self.cash_reserve_pct   = 0.00
        self.min_notional_fee_buffer = 1.5

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.50)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap", 0.80)
        self.min_asset_vol_floor     = 0.05

        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(12 * 24 * 365)  # 12 five-min bars per hour

        self.max_spread_pct         = 0.005
        self.spread_median_window   = 12
        self.spread_widen_mult      = 2.0
        self.min_dollar_volume_usd  = 15000   # $15K/5-min bar ≈ $4.3M/day; was $100K (=$28.8M/day — killed most universe symbols)
        self.min_volume_usd         = 0  # disabled — min_capacity_usd (500K) is the binding gate; this was redundant
        self.min_capacity_usd       = 500000  # Min 24h USD volume for universe inclusion — capacity filter

        self.skip_hours_utc         = [0, 1]  # 00-01 UTC only: absolute dead zone — reduced from 5h to 2h
        self.max_daily_trades       = 2000
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 0.1
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 5

        self.expected_round_trip_fees = 0.0065  # 0.65%: Kraken blended maker+taker round-trip
        self.fee_slippage_buffer      = 0.002
        self.min_expected_profit_pct  = 0.012
        self.adx_min_period           = 10

        self.stale_order_timeout_seconds      = 30
        self.live_stale_order_timeout_seconds = 60
        self.max_concurrent_open_orders       = 8
        self.open_orders_cash_threshold       = 0.90
        self.order_fill_check_threshold_seconds = 60
        self.order_timeout_seconds              = 30
        self.resync_log_interval_seconds        = 1800
        self.portfolio_mismatch_threshold       = 0.10
        self.portfolio_mismatch_min_dollars     = 1.00
        self.portfolio_mismatch_cooldown_seconds = 3600
        self.retry_pending_cooldown_seconds     = 60
        self.rate_limit_cooldown_minutes        = 10

        self.max_drawdown_limit    = 0.25
        self.cooldown_hours        = 6
        self.consecutive_losses    = 0
        self.max_consecutive_losses = 8
        self._consecutive_loss_halve_remaining = 0
        self.circuit_breaker_expiry = None
        self._circuit_breaker_trigger_count = 0

        self._positions_synced    = False
        self._session_blacklist   = set()
        self._max_session_blacklist_size = 100
        self._symbol_entry_cooldowns = {}
        self._spread_warning_times = {}
        self._first_post_warmup   = True
        self._submitted_orders    = {}
        self._symbol_slippage_history = {}
        self._order_retries       = {}
        self._retry_pending       = {}
        self._rate_limit_until    = None
        self._last_mismatch_warning = None
        self._failed_exit_attempts = {}
        self._failed_exit_counts   = {}
        self._daily_open_value     = None
        self.pnl_by_tag            = {}
        self._entry_signal_combos  = {}
        self.pnl_by_signal_combo   = {}
        self.pnl_by_hold_time      = {}

        self.peak_value       = None
        self.drawdown_cooldown = 0
        self.crypto_data      = {}
        self.entry_prices     = {}
        self.highest_prices   = {}
        self.entry_times      = {}
        self.entry_volumes    = {}
        self._partial_tp_taken      = {}
        self._partial_tp_tier       = {}
        self._partial_sell_symbols  = set()
        self._pyramided_symbols     = set()
        self._choppy_regime_entries = {}
        self.partial_tp_tier1_threshold = self._get_param("partial_tp_tier1_threshold", 0.040)
        self.partial_tp_tier1_pct       = 0.25
        self.partial_tp_tier2_threshold = self._get_param("partial_tp_tier2_threshold", 0.080)  # +8% → sell another 25%
        self.partial_tp_tier2_pct       = 0.25
        self.pyramid_enabled = bool(self._get_param("pyramid_enabled", 1.0))
        self.pyramid_threshold = self._get_param("pyramid_threshold", 0.015)
        self.pyramid_size_pct = self._get_param("pyramid_size_pct", 1.0)
        self.trade_count      = 0
        self._pending_orders  = {}
        self._cancel_cooldowns = {}
        self._exit_cooldowns  = {}
        self._symbol_loss_cooldowns = {}
        self._cash_mode_until = None
        self._recent_trade_outcomes = deque(maxlen=20)
        self.trailing_grace_hours = 1
        self._slip_abs        = deque(maxlen=50)
        self._slippage_alert_until = None
        self.slip_alert_threshold  = 0.0015
        self.slip_outlier_threshold = 0.004
        self.slip_alert_duration_hours = 2
        self._bad_symbol_counts = {}
        self._recent_tickets  = deque(maxlen=25)

        self._rolling_wins      = deque(maxlen=50)
        self._rolling_win_sizes = deque(maxlen=50)
        self._rolling_loss_sizes = deque(maxlen=50)
        self._last_live_trade_time = None

        self.btc_symbol       = None
        self.btc_returns      = deque(maxlen=72)
        self.btc_prices       = deque(maxlen=72)
        self.btc_volatility   = deque(maxlen=72)
        self.btc_ema_24       = ExponentialMovingAverage(24)
        self.market_regime    = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth   = 0.5
        self._regime_hold_count = 0

        # Incremental rolling-stat accumulators for BTC (avoid deque→list→numpy each bar)
        self._btc_sma48_window  = deque(maxlen=48)  # 48-bar price window for regime SMA
        self._btc_sma48_sum     = 0.0
        self._btc_mom12_window  = deque(maxlen=12)  # 12-bar return window for momentum
        self._btc_mom12_sum     = 0.0
        self._btc_vol_window    = deque(maxlen=10)  # 10-bar return window for volatility
        self._btc_vol_sum       = 0.0
        self._btc_vol_sum_sq    = 0.0
        self._btc_vol_avg_sum   = 0.0              # running sum of btc_volatility values

        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl = 0
        self._daily_loss_limit = -0.05
        self._drawdown_limit = -0.20
        self._min_trade_capital = 5
        self._max_concurrent_positions = 6
        self._daily_start_equity = None
        self._daily_start_equity_date = None
        self.trade_log      = deque(maxlen=500)
        self.log_budget     = 0
        self.last_log_time  = None
        self.base_max_positions = self.max_positions

        self.max_participation_rate = 0.02
        self.reentry_cooldown_minutes = 1
        self._btc_dump_size_mult = 1.0

        self._symbol_performance      = {}  # values will be deque(maxlen=50) — see compute_ranking_overlay
        self.symbol_penalty_threshold = 3
        self.symbol_penalty_size_mult = 0.50

        # Ranking overlay is adaptive to in-sample PnL; disabled by default for cleaner backtests.
        self.ranking_overlay_enabled  = bool(self._get_param("ranking_overlay_enabled",  0))
        self.ranking_combo_bonus_cap  = self._get_param("ranking_combo_bonus_cap",  0.02)
        self.ranking_symbol_bonus_cap = self._get_param("ranking_symbol_bonus_cap", 0.03)

        # Trade Quality Architecture parameters
        # 1. Session / liquidity regime layer
        self.session_layer_enabled = bool(self._get_param("session_layer_enabled", 1.0))
        for _sname, _sdefaults in [
            ('asia_dead', ( 0.00,  0.80,  0.80)),   # was 0.50 — hours 2-4 now traded, raise size from 50% → 80%
            ('asia',      ( 0.00,  0.90,  0.90)),   # was 0.75 — Asian session quality improved, raise from 75% → 90%
            ('eu_open',   ( 0.00,  1.00,  1.00)),
            ('eu_main',   ( 0.00,  1.00,  1.00)),
            ('us_open',   ( 0.00,  1.05,  1.10)),   # threshold_adj zeroed
            ('us_main',   ( 0.00,  1.00,  1.00)),
            ('us_eve',    ( 0.00,  0.90,  0.90)),   # threshold_adj zeroed
        ]:
            setattr(self, f'_session_thresh_{_sname}',
                    self._get_param(f'session_thresh_{_sname}',  _sdefaults[0]))
            setattr(self, f'_session_size_{_sname}',
                    self._get_param(f'session_size_{_sname}',    _sdefaults[1]))
            setattr(self, f'_session_spread_{_sname}',
                    self._get_param(f'session_spread_{_sname}',  _sdefaults[2]))

        # 2. Adverse-selection filter (disabled in backtest, enabled live)
        self.adverse_selection_enabled         = self.LiveMode
        self.asel_spread_widen_thresh          = self._get_param("asel_spread_widen_thresh",          2.5)
        self.asel_vwap_extension_sd_mult       = self._get_param("asel_vwap_extension_sd_mult",       3.0)

        # 3. BB compression context
        self.bb_compression_rank_enabled       = bool(self._get_param("bb_compression_rank_enabled",  1.0))
        self.bb_compression_rank_bonus_cap     = self._get_param("bb_compression_rank_bonus_cap",     0.04)
        self.bb_compression_pct                = self._get_param("bb_compression_pct",                0.75)

        # 4. RS-vs-BTC rank overlay
        self.rs_rank_overlay_enabled = bool(self._get_param("rs_rank_overlay_enabled", 1.0))
        self.rs_rank_scale           = self._get_param("rs_rank_scale", 3.0)
        self.rs_rank_cap             = self._get_param("rs_rank_cap",   0.05)

        # 5 & 6. Stress modes (backtest realism toggles)
        self.stress_spread_mult        = self._get_param("stress_spread_mult",        1.15)
        self.stress_slippage_mult      = self._get_param("stress_slippage_mult",      1.25)
        self.stress_nonfill_penalty    = self._get_param("stress_nonfill_penalty",    0.05)
        self.stress_spread_floor_mult  = self._get_param("stress_spread_floor_mult",  1.25)
        self.stress_impact_mult        = self._get_param("stress_impact_mult",        1.5)
        self.stress_participation_cap  = self._get_param("stress_participation_cap",  0.20)

        # Non-fill simulation seed (backtest only; default 42 = deterministic)
        non_fill_seed = int(self._get_param("non_fill_seed", 42))
        reseed_non_fill_simulation(non_fill_seed)

        # Execution realism controls for backtests.
        self.breakout_nonfill_penalty = self._get_param("breakout_nonfill_penalty", 0.12)
        self.nonfill_market_fallback_enabled = bool(self._get_param("nonfill_market_fallback_enabled", 1.0))
        self.backtest_entry_adverse_offset = self._get_param("backtest_entry_adverse_offset", 0.0018)
        self.backtest_entry_noquote_offset = self._get_param("backtest_entry_noquote_offset", 0.0022)
        self.backtest_use_market_exits = bool(self._get_param("backtest_use_market_exits", 1.0))

        self.max_universe_size = 50

        self.kraken_status = "unknown"
        self._last_skip_reason = None

        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(CryptoUniverse.Kraken(self.UniverseFilter))

        try:
            btc = self.AddCrypto("BTCUSD", Resolution.Minute, Market.Kraken)
            self.btc_symbol = btc.Symbol
        except Exception as e:
            self.Debug(f"Warning: Could not add BTC - {e}")

        # Fear & Greed Index — regime overlay
        try:
            from alt_data import FearGreedData
            self.fear_greed_symbol = self.AddData(FearGreedData, "FNG", Resolution.Daily).Symbol
            self.fear_greed_value = 50  # neutral default
        except Exception as e:
            self.Debug(f"Warning: Could not add Fear & Greed data - {e}")
            self.fear_greed_symbol = None
            self.fear_greed_value = 50

        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 1), self.DailyReport)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(0, 0), self.ResetDailyCounters)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(hours=6)), self.ReviewPerformance)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(12, 0), self.HealthCheck)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=5)), self.ResyncHoldings)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=2)), self.VerifyOrderFills)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.Every(timedelta(minutes=15)), self.PortfolioSanityCheck)

        self.SetWarmUp(timedelta(days=5))
        self.SetSecurityInitializer(self.CustomSecurityInitializer)
        self.Settings.FreePortfolioValuePercentage = 0.01
        self.Settings.InsightScore = False

        self._scoring_engine = MicroScalpEngine(self)

        # ── Dual-engine regime architecture ────────────────────────────────
        # RegimeRouter classifies market state ('trend', 'chop', 'transition')
        # and gates which engine is allowed to enter new trades.
        # ChopEngine provides signal scoring and risk params for sideways markets.
        self._regime_router = RegimeRouter(self)
        self._chop_engine   = ChopEngine(self)

        # Engine attribution: tracks which engine opened each position.
        # Values: 'trend' | 'chop'
        self._entry_engine  = {}

        # Per-engine PnL buckets for reporting (populated in events.py).
        self.pnl_by_engine  = {'trend': [], 'chop': []}

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug(f"=== LIVE TRADING (MICRO-SCALP) {self.ALGO_VERSION} ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")

    def CustomSecurityInitializer(self, security):
        stress_mult = getattr(self, 'stress_slippage_mult', 1.0)
        spread_floor_mult = getattr(self, 'stress_spread_floor_mult', 1.0)
        impact_mult = getattr(self, 'stress_impact_mult', 1.0)
        participation_cap = getattr(self, 'stress_participation_cap', 0.20)
        security.SetSlippageModel(
            RealisticCryptoSlippage(
                stress_mult=stress_mult,
                spread_floor_mult=spread_floor_mult,
                impact_mult=impact_mult,
                participation_cap=participation_cap
            )
        )
        security.SetFeeModel(KrakenTieredFeeModel())

    def _get_param(self, name, default):
        try:
            param = self.GetParameter(name)
            if param is not None and param != "":
                return float(param)
            return default
        except Exception as e:
            self.Debug(f"Error getting parameter {name}: {e}")
            return default

    def ResetDailyCounters(self):
        self.daily_trade_count = 0
        self.last_trade_date = self.Time.date()
        self._daily_open_value = self.Portfolio.TotalPortfolioValue
        for crypto in self.crypto_data.values():
            crypto['trade_count_today'] = 0
        if len(self._session_blacklist) > 0:
            self.Debug(f"Clearing session blacklist ({len(self._session_blacklist)} items)")
            self._session_blacklist.clear()
        self._symbol_entry_cooldowns.clear()
        # Reset chop engine daily trade counts.
        self._chop_engine.reset_daily_counts()
        persist_state(self)

    def HealthCheck(self):
        if self.IsWarmingUp: return
        health_check(self)

    def ResyncHoldings(self):
        if self.IsWarmingUp: return
        if not self.LiveMode: return
        resync_holdings_full(self)

    def VerifyOrderFills(self):
        if self.IsWarmingUp: return
        verify_order_fills(self)

    def PortfolioSanityCheck(self):
        if self.IsWarmingUp: return
        portfolio_sanity_check(self)

    def ReviewPerformance(self):
        if self.IsWarmingUp or len(self.trade_log) < 10: return
        review_performance(self)

    def UniverseFilter(self, universe):
        selected = []
        for crypto in universe:
            ticker = crypto.Symbol.Value
            if ticker in SYMBOL_BLACKLIST or ticker in self._session_blacklist:
                continue
            if not ticker.endswith("USD"):
                continue
            # Filter forex pairs (base not in known fiat currencies)
            base = ticker[:-3]  # remove "USD" suffix
            if base in KNOWN_FIAT_CURRENCIES:
                continue
            if crypto.VolumeInUsd is None or crypto.VolumeInUsd == 0:
                continue
            if crypto.VolumeInUsd < self.min_volume_usd:
                continue
            # Capacity gate: skip assets below min_capacity_usd 24h dollar volume.
            # Prevents trading assets where a $500 order exceeds 0.1% of daily flow.
            if self.min_capacity_usd > 0 and crypto.VolumeInUsd < self.min_capacity_usd:
                continue
            selected.append(crypto)
        selected.sort(key=lambda x: x.VolumeInUsd, reverse=True)
        return [c.Symbol for c in selected[:self.max_universe_size]]

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data:
                initialize_symbol(self, symbol)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.crypto_data and self.crypto_data[symbol].get('consolidator') is not None:
                try:
                    self.SubscriptionManager.RemoveConsolidator(symbol, self.crypto_data[symbol]['consolidator'])
                except Exception as e:
                    self.Debug(f"Warning: Could not remove consolidator for {symbol.Value} - {e}")
            if not self.IsWarmingUp and is_invested_not_dust(self, symbol):
                smart_liquidate(self, symbol, "Removed from universe")
                self.Debug(f"FORCED EXIT: {symbol.Value} - removed from universe")
            if symbol in self.crypto_data and not is_invested_not_dust(self, symbol):
                del self.crypto_data[symbol]

    def OnData(self, data):
        # Reset daily equity baseline at the start of each new calendar day so the
        # daily-loss-limit check is genuinely per-day, not a permanent lifetime floor.
        today = self.Time.date()
        if self._daily_start_equity is None or today != self._daily_start_equity_date:
            self._daily_start_equity = self.Portfolio.TotalPortfolioValue
            self._daily_start_equity_date = today
        current_equity = self.Portfolio.TotalPortfolioValue
        daily_loss_pct = (current_equity - self._daily_start_equity) / self._daily_start_equity
        if daily_loss_pct < self._daily_loss_limit:
            self.Debug(f"⚠️ DAILY LOSS LIMIT: {daily_loss_pct:.2%} | Pausing trades")
            return
        if self._cash_mode_until is not None and self._cash_mode_until > self.Time:
            return
        if hasattr(self, 'fear_greed_symbol') and self.fear_greed_symbol and data.ContainsKey(self.fear_greed_symbol):
            fg = data[self.fear_greed_symbol]
            if fg is not None:
                self.fear_greed_value = fg.Value
        if self.btc_symbol is not None and data.Bars.ContainsKey(self.btc_symbol):
            btc_bar = data.Bars[self.btc_symbol]
            btc_price = float(btc_bar.Close)
            if len(self.btc_prices) > 0:
                btc_return = (btc_price - self.btc_prices[-1]) / self.btc_prices[-1]
                self.btc_returns.append(btc_return)
                mom_w = self._btc_mom12_window
                if len(mom_w) == mom_w.maxlen:
                    self._btc_mom12_sum -= mom_w[0]
                mom_w.append(btc_return)
                self._btc_mom12_sum += btc_return
                btc_vol_w = self._btc_vol_window
                if len(btc_vol_w) == btc_vol_w.maxlen:
                    old_return = btc_vol_w[0]
                    self._btc_vol_sum    -= old_return
                    self._btc_vol_sum_sq -= old_return * old_return
                btc_vol_w.append(btc_return)
                self._btc_vol_sum    += btc_return
                self._btc_vol_sum_sq += btc_return * btc_return
                if len(btc_vol_w) >= 10:
                    btc_vol_mean = self._btc_vol_sum / len(btc_vol_w)
                    btc_vol_var  = self._btc_vol_sum_sq / len(btc_vol_w) - btc_vol_mean * btc_vol_mean
                    new_vol = math.sqrt(max(btc_vol_var, 0.0))
                    if len(self.btc_volatility) == self.btc_volatility.maxlen:
                        self._btc_vol_avg_sum -= self.btc_volatility[0]
                    self.btc_volatility.append(new_vol)
                    self._btc_vol_avg_sum += new_vol
            sma_w = self._btc_sma48_window
            if len(sma_w) == sma_w.maxlen:
                self._btc_sma48_sum -= sma_w[0]
            sma_w.append(btc_price)
            self._btc_sma48_sum += btc_price
            self.btc_prices.append(btc_price)
            self.btc_ema_24.Update(btc_bar.EndTime, btc_price)
        for symbol in list(self.crypto_data.keys()):
            if not data.Bars.ContainsKey(symbol):
                continue
            try:
                quote_bar = data.QuoteBars[symbol] if data.QuoteBars.ContainsKey(symbol) else None
                if quote_bar is not None:
                    crypto = self.crypto_data[symbol]
                    try:
                        bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
                        ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
                        if bid_sz > 0 or ask_sz > 0:
                            crypto['bid_size'] = bid_sz
                            crypto['ask_size'] = ask_sz
                    except Exception:
                        pass
            except Exception:
                pass
        if self.IsWarmingUp:
            return
        if not self._positions_synced:
            if not self._first_post_warmup:
                cancel_stale_new_orders(self)
            sync_existing_positions(self)
            self._positions_synced = True
            self._first_post_warmup = False
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
                self.Debug("Fallback: kraken_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        update_market_context(self)
        self._regime_router.update()
        self.Rebalance()
        self.CheckExits()

    def _on_five_minute_bar(self, sender, bar):
        """Called every 5 minutes with consolidated bar data."""
        symbol = bar.Symbol
        if symbol not in self.crypto_data:
            return
        # Process the 5-min bar; bid/ask updates happen separately in OnData.
        update_symbol_data(self, symbol, bar)

    def _annualized_vol(self, crypto):
        if crypto is None:
            return None
        if len(crypto.get('volatility', [])) == 0:
            return None
        return float(crypto['volatility'][-1]) * self.sqrt_annualization

    def _compute_portfolio_risk_estimate(self):
        total_value = self.Portfolio.TotalPortfolioValue
        if total_value <= 0:
            return 0.0
        risk = 0.0
        for kvp in self.Portfolio:
            symbol, holding = kvp.Key, kvp.Value
            if not is_invested_not_dust(self, symbol):
                continue
            crypto = self.crypto_data.get(symbol)
            asset_vol_ann = self._annualized_vol(crypto)
            if asset_vol_ann is None:
                asset_vol_ann = self.min_asset_vol_floor
            weight = abs(holding.HoldingsValue) / total_value
            risk += weight * asset_vol_ann
        return risk

    def _normalize(self, v, mn, mx):
        if mx - mn <= 0:
            return 0.5
        return max(0, min(1, (v - mn) / (mx - mn)))

    def _calculate_factor_scores(self, symbol, crypto):
        long_score, long_components = self._scoring_engine.calculate_scalp_score(crypto)

        sp = get_spread_pct(self, symbol)
        if sp is not None and sp > 0:
            spread_penalty = min((sp / 0.005) * 0.15, 0.15)
            long_score *= (1.0 - spread_penalty)

        components = long_components.copy()
        components['_scalp_score'] = long_score
        components['_direction'] = 1
        components['_long_score'] = long_score
        return components

    def _check_correlation(self, new_symbol):
        if not self.entry_prices:
            return True
        new_crypto = self.crypto_data.get(new_symbol)
        if not new_crypto or len(new_crypto['returns']) < 24:
            return True
        # np.array() accepts deques directly — avoids an intermediate Python list copy
        new_rets = np.array(new_crypto['returns'])[-24:]
        if np.std(new_rets) < 1e-10:
            return True
        for sym in list(self.entry_prices.keys()):
            if sym == new_symbol:
                continue
            existing = self.crypto_data.get(sym)
            if not existing or len(existing['returns']) < 24:
                continue
            exist_rets = np.array(existing['returns'])[-24:]
            if np.std(exist_rets) < 1e-10:
                continue
            try:
                corr = np.corrcoef(new_rets, exist_rets)[0, 1]
                if corr > 0.85:
                    return False
            except Exception:
                continue
        return True

    def _daily_loss_exceeded(self):
        if self._daily_open_value is None or self._daily_open_value <= 0:
            return False
        current = self.Portfolio.TotalPortfolioValue
        if current <= 0:
            return True
        drop = (self._daily_open_value - current) / self._daily_open_value
        return drop >= 0.05

    def _log_skip(self, reason):
        if self.LiveMode:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason
        elif reason != self._last_skip_reason:
            debug_limited(self, f"Rebalance skip: {reason}")
            self._last_skip_reason = reason

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        if self.Time.hour in self.skip_hours_utc:
            self._log_skip(f"skip hour {self.Time.hour} UTC")
            return

        if self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded")
            return

        self._btc_dump_size_mult = 1.0
        if len(self.btc_returns) >= 5:
            # Sum last 5 returns without creating a full list copy
            btc_5bar_return = sum(itertools.islice(reversed(self.btc_returns), 5))
            if btc_5bar_return < -0.04:
                self._log_skip("BTC crashing")
                return
            elif btc_5bar_return < -0.02:
                self._btc_dump_size_mult = 0.50
            elif btc_5bar_return > 0.02:
                self._btc_dump_size_mult = 1.30  # 30% size boost on strong BTC momentum
        
        if self._cash_mode_until is not None and self.Time < self._cash_mode_until:
            self._log_skip("cash mode - poor recent performance")
            return

        self.log_budget = 20

        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            self._log_skip("rate limited")
            return

        if self.LiveMode and not live_safety_checks(self):
            return
        if self.LiveMode and getattr(self, 'kraken_status', 'unknown') in ("maintenance", "cancel_only"):
            self._log_skip("kraken not online")
            return
        cancel_stale_new_orders(self)
        if self.daily_trade_count >= self.max_daily_trades:
            self._log_skip("max daily trades")
            return
        val = self.Portfolio.TotalPortfolioValue
        if self.peak_value is None or self.peak_value < 1:
            self.peak_value = val
        if self.drawdown_cooldown > 0:
            self.drawdown_cooldown -= 1
            if self.drawdown_cooldown <= 0:
                self.peak_value = val
                self.consecutive_losses = 0
            else:
                self._log_skip(f"drawdown cooldown {self.drawdown_cooldown}h")
                return
        self.peak_value = max(self.peak_value, val)
        dd = (self.peak_value - val) / self.peak_value if self.peak_value > 0 else 0
        if dd > self.max_drawdown_limit:
            self.drawdown_cooldown = self.cooldown_hours
            self._log_skip(f"drawdown {dd:.1%} > limit")
            return
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.drawdown_cooldown = 3
            self._consecutive_loss_halve_remaining = 3
            self.consecutive_losses = 0
            self._log_skip("consecutive loss cooldown (5 losses)")
            return
        if self.consecutive_losses >= 6:
            self._circuit_breaker_trigger_count += 1
            backoff_hours = min(1 * (2 ** (self._circuit_breaker_trigger_count - 1)), 8)
            self.circuit_breaker_expiry = self.Time + timedelta(hours=backoff_hours)
            self.consecutive_losses = 0
            self._log_skip(f"circuit breaker triggered (6 losses, {backoff_hours}h cooldown #{self._circuit_breaker_trigger_count})")
            return
        if self.circuit_breaker_expiry is not None and self.Time < self.circuit_breaker_expiry:
            self._log_skip("circuit breaker active")
            return
        elif self.circuit_breaker_expiry is not None and self.Time >= self.circuit_breaker_expiry:
            self._circuit_breaker_trigger_count = max(0, self._circuit_breaker_trigger_count - 1)  # decay
            self.circuit_breaker_expiry = None
        pos_count = get_actual_position_count(self)
        if pos_count >= self.max_positions:
            self._log_skip("at max positions")
            return
        fg_value = getattr(self, 'fear_greed_value', 50)
        if fg_value >= 85 and self.market_regime != "bull":
            effective_max_pos = max(1, self.max_positions // 2)
            if pos_count >= effective_max_pos:
                self._log_skip(f"Fear&Greed extreme greed ({fg_value}) — reduced max positions")
                return
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
            return

        # ── Regime routing (dual-engine architecture) ──────────────────────
        # The RegimeRouter classifies the current market state and determines
        # which engine is allowed to open new trades:
        #   'trend'      → existing MicroScalpEngine logic continues below
        #   'chop'       → delegate to the dedicated ChopEngine and return
        #   'transition' → suppress all new entries; only manage existing exits
        active_regime = self._regime_router.route()
        if active_regime == "transition":
            self._log_skip(f"regime router: transition (btc={self.market_regime})")
            return
        if active_regime == "chop":
            self._run_chop_rebalance()
            return
        # active_regime == "trend" → fall through to existing trend engine logic.

        count_scored = 0
        count_above_thresh = 0
        scores = []
        threshold_now = self.entry_threshold
        # Regime-adaptive threshold: slightly higher conviction needed in unfavorable conditions.
        if self.market_regime == "bear":
            threshold_now = max(threshold_now, 0.42)
        elif self.market_regime == "sideways":
            threshold_now = max(threshold_now, 0.38)
        elif self.market_regime == "bull":
            threshold_now = min(threshold_now, 0.35)  # lower bar in bull — more trades when edge is strongest
        # Regime-adaptive position size cap
        effective_max_position_pct = self.max_position_pct
        if self.market_regime == "bull":
            effective_max_position_pct = min(self.max_position_pct * 1.5, 0.40)
        # Session-layer threshold adjustment (additive, conservative by default).
        _sess_thresh_adj, _sess_size_mult, _sess_spread_cap_mult = get_session_quality(
            self, self.Time.hour)
        threshold_now = max(0.0, threshold_now + _sess_thresh_adj)
        for symbol in list(self.crypto_data.keys()):
            if symbol.Value in SYMBOL_BLACKLIST or symbol.Value in self._session_blacklist:
                continue
            if symbol.Value in self._symbol_entry_cooldowns and self.Time < self._symbol_entry_cooldowns[symbol.Value]:
                continue
            if has_open_orders(self, symbol):
                continue

            if not spread_ok(self, symbol):
                continue

            crypto = self.crypto_data[symbol]
            if not self._is_ready(crypto):
                continue

            factor_scores = self._calculate_factor_scores(symbol, crypto)
            if not factor_scores:
                continue
            count_scored += 1

            composite_score = factor_scores.get('_scalp_score', 0.0)
            net_score = composite_score

            # Log vol-only candidates filtered by multi-signal gate (logging only).
            _sig_keys = MicroScalpEngine.SIGNAL_KEYS
            if (factor_scores.get('vol_ignition', 0) >= 0.10
                    and sum(1 for k in _sig_keys if factor_scores.get(k, 0) >= 0.10) < self.min_signal_count):
                sig_vals = ' '.join(f"{k[:4]}={factor_scores.get(k, 0):.2f}" for k in _sig_keys)
                debug_limited(self, f"WEAK SIGNAL {symbol.Value}: {sig_vals} — skipped (need {self.min_signal_count} active signals)")

            crypto['recent_net_scores'].append(net_score)

            # RS override: altcoins with positive RS vs BTC enter at base threshold in bear/sideways.
            effective_threshold = threshold_now
            if self.market_regime in ("bear", "sideways") and len(crypto.get('rs_vs_btc', [])) >= 3:
                # Sum last 3 RS values without creating a list copy
                recent_rs = sum(itertools.islice(reversed(crypto['rs_vs_btc']), 3))
                if recent_rs > 0.005:
                    effective_threshold = self.entry_threshold

            if net_score >= effective_threshold:
                count_above_thresh += 1
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'net_score': net_score,
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'crypto': crypto,   # passed to compute_ranking_overlay for BB/RS context
                })

        try:
            cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            cash = self.Portfolio.Cash

        debug_limited(self, f"REBALANCE: {count_above_thresh}/{count_scored} above thresh={threshold_now:.2f} | cash=${cash:.2f}")

        if len(scores) == 0:
            self._log_skip("no candidates passed filters")
            return
        if self.ranking_overlay_enabled:
            for s in scores:
                s['rank_adj'] = compute_ranking_overlay(self, s)
                s['rank_score'] = s['net_score'] + s['rank_adj']
            scores.sort(key=lambda x: x['rank_score'], reverse=True)
            if len(scores) > 1:
                adj_str = ' '.join(f"{s['symbol'].Value}({s['net_score']:.2f}{s['rank_adj']:+.3f})" for s in scores[:3])
                debug_limited(self, f"RANK: {adj_str}")
        else:
            scores.sort(key=lambda x: x['net_score'], reverse=True)
        self._last_skip_reason = None
        execute_trend_trades(self, scores, threshold_now, effective_max_position_pct)

    def _run_chop_rebalance(self):
        run_chop_rebalance(self)

    def _is_ready(self, c):
        return len(c['prices']) >= 10 and c['rsi'].IsReady

    def CheckExits(self):
        if self.IsWarmingUp:
            return

        if self._rate_limit_until is not None and self.Time < self._rate_limit_until:
            return
        for kvp in self.Portfolio:
            if not is_invested_not_dust(self, kvp.Key):
                self._failed_exit_attempts.pop(kvp.Key, None)
                self._failed_exit_counts.pop(kvp.Key, None)
                continue

            # Orphan: invested but no crypto_data — force liquidate
            if self.crypto_data.get(kvp.Key) is None:
                self.Debug(f"ORPHAN DETECTED: {kvp.Key.Value} has no crypto_data — force liquidating")
                smart_liquidate(self, kvp.Key, "Orphan Position")
                continue

            if self._failed_exit_counts.get(kvp.Key, 0) >= 3:
                continue
            self._check_exit(kvp.Key, self.Securities[kvp.Key].Price, kvp.Value)

        for kvp in self.Portfolio:
            symbol = kvp.Key
            if not is_invested_not_dust(self, symbol):
                continue
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = kvp.Value.AveragePrice
                self.highest_prices[symbol] = kvp.Value.AveragePrice
                self.entry_times[symbol] = self.Time
                self.Debug(f"ORPHAN RECOVERY: {symbol.Value} re-tracked")

    def _check_exit(self, symbol, price, holding):
        if len(self.Transactions.GetOpenOrders(symbol)) > 0:
            return
        if symbol in self._cancel_cooldowns and self.Time < self._cancel_cooldowns[symbol]:
            return

        # Update per-trade MFE/MAE excursion tracker on every CheckExits call.
        update_trade_excursion(self, symbol, price)

        min_notional_usd = get_min_notional_usd(self, symbol)
        if price > 0 and abs(holding.Quantity) * price < min_notional_usd * 0.3:
            try:
                self.Liquidate(symbol)
            except Exception as e:
                self.Debug(f"DUST liquidation failed for {symbol.Value}: {e}")
            cleanup_position(self, symbol)
            self._failed_exit_counts.pop(symbol, None)
            return

        actual_qty = abs(holding.Quantity)
        rounded_sell = round_quantity(self, symbol, actual_qty)
        if rounded_sell > actual_qty:
            self.Debug(f"DUST (rounded sell > actual): {symbol.Value} | actual={actual_qty} rounded={rounded_sell} — cleaning up")
            cleanup_position(self, symbol)
            self._failed_exit_counts.pop(symbol, None)
            return
        if symbol not in self.entry_prices:
            self.entry_prices[symbol] = holding.AveragePrice
            self.highest_prices[symbol] = holding.AveragePrice
            self.entry_times[symbol] = self.Time
        entry = self.entry_prices[symbol]
        highest = self.highest_prices.get(symbol, entry)
        if price > highest:
            self.highest_prices[symbol] = price
        pnl = (price - entry) / entry if entry > 0 else 0

        crypto = self.crypto_data.get(symbol)
        dd = (highest - price) / highest if highest > 0 else 0
        hours = (self.Time - self.entry_times.get(symbol, self.Time)).total_seconds() / 3600
        minutes = hours * 60

        atr = crypto['atr'].Current.Value if crypto and crypto['atr'].IsReady else None

        # ── Chop engine exit path ──────────────────────────────────────────
        # If this position was opened by the chop engine, use its own tighter
        # exit parameters and simpler exit logic (no partial TP tiers, no
        # ATR trail, no pyramid — chop trades target small fast moves).
        if self._entry_engine.get(symbol) == 'chop':
            chop_params = self._chop_engine.get_exit_params(symbol, entry, crypto)
            chop_sl = chop_params['sl']
            chop_tp = chop_params['tp']
            chop_max_h = chop_params['max_hold_hours']
            chop_ts_min = chop_params['time_stop_min_pnl']

            chop_tag = ""
            if pnl <= -chop_sl:
                chop_tag = "Chop Stop Loss"
            elif pnl >= chop_tp:
                chop_tag = "Chop Take Profit"
            elif hours >= chop_max_h and pnl < chop_ts_min:
                chop_tag = "Chop Time Stop"

            if chop_tag:
                if price * abs(holding.Quantity) < min_notional_usd * 0.9:
                    return
                if pnl < 0:
                    self._symbol_loss_cooldowns[symbol] = self.Time + timedelta(minutes=30)
                sold = smart_liquidate(self, symbol, chop_tag)
                if sold:
                    cooldown_delta = timedelta(minutes=self.reentry_cooldown_minutes)
                    exit_cooldown_delta = timedelta(hours=self.exit_cooldown_hours)
                    self._exit_cooldowns[symbol] = self.Time + max(cooldown_delta, exit_cooldown_delta)
                    self.entry_volumes.pop(symbol, None)
                    self._choppy_regime_entries.pop(symbol, None)
                    self._entry_engine.pop(symbol, None)
                    if hasattr(self, '_pyramided_symbols'):
                        self._pyramided_symbols.discard(symbol)
                    hold_bucket = get_hold_bucket(hours)
                    self.Debug(
                        f"{chop_tag}: {symbol.Value} | PnL:{pnl:+.2%} | "
                        f"Held:{hours:.1f}h ({hold_bucket})"
                    )
                else:
                    fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                    self._failed_exit_counts[symbol] = fail_count
                    self.Debug(
                        f"⚠️ CHOP EXIT FAILED ({chop_tag}) #{fail_count}: "
                        f"{symbol.Value} | PnL:{pnl:+.2%}"
                    )
                    if fail_count >= 3:
                        self.Debug(f"FATAL CHOP EXIT: {symbol.Value} — escalating to market order")
                        try:
                            qty = abs(holding.Quantity)
                            if qty > 0:
                                self.MarketOrder(symbol, -qty,
                                                 tag=f"Force Chop Exit (fail#{fail_count})")
                        except Exception as e:
                            self.Debug(f"Force chop market exit error for {symbol.Value}: {e}")
                        self._failed_exit_counts.pop(symbol, None)
                        self.entry_volumes.pop(symbol, None)
                        self._choppy_regime_entries.pop(symbol, None)
                        self._entry_engine.pop(symbol, None)
            return  # chop positions are only managed by chop exit logic above

        # ── Trend engine exit path (unchanged from original) ───────────────
        if atr and entry > 0:
            sl = max((atr * self.atr_sl_mult) / entry, self.tight_stop_loss)
            tp = max((atr * self.atr_tp_mult) / entry, self.quick_take_profit)
        else:
            sl = self.tight_stop_loss
            tp = self.quick_take_profit

        if tp < sl * 1.2:
            tp = sl * 1.2

        tp_mult = 1.0
        if self._choppy_regime_entries.get(symbol, False):
            tp_mult *= 0.75
        if self.volatility_regime == "low":
            tp_mult *= 0.85
        tp_mult = max(tp_mult, 0.55)
        tp = tp * tp_mult

        trailing_activation = self.trail_activation
        trailing_stop_pct = self.trail_stop_pct

        tier = self._partial_tp_tier.get(symbol, 0)

        # Pyramid into winner: add to position when up >pyramid_threshold and not yet pyramided
        if (getattr(self, 'pyramid_enabled', False)
                and pnl >= getattr(self, 'pyramid_threshold', 0.015)
                and not self._partial_tp_taken.get(symbol, False)
                and symbol not in getattr(self, '_pyramided_symbols', set())
                and get_actual_position_count(self) < self.max_positions
                and not has_open_orders(self, symbol)):
            if not hasattr(self, '_pyramided_symbols'):
                self._pyramided_symbols = set()
            try:
                price = self.Securities[symbol].Price
                original_val = abs(holding.Quantity) * self.entry_prices.get(symbol, price)
                add_val = original_val * self.pyramid_size_pct
                add_qty = round_quantity(self, symbol, add_val / price)
                min_qty = get_min_quantity(self, symbol)
                if add_qty >= min_qty and add_val <= self.Portfolio.Cash * 0.90:
                    ticket = place_limit_or_market(self, symbol, add_qty, tag="Pyramid Add")
                    if ticket is not None:
                        self._pyramided_symbols.add(symbol)
                        self.Debug(f"PYRAMID: {symbol.Value} | PnL:{pnl:+.2%} | add_val=${add_val:.2f}")
            except Exception as e:
                self.Debug(f"Pyramid error for {symbol.Value}: {e}")

        if tier == 0 and pnl >= self.partial_tp_tier1_threshold:
            if partial_smart_sell(self, symbol, self.partial_tp_tier1_pct, "Partial TP"):
                self._partial_tp_tier[symbol] = 1
                self._partial_tp_taken[symbol] = True
                self.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%} | Held:{minutes:.0f}min")
                return

        if tier == 1 and pnl >= self.partial_tp_tier2_threshold:
            if partial_smart_sell(self, symbol, self.partial_tp_tier2_pct, "Partial TP T2"):
                self._partial_tp_tier[symbol] = 2
                self.Debug(f"PARTIAL TP T2: {symbol.Value} | PnL:{pnl:+.2%} | Held:{minutes:.0f}min")
                return

        tag = ""
        if pnl <= -sl:
            tag = "Stop Loss"
        if not tag and not self._partial_tp_taken.get(symbol, False) and pnl >= tp:
            tag = "Take Profit"
        if not tag and pnl > trailing_activation and dd >= trailing_stop_pct:
            tag = "Trailing Stop"
        if not tag and atr and entry > 0 and holding.Quantity > 0:
            mult = self.post_partial_tp_trail_mult if self._partial_tp_taken.get(symbol, False) else self.atr_trail_mult
            trail_level = highest - atr * mult
            if crypto: crypto['trail_stop'] = trail_level
            if minutes >= self.min_trail_hold_minutes and price <= trail_level:
                tag = "ATR Trail"
        if not tag and hours >= self.time_stop_hours and pnl < self.time_stop_pnl_min:
            tag = "Time Stop"

        if tag:
            if price * abs(holding.Quantity) < min_notional_usd * 0.9:
                return
            if pnl < 0:
                self._symbol_loss_cooldowns[symbol] = self.Time + timedelta(minutes=30)
            sold = smart_liquidate(self, symbol, tag)
            if sold:
                cooldown_delta = timedelta(minutes=self.reentry_cooldown_minutes)
                exit_cooldown_delta = timedelta(hours=self.exit_cooldown_hours)
                self._exit_cooldowns[symbol] = self.Time + max(cooldown_delta, exit_cooldown_delta)
                self.entry_volumes.pop(symbol, None)
                self._choppy_regime_entries.pop(symbol, None)
                self._entry_engine.pop(symbol, None)
                if hasattr(self, '_pyramided_symbols'):
                    self._pyramided_symbols.discard(symbol)
                hold_bucket = get_hold_bucket(hours)
                self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h ({hold_bucket})")
            else:
                fail_count = self._failed_exit_counts.get(symbol, 0) + 1
                self._failed_exit_counts[symbol] = fail_count
                self.Debug(f"⚠️ EXIT FAILED ({tag}) #{fail_count}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
                if fail_count >= 3:
                    self.Debug(f"FATAL EXIT FAILURE: {symbol.Value} — {fail_count} attempts failed, escalating to market order")
                    try:
                        holding = self.Portfolio[symbol]
                        qty = abs(holding.Quantity)
                        if qty > 0:
                            self.MarketOrder(symbol, -qty, tag=f"Force Exit (fail#{fail_count})")
                    except Exception as e:
                        self.Debug(f"Force market exit error for {symbol.Value}: {e}")
                    self._failed_exit_counts.pop(symbol, None)
                    self.entry_volumes.pop(symbol, None)
                    self._choppy_regime_entries.pop(symbol, None)
                    self._entry_engine.pop(symbol, None)

    def OnOrderEvent(self, event):
        on_order_event(self, event)

    def OnBrokerageMessage(self, message):
        try:
            txt = message.Message.lower()
            if "system status:" in txt:
                if "online" in txt:
                    self.kraken_status = "online"
                elif "maintenance" in txt:
                    self.kraken_status = "maintenance"
                elif "cancel_only" in txt:
                    self.kraken_status = "cancel_only"
                elif "post_only" in txt:
                    self.kraken_status = "post_only"
                else:
                    self.kraken_status = "unknown"
                self.Debug(f"Kraken status update: {self.kraken_status}")
            
            if "rate limit" in txt or "too many" in txt:
                self.Debug(f"⚠️ RATE LIMIT - pausing {self.rate_limit_cooldown_minutes}min")
                self._rate_limit_until = self.Time + timedelta(minutes=self.rate_limit_cooldown_minutes)
        except Exception as e:
            self.Debug(f"BrokerageMessage parse error: {e}")

    def OnEndOfAlgorithm(self):
        final_report(self)

    def DailyReport(self):
        if self.IsWarmingUp: return
        daily_report(self)
