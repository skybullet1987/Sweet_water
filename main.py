from AlgorithmImports import *
from execution import *
from reporting import *
from order_management import *
from realistic_slippage import RealisticCryptoSlippage
from events import on_order_event
from scoring import MicroScalpEngine
from collections import deque
import numpy as np
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
# endregion


class KrakenTieredFeeModel(FeeModel):
    """
    Volume-tiered fee model matching Kraken Pro (Canada) schedule.
    Tracks cumulative volume and applies the correct tier.
    Limit orders: 75% maker / 25% taker blend.
    """

    LIMIT_TAKER_RATIO = 0.25

    # Kraken Pro Canada fee tiers: (min_volume_usd, maker_pct, taker_pct)
    FEE_TIERS = [
        (500_000, 0.0008, 0.0018),   # $500K+
        (250_000, 0.0010, 0.0020),   # $250K+
        (100_000, 0.0012, 0.0022),   # $100K+
        (50_000,  0.0014, 0.0024),   # $50K+
        (25_000,  0.0020, 0.0035),   # $25K+
        (10_000,  0.0022, 0.0038),   # $10K+
        (2_500,   0.0030, 0.0060),   # $2.5K+
        (0,       0.0040, 0.0080),   # $0+
    ]

    def __init__(self):
        self._cumulative_volume = 0.0
        self._start_time = None

    def GetOrderFee(self, parameters):
        order = parameters.Order
        price = parameters.Security.Price
        trade_value = order.AbsoluteQuantity * price

        # Track volume
        self._cumulative_volume += trade_value
        if self._start_time is None:
            self._start_time = order.Time

        # Approximate 30-day volume
        elapsed_days = max((order.Time - self._start_time).days, 1)
        monthly_volume = self._cumulative_volume * 30.0 / elapsed_days

        # Find the correct tier
        maker_rate, taker_rate = self.FEE_TIERS[-1][1], self.FEE_TIERS[-1][2]  # default: $0+ tier
        for min_vol, maker, taker in self.FEE_TIERS:
            if monthly_volume >= min_vol:
                maker_rate, taker_rate = maker, taker
                break

        # Calculate blended fee
        if order.Type == OrderType.Limit:
            fee_pct = (1 - self.LIMIT_TAKER_RATIO) * maker_rate + self.LIMIT_TAKER_RATIO * taker_rate
        else:
            fee_pct = taker_rate  # Market orders always taker

        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


class SimplifiedCryptoStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2026, 1, 1)
        self.SetEndDate(2026, 12, 1)
        self.SetCash(50)
        self.SetBrokerageModel(BrokerageName.Kraken, AccountType.Cash)

        self.entry_threshold = 0.40
        self.high_conviction_threshold = 0.50
        self.quick_take_profit = self._get_param("quick_take_profit", 0.100)   # was 0.150
        self.tight_stop_loss   = self._get_param("tight_stop_loss",   0.050)   # was 0.035
        self.atr_tp_mult  = self._get_param("atr_tp_mult",  4.0)
        self.atr_sl_mult  = self._get_param("atr_sl_mult",  2.0)
        self.trail_activation  = self._get_param("trail_activation",  0.030)   # was 0.040
        self.trail_stop_pct    = self._get_param("trail_stop_pct",    0.020)   # was 0.025
        self.time_stop_hours   = self._get_param("time_stop_hours",   5.0)     # was 3.0
        self.time_stop_pnl_min = self._get_param("time_stop_pnl_min", 0.005)

        self.atr_trail_mult             = 6.0
        self.post_partial_tp_trail_mult = 3.5
        self.min_trail_hold_minutes     = 60

        self.position_size_pct  = 0.90
        self.max_positions      = 4
        self.min_notional       = 5.5
        self.max_position_pct   = self._get_param("max_position_pct", 0.40)  # 40% max per position
        self.min_price_usd      = 0.001
        self.cash_reserve_pct   = 0.00
        self.min_notional_fee_buffer = 1.5

        self.target_position_ann_vol = self._get_param("target_position_ann_vol", 0.35)
        self.portfolio_vol_cap       = self._get_param("portfolio_vol_cap", 0.80)
        self.min_asset_vol_floor     = 0.05

        self.ultra_short_period = 3
        self.short_period       = 6
        self.medium_period      = 12
        self.lookback           = 48
        self.sqrt_annualization = np.sqrt(60 * 24 * 365)   # 1-minute bar math

        self.max_spread_pct         = 0.005  # Tighter: 0.5% max spread (was 0.8%)
        self.spread_median_window   = 12
        self.spread_widen_mult      = 2.0    # Tighter: 2x median (was 2.5x)
        self.min_dollar_volume_usd  = 50000  # $50K min (was $20K)
        self.min_volume_usd         = 25000  # $25K min (was $10K)

        self.skip_hours_utc         = []
        self.max_daily_trades       = 24000
        self.daily_trade_count      = 0
        self.last_trade_date        = None
        self.exit_cooldown_hours    = 0.5
        self.cancel_cooldown_minutes = 1
        self.max_symbol_trades_per_day = 5

        self.expected_round_trip_fees = 0.0060
        self.fee_slippage_buffer      = 0.002
        self.min_expected_profit_pct  = 0.012
        self.adx_min_period           = 10

        self.stale_order_timeout_seconds      = 30
        self.live_stale_order_timeout_seconds = 60
        self.max_concurrent_open_orders       = 5
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
        self._entry_signal_combos  = {}   # {symbol: "vol+mean_rev+vwap" etc.}
        self.pnl_by_signal_combo   = {}   # {"vol+mean_rev": [pnl1, pnl2, ...]}
        self.pnl_by_hold_time      = {}   # {"<30min": [pnl1, ...], "30min-2h": [...], ...}

        self.peak_value       = None
        self.drawdown_cooldown = 0
        self.crypto_data      = {}
        self.entry_prices     = {}
        self.highest_prices   = {}
        self.entry_times      = {}
        self.entry_volumes    = {}   # for volume dry-up exit
        self._partial_tp_taken      = {}
        self._partial_tp_tier       = {}   # 0 = none, 1 = tier1 taken
        self._partial_sell_symbols  = set()
        self._choppy_regime_entries = {}
        self.partial_tp_tier1_threshold = 0.025  # First partial TP at 2.5%
        self.partial_tp_tier1_pct   = 0.33        # Sell 33% at tier 1
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
        self.btc_returns      = deque(maxlen=72)   # was 144
        self.btc_prices       = deque(maxlen=72)   # was 144
        self.btc_volatility   = deque(maxlen=72)   # was 144
        self.btc_ema_24       = ExponentialMovingAverage(24)
        self.market_regime    = "unknown"
        self.volatility_regime = "normal"
        self.market_breadth   = 0.5
        self._regime_hold_count = 0

        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl = 0
        
        # Paper trading safety limits for $50 capital
        self._daily_loss_limit = -0.05  # Stop trading if down 5% daily
        self._drawdown_limit = -0.20  # Stop for day if down 20%
        self._min_trade_capital = 5  # Minimum $5 per trade
        self._max_concurrent_positions = 4  # Max 4 concurrent positions
        self._daily_start_equity = None
        self.trade_log      = deque(maxlen=500)
        self.log_budget     = 0
        self.last_log_time  = None
        self.base_max_positions = self.max_positions  # Baseline for performance recovery logic

        # Risk management parameters
        self.max_participation_rate = 0.02   # Max 2% of daily dollar volume per position
        self.reentry_cooldown_minutes = 5   # Min 5 min re-entry cooldown
        self._btc_dump_size_mult = 1.0       # Position size multiplier during BTC weakness

        # Per-symbol performance tracking
        self._symbol_performance      = {}   # {symbol_value: deque of recent PnLs}
        self.symbol_penalty_threshold = 3    # consecutive losses to trigger penalty
        self.symbol_penalty_size_mult = 0.50 # halve position size on penalized symbols

        self.max_universe_size = 30  # Focus on top 30 liquid assets (was 75)

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

        if self.LiveMode:
            cleanup_object_store(self)
            load_persisted_state(self)
            self.Debug("=== LIVE TRADING (MICRO-SCALP) v7.3.1 ===")
            self.Debug(f"Capital: ${self.Portfolio.Cash:.2f} | Max pos: {self.max_positions} | Size: {self.position_size_pct:.0%}")

    def CustomSecurityInitializer(self, security):
        security.SetSlippageModel(RealisticCryptoSlippage())
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

    def _normalize_order_time(self, order_time):
        return normalize_order_time(order_time)

    def _record_exit_pnl(self, symbol, entry_price, exit_price, exit_tag="Unknown"):
        return record_exit_pnl(self, symbol, entry_price, exit_price, exit_tag=exit_tag)

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

    def _cancel_stale_orders(self):
        try:
            open_orders = self.Transactions.GetOpenOrders()
            if len(open_orders) > 0:
                self.Debug(f"Found {len(open_orders)} open orders - canceling all...")
                for order in open_orders:
                    self.Transactions.CancelOrder(order.Id)
        except Exception as e:
            self.Debug(f"Error canceling stale orders: {e}")

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
            if crypto.VolumeInUsd >= self.min_volume_usd:
                selected.append(crypto)
        selected.sort(key=lambda x: x.VolumeInUsd, reverse=True)
        return [c.Symbol for c in selected[:self.max_universe_size]]

    def _initialize_symbol(self, symbol):
        self.crypto_data[symbol] = {
            'prices': deque(maxlen=self.lookback),
            'returns': deque(maxlen=self.lookback),
            'volume': deque(maxlen=self.lookback),
            'volume_ma': deque(maxlen=self.medium_period),
            'dollar_volume': deque(maxlen=self.lookback),
            'ema_ultra_short': ExponentialMovingAverage(self.ultra_short_period),
            'ema_short': ExponentialMovingAverage(self.short_period),
            'ema_medium': ExponentialMovingAverage(self.medium_period),
            'ema_5': ExponentialMovingAverage(5),
            'atr': AverageTrueRange(14),
            'adx': AverageDirectionalIndex(self.adx_min_period),
            'volatility': deque(maxlen=self.medium_period),
            'rsi': RelativeStrengthIndex(7),
            'rs_vs_btc': deque(maxlen=self.medium_period),
            'last_price': 0,
            'recent_net_scores': deque(maxlen=3),
            'spreads': deque(maxlen=self.spread_median_window),
            'trail_stop': None,
            'highs': deque(maxlen=self.lookback),
            'lows': deque(maxlen=self.lookback),
            'bb_upper': deque(maxlen=self.short_period),
            'bb_lower': deque(maxlen=self.short_period),
            'bb_width': deque(maxlen=self.medium_period),
            'trade_count_today': 0,
            'last_loss_time': None,
            'bid_size': 0.0,
            'ask_size': 0.0,
            'vwap_pv': deque(maxlen=20),
            'vwap_v': deque(maxlen=20),
            'vwap': 0.0,
            'volume_long': deque(maxlen=1440),
            'vwap_sd': 0.0,
            'vwap_sd2_lower': 0.0,
            'vwap_sd3_lower': 0.0,
            'consolidator': None,
        }
        # Create 5-minute consolidator for this symbol
        try:
            consolidator = TradeBarConsolidator(timedelta(minutes=5))
            consolidator.DataConsolidated += self._on_five_minute_bar
            self.SubscriptionManager.AddConsolidator(symbol, consolidator)
            self.crypto_data[symbol]['consolidator'] = consolidator
        except Exception as e:
            self.Debug(f"Warning: Could not add consolidator for {symbol.Value} - {e}")

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.crypto_data:
                self._initialize_symbol(symbol)
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            # Remove consolidator before liquidating/cleaning up
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
        # Initialize daily equity tracker
        if self._daily_start_equity is None:
            self._daily_start_equity = self.Portfolio.TotalPortfolioValue
        
        # Check daily loss limit
        current_equity = self.Portfolio.TotalPortfolioValue
        daily_loss_pct = (current_equity - self._daily_start_equity) / self._daily_start_equity
        if daily_loss_pct < self._daily_loss_limit:
            self.Debug(f"⚠️ DAILY LOSS LIMIT: {daily_loss_pct:.2%} | Pausing trades")
            return
        
        # Check if in cash mode
        if self._cash_mode_until is not None and self._cash_mode_until > self.Time:
            return
        # === Fear & Greed data ===
        if hasattr(self, 'fear_greed_symbol') and self.fear_greed_symbol and data.ContainsKey(self.fear_greed_symbol):
            fg = data[self.fear_greed_symbol]
            if fg is not None:
                self.fear_greed_value = fg.Value
        # === BTC market context update from 1-minute bars ===
        if self.btc_symbol is not None and data.Bars.ContainsKey(self.btc_symbol):
            btc_bar = data.Bars[self.btc_symbol]
            btc_price = float(btc_bar.Close)
            if len(self.btc_prices) > 0:
                btc_return = (btc_price - self.btc_prices[-1]) / self.btc_prices[-1]
                self.btc_returns.append(btc_return)
            self.btc_prices.append(btc_price)
            self.btc_ema_24.Update(btc_bar.EndTime, btc_price)
            if len(self.btc_returns) >= 10:
                self.btc_volatility.append(np.std(list(self.btc_returns)[-10:]))
        # Update quote data (bid/ask) from 1-min bars; indicator updates use the 5-min consolidator.
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
                self._cancel_stale_orders()
            sync_existing_positions(self)
            self._positions_synced = True
            self._first_post_warmup = False
            # Assume online if status never set after warmup
            if self.kraken_status == "unknown":
                self.kraken_status = "online"
                self.Debug("Fallback: kraken_status set to online after warmup")
            ready_count = sum(1 for c in self.crypto_data.values() if self._is_ready(c))
            self.Debug(f"Post-warmup: {ready_count} symbols ready")
        self._update_market_context()
        self.Rebalance()
        self.CheckExits()

    def _update_symbol_data(self, symbol, bar, quote_bar=None):
        crypto = self.crypto_data[symbol]
        price = float(bar.Close)
        high = float(bar.High)
        low = float(bar.Low)
        volume = float(bar.Volume)
        crypto['prices'].append(price)
        crypto['highs'].append(high)
        crypto['lows'].append(low)
        if crypto['last_price'] > 0:
            ret = (price - crypto['last_price']) / crypto['last_price']
            crypto['returns'].append(ret)
        crypto['last_price'] = price
        crypto['volume'].append(volume)
        crypto['dollar_volume'].append(price * volume)
        if len(crypto['volume']) >= self.short_period:
            crypto['volume_ma'].append(np.mean(list(crypto['volume'])[-self.short_period:]))
        crypto['ema_ultra_short'].Update(bar.EndTime, price)
        crypto['ema_short'].Update(bar.EndTime, price)
        crypto['ema_medium'].Update(bar.EndTime, price)
        crypto['ema_5'].Update(bar.EndTime, price)
        crypto['atr'].Update(bar)
        crypto['adx'].Update(bar)
        crypto['vwap_pv'].append(price * volume)
        crypto['vwap_v'].append(volume)
        total_v = sum(crypto['vwap_v'])
        if total_v > 0:
            crypto['vwap'] = sum(crypto['vwap_pv']) / total_v
        crypto['volume_long'].append(volume)
        if len(crypto['vwap_v']) >= 5 and crypto['vwap'] > 0:
            vwap_val = crypto['vwap']
            pv_list = list(crypto['vwap_pv'])
            v_list = list(crypto['vwap_v'])
            bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
            if len(bar_prices) >= 5:
                sd = float(np.std(bar_prices))
                crypto['vwap_sd'] = sd
                crypto['vwap_sd2_lower'] = vwap_val - 2.0 * sd
                crypto['vwap_sd3_lower'] = vwap_val - 3.0 * sd
        if len(crypto['returns']) >= 10:
            crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))
        crypto['rsi'].Update(bar.EndTime, price)
        if len(crypto['returns']) >= self.short_period and len(self.btc_returns) >= self.short_period:
            coin_ret = np.sum(list(crypto['returns'])[-self.short_period:])
            btc_ret = np.sum(list(self.btc_returns)[-self.short_period:])
            crypto['rs_vs_btc'].append(coin_ret - btc_ret)
        if len(crypto['prices']) >= self.medium_period:
            prices_arr = np.array(list(crypto['prices'])[-self.medium_period:])
            std = np.std(prices_arr)
            mean = np.mean(prices_arr)
            if std > 0:
                crypto['bb_upper'].append(mean + 2 * std)
                crypto['bb_lower'].append(mean - 2 * std)
                crypto['bb_width'].append(4 * std / mean if mean > 0 else 0)
        sp = get_spread_pct(self, symbol)
        if sp is not None:
            crypto['spreads'].append(sp)
        if quote_bar is not None:
            try:
                bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
                ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
                if bid_sz > 0 or ask_sz > 0:
                    crypto['bid_size'] = bid_sz
                    crypto['ask_size'] = ask_sz
            except Exception:
                pass

    def _on_five_minute_bar(self, sender, bar):
        """Called every 5 minutes with consolidated bar data.
        Note: quote data (bid/ask sizes) is updated separately in OnData() from 1-min bars.
        """
        symbol = bar.Symbol
        if symbol not in self.crypto_data:
            return
        # Process the 5-min bar; bid/ask updates happen separately in OnData.
        self._update_symbol_data(symbol, bar)

    def _update_market_context(self):
        if len(self.btc_prices) >= 48:
            btc_arr = np.array(list(self.btc_prices))
            current_btc = btc_arr[-1]
            btc_mom_12 = np.mean(list(self.btc_returns)[-12:]) if len(self.btc_returns) >= 12 else 0.0
            btc_sma = np.mean(btc_arr[-48:])
            if current_btc > btc_sma * 1.02:
                new_regime = "bull"
            elif current_btc < btc_sma * 0.98:
                new_regime = "bear"
            else:
                new_regime = "sideways"
            if new_regime == "sideways" and len(self.btc_returns) >= 12:
                if btc_mom_12 > 0.0001:
                    new_regime = "bull"
                elif btc_mom_12 < -0.0001:
                    new_regime = "bear"
            # Hysteresis: only change if held for 3+ bars
            if new_regime != self.market_regime:
                self._regime_hold_count += 1
                if self._regime_hold_count >= 3:
                    self.market_regime = new_regime
                    self._regime_hold_count = 0
            else:
                self._regime_hold_count = 0
        if len(self.btc_volatility) >= 5:
            current_vol = self.btc_volatility[-1]
            avg_vol = np.mean(list(self.btc_volatility))
            if current_vol > avg_vol * 1.5:
                self.volatility_regime = "high"
            elif current_vol < avg_vol * 0.5:
                self.volatility_regime = "low"
            else:
                self.volatility_regime = "normal"
        uptrend_count = 0
        total_ready = 0
        for crypto in self.crypto_data.values():
            if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
                total_ready += 1
                if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                    uptrend_count += 1
        if total_ready > 5:
            self.market_breadth = uptrend_count / total_ready

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
        """Evaluate long signals only. Short scoring disabled (Cash account)."""
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

    def _calculate_composite_score(self, factors, crypto=None):
        """Return the pre-computed scalp score."""
        return factors.get('_scalp_score', 0.0)

    def _apply_fee_adjustment(self, score):
        """Return score unchanged – signal thresholds already require >1% moves."""
        return score

    def _calculate_position_size(self, score, threshold, asset_vol_ann):
        """Aggressive 70% base size, Kelly-adjusted, bear-halved."""
        return self._scoring_engine.calculate_position_size(score, threshold, asset_vol_ann)

    def _kelly_fraction(self):
        return kelly_fraction(self)

    def _get_max_daily_trades(self):
        return self.max_daily_trades

    def _get_threshold(self):
        return self.entry_threshold

    def _check_correlation(self, new_symbol):
        """Reject candidate if it is too correlated with any existing position (item 8)."""
        if not self.entry_prices:
            return True
        new_crypto = self.crypto_data.get(new_symbol)
        if not new_crypto or len(new_crypto['returns']) < 24:
            return True
        new_rets = np.array(list(new_crypto['returns'])[-24:])
        if np.std(new_rets) < 1e-10:
            return True
        for sym in list(self.entry_prices.keys()):
            if sym == new_symbol:
                continue
            existing = self.crypto_data.get(sym)
            if not existing or len(existing['returns']) < 24:
                continue
            exist_rets = np.array(list(existing['returns'])[-24:])
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
        """Returns True if the portfolio has dropped >= 5% from today's open value."""
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

        if self._daily_loss_exceeded():
            self._log_skip("max daily loss exceeded")
            return

        # BTC weakness: reduce position sizes; only hard-block on severe crashes (>4% drop in 5 bars)
        self._btc_dump_size_mult = 1.0
        if len(self.btc_returns) >= 5:
            btc_5bar_return = sum(list(self.btc_returns)[-5:])
            if btc_5bar_return < -0.04:
                self._log_skip("BTC crashing")
                return
            elif btc_5bar_return < -0.02:
                self._btc_dump_size_mult = 0.50  # Half size during BTC weakness
        
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
        if self.daily_trade_count >= self._get_max_daily_trades():
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
        # Circuit breaker: exponential backoff after 6 consecutive losses
        if self.consecutive_losses >= 6:
            self._circuit_breaker_trigger_count += 1
            backoff_hours = min(1 * (2 ** (self._circuit_breaker_trigger_count - 1)), 8)  # 1h, 2h, 4h, 8h max
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
        if fg_value >= 85:
            effective_max_pos = max(1, self.max_positions // 2)
            if pos_count >= effective_max_pos:
                self._log_skip(f"Fear&Greed extreme greed ({fg_value}) — reduced max positions")
                return
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            self._log_skip("too many open orders")
            return

        count_scored = 0
        count_above_thresh = 0
        scores = []
        threshold_now = self._get_threshold()
        # Regime-adaptive threshold: slightly higher conviction needed in unfavorable conditions.
        if self.market_regime == "bear":
            threshold_now = max(threshold_now, 0.50)
        elif self.market_regime == "sideways":
            threshold_now = max(threshold_now, 0.45)
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

            composite_score = self._calculate_composite_score(factor_scores, crypto)
            net_score = self._apply_fee_adjustment(composite_score)

            crypto['recent_net_scores'].append(net_score)

            # RS override: altcoins showing relative strength vs BTC enter at base threshold.
            effective_threshold = threshold_now
            if self.market_regime in ("bear", "sideways") and len(crypto.get('rs_vs_btc', [])) >= 3:
                recent_rs = sum(list(crypto['rs_vs_btc'])[-3:])
                if recent_rs > 0.005:  # 0.5% cumulative RS outperformance vs BTC
                    effective_threshold = self.entry_threshold  # base 0.40 instead of elevated

            if net_score >= effective_threshold:
                count_above_thresh += 1
                scores.append({
                    'symbol': symbol,
                    'composite_score': composite_score,
                    'net_score': net_score,
                    'factors': factor_scores,
                    'volatility': crypto['volatility'][-1] if len(crypto['volatility']) > 0 else 0.05,
                    'dollar_volume': list(crypto['dollar_volume'])[-6:] if len(crypto['dollar_volume']) >= 6 else [],
                })

        try:
            cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            cash = self.Portfolio.Cash

        debug_limited(self, f"REBALANCE: {count_above_thresh}/{count_scored} above thresh={threshold_now:.2f} | cash=${cash:.2f}")

        if len(scores) == 0:
            self._log_skip("no candidates passed filters")
            return
        scores.sort(key=lambda x: x['net_score'], reverse=True)
        self._last_skip_reason = None
        self._execute_trades(scores, threshold_now)

    def _get_open_buy_orders_value(self):
        return get_open_buy_orders_value(self)

    def _execute_trades(self, candidates, threshold_now):
        if not self._positions_synced:
            return
        if self.LiveMode and self.kraken_status in ("maintenance", "cancel_only"):
            return
        cancel_stale_new_orders(self)
        if len(self.Transactions.GetOpenOrders()) >= self.max_concurrent_open_orders:
            return
        if self._compute_portfolio_risk_estimate() > self.portfolio_vol_cap:
            return
        
        try:
            available_cash = self.Portfolio.CashBook["USD"].Amount
        except (KeyError, AttributeError):
            available_cash = self.Portfolio.Cash
        
        open_buy_orders_value = self._get_open_buy_orders_value()
        
        if available_cash <= 0:
            debug_limited(self, f"SKIP TRADES: No cash available (${available_cash:.2f})")
            return
        if open_buy_orders_value > available_cash * self.open_orders_cash_threshold:
            debug_limited(self, f"SKIP TRADES: ${open_buy_orders_value:.2f} reserved (>{self.open_orders_cash_threshold:.0%} of ${available_cash:.2f})")
            return
        
        reject_pending_orders = 0
        reject_open_orders = 0
        reject_already_invested = 0
        reject_spread = 0
        reject_exit_cooldown = 0
        reject_loss_cooldown = 0
        reject_correlation = 0
        reject_price_invalid = 0
        reject_price_too_low = 0
        reject_cash_reserve = 0
        reject_min_qty_too_large = 0
        reject_dollar_volume = 0
        reject_notional = 0
        success_count = 0

        for cand in candidates:
            if self.daily_trade_count >= self._get_max_daily_trades():
                break
            if get_actual_position_count(self) >= self.max_positions:
                break
            sym = cand['symbol']
            net_score = cand.get('net_score', 0.5)
            if sym in self._pending_orders and self._pending_orders[sym] > 0:
                reject_pending_orders += 1
                continue
            if has_open_orders(self, sym):
                reject_open_orders += 1
                continue
            if is_invested_not_dust(self, sym):
                reject_already_invested += 1
                continue
            if not spread_ok(self, sym):
                reject_spread += 1
                continue
            if sym in self._exit_cooldowns and self.Time < self._exit_cooldowns[sym]:
                reject_exit_cooldown += 1
                continue
            if sym.Value in self._symbol_entry_cooldowns and self.Time < self._symbol_entry_cooldowns[sym.Value]:
                reject_loss_cooldown += 1
                continue
            if sym in self._symbol_loss_cooldowns and self.Time < self._symbol_loss_cooldowns[sym]:
                reject_loss_cooldown += 1
                continue
            if not self._check_correlation(sym):
                reject_correlation += 1
                continue
            sec = self.Securities[sym]
            price = sec.Price
            if price is None or price <= 0:
                reject_price_invalid += 1
                continue
            if price < self.min_price_usd:
                reject_price_too_low += 1
                continue

            try:
                available_cash = self.Portfolio.CashBook["USD"].Amount
            except (KeyError, AttributeError):
                available_cash = self.Portfolio.Cash

            available_cash = max(0, available_cash - open_buy_orders_value)
            total_value = self.Portfolio.TotalPortfolioValue
            # Minimal fee reserve only
            fee_reserve = max(total_value * self.cash_reserve_pct, 0.50)
            reserved_cash = available_cash - fee_reserve
            if reserved_cash <= 0:
                reject_cash_reserve += 1
                continue

            min_qty = get_min_quantity(self, sym)
            min_notional_usd = get_min_notional_usd(self, sym)
            if min_qty * price > reserved_cash * 0.90:
                reject_min_qty_too_large += 1
                continue

            crypto = self.crypto_data.get(sym)
            if not crypto:
                continue

            if crypto.get('trade_count_today', 0) >= self.max_symbol_trades_per_day:
                continue

            atr_val = crypto['atr'].Current.Value if crypto['atr'].IsReady else None
            if atr_val and price > 0:
                expected_move_pct = (atr_val * self.atr_tp_mult) / price
                min_profit_gate = self.min_expected_profit_pct
                min_required = self.expected_round_trip_fees + self.fee_slippage_buffer + min_profit_gate
                if expected_move_pct < min_required:
                    continue

            if len(crypto['dollar_volume']) >= 3:
                dv_window = min(len(crypto['dollar_volume']), 12)
                recent_dv = np.mean(list(crypto['dollar_volume'])[-dv_window:])
                dv_threshold = self.min_dollar_volume_usd
                if recent_dv < dv_threshold:
                    reject_dollar_volume += 1
                    continue

            vol = self._annualized_vol(crypto)
            size = self._calculate_position_size(net_score, threshold_now, vol)

            if self._consecutive_loss_halve_remaining > 0:
                size *= 0.50

            if self.volatility_regime == "high":
                size = min(size * 1.1, self.position_size_pct)

            # Per-symbol penalty: halve size after consecutive losses
            sym_val = sym.Value
            if sym_val in self._symbol_performance:
                recent_pnls = list(self._symbol_performance[sym_val])
                if len(recent_pnls) >= self.symbol_penalty_threshold:
                    recent_losses = sum(1 for p in recent_pnls[-self.symbol_penalty_threshold:] if p < 0)
                    if recent_losses == self.symbol_penalty_threshold:
                        size *= self.symbol_penalty_size_mult
                        self.Debug(f"PENALTY: {sym_val} size halved ({self.symbol_penalty_threshold} consecutive losses)")

            slippage_penalty = get_slippage_penalty(self, sym)
            size *= slippage_penalty

            # Apply BTC weakness size reduction
            size *= self._btc_dump_size_mult

            # Spread penalty: linear reduction for spreads above 0.2%
            current_spread = get_spread_pct(self, sym)
            if current_spread is not None and current_spread > 0.002:  # > 0.2% spread
                # 1% penalty per 0.05% excess spread, floored at 50%
                spread_penalty = max(0.5, 1.0 - (current_spread - 0.002) * 20)
                size *= spread_penalty

            val = reserved_cash * size

            # Liquidity cap: max 1% of estimated daily dollar volume (288 5-min bars/day)
            FIVE_MIN_BARS_PER_DAY = 288
            if len(crypto['dollar_volume']) >= 3:
                dv_window = min(len(crypto['dollar_volume']), 12)
                recent_dv = np.mean(list(crypto['dollar_volume'])[-dv_window:])
                estimated_daily_dv = recent_dv * FIVE_MIN_BARS_PER_DAY
                liquidity_cap = estimated_daily_dv * self.max_participation_rate
            else:
                liquidity_cap = float('inf')
            val = min(val, liquidity_cap)

            val = max(val, self.min_notional)
            val = min(val, self.Portfolio.TotalPortfolioValue * self.max_position_pct)

            qty = round_quantity(self, sym, val / price)
            if qty < min_qty:
                qty = round_quantity(self, sym, min_qty)
                val = qty * price
            total_cost_with_fee = val * 1.006
            if total_cost_with_fee > available_cash:
                reject_cash_reserve += 1
                continue
            if val < min_notional_usd * self.min_notional_fee_buffer or val < self.min_notional or val > reserved_cash:
                reject_notional += 1
                continue

            try:
                sec = self.Securities[sym]
                min_order_size = float(sec.SymbolProperties.MinimumOrderSize or 0)
                lot_size = float(sec.SymbolProperties.LotSize or 0)
                actual_min = max(min_order_size, lot_size)
                if actual_min > 0 and qty < actual_min:
                    self.Debug(f"REJECT ENTRY {sym.Value}: qty={qty} < min_order_size={actual_min} (unsellable)")
                    reject_notional += 1
                    continue
                if min_order_size > 0:
                    post_fee_qty = qty * (1.0 - KRAKEN_SELL_FEE_BUFFER)
                    if post_fee_qty < min_order_size:
                        required_qty = round_quantity(self, sym, min_order_size / (1.0 - KRAKEN_SELL_FEE_BUFFER))
                        if required_qty * price <= available_cash * 0.99:  # 1% safety margin
                            qty = required_qty
                            val = qty * price
                        else:
                            self.Debug(f"REJECT ENTRY {sym.Value}: post-fee qty={post_fee_qty:.6f} < min_order_size={min_order_size} and can't upsize")
                            reject_notional += 1
                            continue
            except Exception as e:
                self.Debug(f"Warning: could not check min_order_size for {sym.Value}: {e}")

            if not intraday_volume_ok(self, sym, val):
                continue

            try:
                ticket = place_limit_or_market(self, sym, qty, timeout_seconds=30, tag="Entry")
                if ticket is not None:
                    self._recent_tickets.append(ticket)
                    components = cand.get('factors', {})
                    sig_str = (f"vol={components.get('vol_ignition', 0):.2f} "
                               f"mean_rev={components.get('mean_reversion', 0):.2f} "
                               f"vwap={components.get('vwap_signal', 0):.2f}")
                    self.Debug(f"SCALP ENTRY: {sym.Value} | score={net_score:.2f} | ${val:.2f} | {sig_str}")
                    # Track signal combination for attribution
                    combo_parts = []
                    if components.get('vol_ignition', 0) >= 0.10:
                        combo_parts.append('vol')
                    if components.get('mean_reversion', 0) >= 0.10:
                        combo_parts.append('mean_rev')
                    if components.get('vwap_signal', 0) >= 0.10:
                        combo_parts.append('vwap')
                    signal_combo = '+'.join(combo_parts) if combo_parts else 'none'
                    self._entry_signal_combos[sym] = signal_combo
                    success_count += 1
                    self.trade_count += 1
                    crypto['trade_count_today'] = crypto.get('trade_count_today', 0) + 1
                    adx_ind = crypto.get('adx')
                    is_choppy = (adx_ind is not None and adx_ind.IsReady
                                 and adx_ind.Current.Value < 25)
                    self._choppy_regime_entries[sym] = is_choppy
                    if self._consecutive_loss_halve_remaining > 0:
                        self._consecutive_loss_halve_remaining -= 1
                    if self.LiveMode:
                        self._last_live_trade_time = self.Time
            except Exception as e:
                self.Debug(f"ORDER FAILED: {sym.Value} - {e}")
                self._session_blacklist.add(sym.Value)
                continue
            if self.LiveMode and success_count >= 3:
                break

        if success_count > 0 or (reject_exit_cooldown + reject_loss_cooldown) > 3:
            debug_limited(self, f"EXECUTE: {success_count}/{len(candidates)} | rejects: cd={reject_exit_cooldown} loss={reject_loss_cooldown} corr={reject_correlation} dv={reject_dollar_volume}")

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

        # --- Partial TP (single tier) ---
        tier = self._partial_tp_tier.get(symbol, 0)
        if tier == 0 and pnl >= self.partial_tp_tier1_threshold:
            if partial_smart_sell(self, symbol, self.partial_tp_tier1_pct, "Partial TP"):
                self._partial_tp_tier[symbol] = 1
                self._partial_tp_taken[symbol] = True
                self.Debug(f"PARTIAL TP: {symbol.Value} | PnL:{pnl:+.2%}")
                return

        # --- Exit tag determination ---
        tag = ""

        # 1. Stop Loss
        if pnl <= -sl:
            tag = "Stop Loss"

        # 2. Take Profit (only if no partial TP taken)
        if not tag and not self._partial_tp_taken.get(symbol, False) and pnl >= tp:
            tag = "Take Profit"

        # 3. Trailing Stop
        if not tag and pnl > trailing_activation and dd >= trailing_stop_pct:
            tag = "Trailing Stop"

        # 4. ATR Trail
        if not tag and atr and entry > 0 and holding.Quantity > 0:
            effective_trail_mult = self.atr_trail_mult
            if self._partial_tp_taken.get(symbol, False):
                effective_trail_mult = self.post_partial_tp_trail_mult
            trail_offset = atr * effective_trail_mult
            trail_level = highest - trail_offset
            if crypto:
                crypto['trail_stop'] = trail_level
            if minutes >= self.min_trail_hold_minutes and price <= trail_level:
                tag = "ATR Trail"

        # 5. Time Stop
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
                self.Debug(f"{tag}: {symbol.Value} | PnL:{pnl:+.2%} | Held:{hours:.1f}h")
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
        total = self.winning_trades + self.losing_trades
        wr = self.winning_trades / total if total > 0 else 0
        self.Debug("=== FINAL ===")
        self.Debug(f"Trades: {self.trade_count} | WR: {wr:.1%}")
        self.Debug(f"Final: ${self.Portfolio.TotalPortfolioValue:.2f}")
        self.Debug(f"PnL: {self.total_pnl:+.2%}")
        persist_state(self)
        if hasattr(self, 'pnl_by_regime') and self.pnl_by_regime:
            self.Debug("=== PnL BY REGIME ===")
            for regime, pnls in self.pnl_by_regime.items():
                avg = np.mean(pnls) if pnls else 0
                wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
                self.Debug(f"  {regime}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
        if hasattr(self, 'pnl_by_vol_regime') and self.pnl_by_vol_regime:
            self.Debug("=== PnL BY VOL REGIME ===")
            for vol_regime, pnls in self.pnl_by_vol_regime.items():
                avg = np.mean(pnls) if pnls else 0
                wr = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
                self.Debug(f"  {vol_regime}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%}")
        # Exit-tag performance summary
        if hasattr(self, 'pnl_by_tag') and self.pnl_by_tag:
            self.Debug("=== PnL BY EXIT TAG ===")
            for tag, pnls in sorted(self.pnl_by_tag.items()):
                if len(pnls) == 0:
                    continue
                avg = sum(pnls) / len(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                total = sum(pnls)
                self.Debug(f"  {tag}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
        # Signal-combination performance summary
        if hasattr(self, 'pnl_by_signal_combo') and self.pnl_by_signal_combo:
            self.Debug("=== PnL BY SIGNAL COMBO ===")
            for combo, pnls in sorted(self.pnl_by_signal_combo.items()):
                if len(pnls) == 0:
                    continue
                avg = sum(pnls) / len(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                total = sum(pnls)
                self.Debug(f"  {combo}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")
        # Hold-time performance summary
        if hasattr(self, 'pnl_by_hold_time') and self.pnl_by_hold_time:
            self.Debug("=== PnL BY HOLD TIME ===")
            for bucket in ['<30min', '30min-2h', '2h-6h', '6h+', 'unknown']:
                pnls = self.pnl_by_hold_time.get(bucket)
                if not pnls:
                    continue
                avg = sum(pnls) / len(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                total = sum(pnls)
                self.Debug(f"  {bucket}: {len(pnls)} trades | WR:{wr:.0%} | Avg:{avg:+.3%} | Total:{total:+.3%}")

    def DailyReport(self):
        if self.IsWarmingUp: return
        daily_report(self)
