"""
chop_engine.py — Dedicated Choppy-Market Trading Engine

Operates in range-bound / sideways market conditions as classified by
RegimeRouter.  Uses its own signal logic, entry scoring, and risk profile
entirely separate from the trend-oriented MicroScalpEngine.

Design principles
-----------------
- Do NOT simply make the trend engine more permissive.  This engine uses
  mean-reversion logic suited to sideways markets, not volume-breakout logic.
- Signals are additive (same 0–0.60 scale as MicroScalpEngine for comparability).
- Risk profile is deliberately smaller/tighter than trend trades to match the
  lower expected move in chop conditions.
- Anti-overtrading guardrails: per-symbol cooldown after failed fades and a
  daily trade count cap.

Signals (max 0.60 total)
-------------------------
1. range_reversion  (0.20 max)
   Price within RANGE_REVERSION_PCT of the 5-bar low AND RSI is oversold.
   Strongest signal for sideways reversion (price bouncing off range floor).

2. vwap_stretch     (0.20 max)
   Price stretched below VWAP by at least VWAP_STRETCH_SD standard deviations
   and not below VWAP_FLOOR_SD (avoids catching free-falling prices).
   Higher credit when price is near the -2 SD level (cleaner institutional level).

3. failed_breakout  (0.10 max)
   A volume spike occurred in the last FAILED_BKO_LOOKBACK bars but price
   subsequently retreated below the spike's high — failed breakout reclaim.

4. bb_reversion     (0.10 max)
   Price at or below the lower Bollinger Band AND RSI is deeply oversold.

Entry gate
----------
At least MIN_SIGNAL_COUNT (2) signals must be active (value >= 0.10) for any
entry.  This prevents single-signal "catching a falling knife" entries.

Risk profile (all configurable)
--------------------------------
Position size:  CHOP_SIZE_BASE – CHOP_SIZE_MAX fraction of available cash
                (default 15–25 %), smaller than trend's 30–50 %.
Stop-loss:      max(ATR × CHOP_ATR_SL_MULT / price, CHOP_FIXED_SL)
                default max(ATR × 1.5 / price, 3 %) — tighter than trend (8 %).
Take-profit:    max(ATR × CHOP_ATR_TP_MULT / price, CHOP_FIXED_TP)
                default max(ATR × 2.0 / price, 1.0 %) — smaller than trend (4×ATR).
Max hold time:  CHOP_MAX_HOLD_HOURS (default 2 h) — shorter than trend (8 h).
Fail cooldown:  CHOP_FAIL_COOLDOWN_MINUTES (default 30 min) per symbol after
                a stop-loss is hit — prevents rapid re-entry into losing fades.
Daily cap:      CHOP_MAX_TRADES_PER_SYMBOL_DAY per symbol per day.
"""

from datetime import timedelta
from collections import deque
import math


class ChopEngine:
    """
    Choppy-market signal engine and risk manager.

    Called from main.py when RegimeRouter.route() == 'chop'.
    """

    # ── Signal thresholds ─────────────────────────────────────────────────
    # Range-reversion
    RANGE_REVERSION_PCT     = 0.005   # price within 0.5 % of 5-bar low
    RSI_DEEP_OVERSOLD       = 30      # RSI < 30 → strong reversion signal
    RSI_OVERSOLD            = 40      # RSI < 40 → moderate reversion signal

    # VWAP stretch
    VWAP_STRETCH_SD         = 1.5     # price < VWAP - 1.5 SD triggers signal
    VWAP_FLOOR_SD           = 3.5     # price < VWAP - 3.5 SD → ignore (free fall)

    # Failed breakout lookback
    FAILED_BKO_LOOKBACK     = 3       # bars to scan for a failed breakout
    FAILED_BKO_VOL_MULT     = 2.5     # volume spike threshold for breakout detection

    # BB reversion
    BB_NEAR_LOWER_PCT       = 0.005   # price within 0.5 % of lower Bollinger Band

    # Minimum bar dollar-volume floor (same as trend engine, prevents junk signals)
    MIN_BAR_DOLLAR_VOL      = 1000.0

    # ── Risk parameters ───────────────────────────────────────────────────
    CHOP_SIZE_BASE               = 0.15  # 15 % of available cash at threshold score
    CHOP_SIZE_MAX                = 0.25  # 25 % at maximum score
    CHOP_SIZE_MIN_NOTIONAL       = 5.5   # hard minimum USD notional (same as algo)

    CHOP_ATR_SL_MULT             = 1.5   # ATR multiplier for stop-loss
    CHOP_FIXED_SL                = 0.03  # 3 % hard stop floor

    CHOP_ATR_TP_MULT             = 2.0   # ATR multiplier for take-profit
    CHOP_FIXED_TP                = 0.010 # 1.0 % minimum take-profit

    CHOP_MAX_HOLD_HOURS          = 2.0   # hours before time-stop fires
    CHOP_TIME_STOP_MIN_PNL       = 0.002 # time-stop only fires if PnL < 0.2 %

    CHOP_FAIL_COOLDOWN_MINUTES   = 30    # cooldown after a failed fade (stop hit)
    CHOP_MAX_TRADES_PER_SYMBOL_DAY = 3   # anti-overtrading: max chop trades/symbol/day

    # ── Entry gate ────────────────────────────────────────────────────────
    MIN_SIGNAL_COUNT        = 2     # require at least 2 active signals (>= 0.10)
    CHOP_ENTRY_THRESHOLD    = 0.25  # minimum aggregate score for a chop entry

    # Canonical signal keys (used for attribution logging)
    SIGNAL_KEYS = ('range_reversion', 'vwap_stretch', 'failed_breakout', 'bb_reversion')

    def __init__(self, algo):
        self.algo = algo

        # Per-symbol state
        self._fail_cooldowns    = {}   # symbol → datetime expiry of fail cooldown
        self._chop_entries      = {}   # symbol → True when opened by chop engine
        self._chop_entry_times  = {}   # symbol → entry datetime
        self._daily_chop_counts = {}   # symbol.Value → int (trade count today)

    # ------------------------------------------------------------------ #
    #  Lifecycle helpers                                                   #
    # ------------------------------------------------------------------ #

    def reset_daily_counts(self):
        """Call at the start of each trading day (ResetDailyCounters)."""
        self._daily_chop_counts.clear()

    def is_in_fail_cooldown(self, symbol) -> bool:
        """True if this symbol is cooling down after a failed chop fade."""
        expiry = self._fail_cooldowns.get(symbol)
        if expiry is None:
            return False
        if self.algo.Time >= expiry:
            self._fail_cooldowns.pop(symbol, None)
            return False
        return True

    def register_entry(self, symbol):
        """Record that a chop trade was entered on this symbol."""
        self._chop_entries[symbol]     = True
        self._chop_entry_times[symbol] = self.algo.Time
        sym_val = symbol.Value
        self._daily_chop_counts[sym_val] = self._daily_chop_counts.get(sym_val, 0) + 1

    def register_exit(self, symbol, exit_tag: str, pnl: float):
        """
        Record that a chop trade exited.  Apply fail cooldown on losses.
        Call from events.py when a chop position fills on the sell side.
        """
        self._chop_entries.pop(symbol, None)
        self._chop_entry_times.pop(symbol, None)
        if pnl < 0 or "Stop" in exit_tag:
            expiry = self.algo.Time + timedelta(minutes=self.CHOP_FAIL_COOLDOWN_MINUTES)
            self._fail_cooldowns[symbol] = expiry
            self.algo.Debug(
                f"CHOP FAIL COOLDOWN: {symbol.Value} | "
                f"PnL:{pnl:+.2%} | tag:{exit_tag} | "
                f"cooldown {self.CHOP_FAIL_COOLDOWN_MINUTES}min"
            )

    def is_chop_position(self, symbol) -> bool:
        """True if this symbol was opened by the chop engine."""
        return self._chop_entries.get(symbol, False)

    def daily_trade_count(self, symbol) -> int:
        """Return the number of chop trades placed today for this symbol."""
        return self._daily_chop_counts.get(symbol.Value, 0)

    # ------------------------------------------------------------------ #
    #  Signal scoring                                                      #
    # ------------------------------------------------------------------ #

    def calculate_score(self, crypto) -> tuple:
        """
        Compute the chop-engine aggregate score for a symbol.

        Returns
        -------
        (score, components)
            score      : float in [0.0, 0.60]  (0 if gate fails)
            components : dict  signal_name → contribution
        """
        components = {
            'range_reversion': 0.0,
            'vwap_stretch':    0.0,
            'failed_breakout': 0.0,
            'bb_reversion':    0.0,
        }

        try:
            prices = crypto.get('prices', [])
            if len(prices) < 10:
                return 0.0, components

            price = float(prices[-1])
            if price <= 0:
                return 0.0, components

            # Dollar-volume floor: skip illiquid bars.
            volume_deque = crypto.get('volume', [])
            if len(volume_deque) >= 1:
                bar_dv = float(volume_deque[-1]) * price
                if bar_dv < self.MIN_BAR_DOLLAR_VOL:
                    return 0.0, components

            # ── Signal 1: Range-bound reversion ───────────────────────────
            # Price near the 5-bar rolling low + RSI showing oversold condition.
            lows    = crypto.get('lows', [])
            rsi_ind = crypto.get('rsi')
            if len(lows) >= 5 and rsi_ind is not None and rsi_ind.IsReady:
                lows_list  = list(lows)[-5:]
                recent_low = min(lows_list)
                rsi_val    = rsi_ind.Current.Value
                if recent_low > 0 and price <= recent_low * (1.0 + self.RANGE_REVERSION_PCT):
                    if rsi_val < self.RSI_DEEP_OVERSOLD:
                        components['range_reversion'] = 0.20
                    elif rsi_val < self.RSI_OVERSOLD:
                        components['range_reversion'] = 0.10

            # ── Signal 2: VWAP stretch / reversion ────────────────────────
            # Price stretched below VWAP by at least 1.5 SD — mean-reversion
            # opportunity. Avoid free-falling prices by enforcing a floor at
            # -3.5 SD (gap down / momentum continuation should not fire here).
            vwap        = crypto.get('vwap',          0.0)
            vwap_sd     = crypto.get('vwap_sd',       0.0)
            vwap_sd2_lower = crypto.get('vwap_sd2_lower', 0.0)
            if vwap > 0 and vwap_sd > 0:
                stretch_level = vwap - self.VWAP_STRETCH_SD * vwap_sd
                floor_level   = vwap - self.VWAP_FLOOR_SD   * vwap_sd
                if price <= stretch_level and price >= floor_level:
                    # Extra credit if price is near the cleaner -2 SD level.
                    if vwap_sd2_lower > 0 and price >= vwap_sd2_lower * 0.997:
                        components['vwap_stretch'] = 0.20
                    else:
                        components['vwap_stretch'] = 0.10

            # ── Signal 3: Failed breakout ──────────────────────────────────
            # A volume surge occurred recently but price retreated back below
            # the breakout high — classic chop pattern (reclaim failure).
            highs    = crypto.get('highs',    [])
            volume   = crypto.get('volume',   [])
            vol_long = crypto.get('volume_long', [])
            if (len(highs)  >= self.FAILED_BKO_LOOKBACK + 1
                    and len(volume) >= self.FAILED_BKO_LOOKBACK + 1
                    and len(vol_long) >= 60):
                vol_baseline = crypto.get('_vol_long_sum', 0.0) / len(vol_long)
                if vol_baseline > 0:
                    highs_scan = list(highs) [-(self.FAILED_BKO_LOOKBACK + 1):-1]
                    vol_scan   = list(volume)[-(self.FAILED_BKO_LOOKBACK + 1):-1]
                    had_spike  = any(v > vol_baseline * self.FAILED_BKO_VOL_MULT
                                     for v in vol_scan)
                    if had_spike and len(highs_scan) > 0:
                        spike_high = max(highs_scan)
                        if spike_high > 0 and price < spike_high * 0.998:
                            # Price failed to hold the breakout high.
                            components['failed_breakout'] = 0.10

            # ── Signal 4: BB reversion ─────────────────────────────────────
            # Price at or below the lower Bollinger Band AND deeply oversold.
            # Confirms range extreme without requiring a VWAP computation.
            bb_lower_deque = crypto.get('bb_lower', [])
            rsi_ind        = crypto.get('rsi')
            if (len(bb_lower_deque) >= 1
                    and rsi_ind is not None and rsi_ind.IsReady):
                bb_lower = float(bb_lower_deque[-1])
                rsi_val  = rsi_ind.Current.Value
                if bb_lower > 0 and price <= bb_lower * (1.0 + self.BB_NEAR_LOWER_PCT):
                    if rsi_val < self.RSI_DEEP_OVERSOLD:
                        components['bb_reversion'] = 0.10

        except Exception as e:
            self.algo.Debug(f"ChopEngine.calculate_score error: {e}")

        score = sum(components.values())
        score = min(score, 1.0)

        # Entry gate: require at least MIN_SIGNAL_COUNT active signals.
        active = sum(1 for v in components.values() if v >= 0.10)
        if active < self.MIN_SIGNAL_COUNT:
            score = 0.0

        return score, components

    # ------------------------------------------------------------------ #
    #  Position sizing                                                     #
    # ------------------------------------------------------------------ #

    def calculate_position_size(self, score: float) -> float:
        """
        Return the fraction of available cash to allocate to this chop trade.

        Scales linearly from CHOP_SIZE_BASE (at entry threshold) to
        CHOP_SIZE_MAX (at maximum possible score 0.60).

        Returns a fraction in [CHOP_SIZE_BASE, CHOP_SIZE_MAX].
        """
        max_score  = 0.60
        min_score  = self.CHOP_ENTRY_THRESHOLD
        score_range = max_score - min_score
        if score_range > 0:
            conviction = max(0.0, min(1.0, (score - min_score) / score_range))
        else:
            conviction = 1.0
        return self.CHOP_SIZE_BASE + (self.CHOP_SIZE_MAX - self.CHOP_SIZE_BASE) * conviction

    # ------------------------------------------------------------------ #
    #  Exit parameters                                                     #
    # ------------------------------------------------------------------ #

    def get_exit_params(self, symbol, entry_price: float, crypto) -> dict:
        """
        Return chop-specific exit parameters for an open position.

        Returns
        -------
        dict with keys:
            'sl'             : float  stop-loss fraction from entry
            'tp'             : float  take-profit fraction from entry
            'max_hold_hours' : float  maximum holding time
            'time_stop_min_pnl' : float  time-stop fires only below this PnL
        """
        atr = None
        if crypto and crypto.get('atr') and crypto['atr'].IsReady:
            atr = crypto['atr'].Current.Value

        if atr and entry_price > 0:
            sl = max((atr * self.CHOP_ATR_SL_MULT) / entry_price, self.CHOP_FIXED_SL)
            tp = max((atr * self.CHOP_ATR_TP_MULT) / entry_price, self.CHOP_FIXED_TP)
        else:
            sl = self.CHOP_FIXED_SL
            tp = self.CHOP_FIXED_TP

        # Ensure TP > SL to avoid immediately exiting on the first tick.
        if tp < sl * 1.1:
            tp = sl * 1.1

        return {
            'sl':               sl,
            'tp':               tp,
            'max_hold_hours':   self.CHOP_MAX_HOLD_HOURS,
            'time_stop_min_pnl': self.CHOP_TIME_STOP_MIN_PNL,
        }
