# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v8.2.0

    Simplified 3-signal scoring system with ADX regime filter.
    Strips kitchen-sink approach down to the 3 signals with proven edge.

    Score: 0.0 – 0.60 max (3 signals × 0.20 each).
      Gate 1: vol_ignition >= 0.10 required for trending entries; in choppy/sideways
              regime, mean_reversion >= 0.10 is required instead (regime-aware gate).
      Gate 2: at least MIN_SIGNAL_COUNT (default 2) active components required.
              Rejects vol-only entries; allows vol+mean_rev, vol+vwap, vol+mean_rev+vwap,
              or mean_rev+vwap in choppy regime.
      ADX filter: ADX > 25 AND DI- > DI+ × 1.2 → reject (strong bearish trend).
      EMA momentum cap: ema_ultra_short < ema_short → cap total score at 0.30.
      Vol-ignition dollar-volume floor: bar dollar volume < $1,000 → no vol score.
      VWAP SD bands: computed via true volume-weighted variance (fixed from np.std).

    Signals
    -------
    1. Volume Ignition (gate): 3× volume surge; 2.5× in choppy markets (ADX < 25);
       1.5× partial in choppy (was 1.0×); $1K dollar-volume floor per bar.
    2. Mean Reversion: RSI oversold + price near lower BB when ADX is low (max 0.20);
       BB compression and range-position gating applied in choppy/sideways regime.
    3. VWAP Reclaim / SD Band Bounce: price above VWAP×1.0015 (0.15% buffer) with
       EMA5 confirmation for full credit; partial credit 0.10 between VWAP and buffer;
       bouncing off -2/-3 SD lower band (max 0.20)

    Removed: OBI, MTF trend, steady grind, ADX trend, CVD divergence, Kalman reversion, BB squeeze
    """

    # Tunable signal thresholds (easy to adjust for backtesting)
    VOL_SURGE_STRONG        = 3.5    # 3.5× — was 4.0; captures more genuine breakouts (3.0–3.5× is significant)
    VOL_SURGE_PARTIAL       = 2.5    # 2.5× — was 3.0; partial credit starts earlier
    ADX_STRONG_THRESHOLD    = 14     # (kept for mean reversion gating)
    ADX_MODERATE_THRESHOLD  = 20     # moderate directional threshold (was 10)
    VWAP_BUFFER             = 1.0015  # 0.15% above VWAP for confirmed reclaim
    # Ranging-market mean reversion thresholds (used when ADX < ADX_MODERATE_THRESHOLD)
    RSI_OVERSOLD_THRESHOLD        = 45   # RSI < 45 → ranging-market entry
    RSI_MILDLY_OVERSOLD_THRESHOLD = 42   # was 50; tighter partial credit
    BB_NEAR_LOWER_PCT             = 0.03  # within 3% of lower Bollinger Band = near support
    # Canonical signal names — used here and by callers for attribution / logging.
    SIGNAL_KEYS = ('vol_ignition', 'mean_reversion', 'vwap_signal')
    # Entry-quality gate: minimum number of active signal components (value >= 0.10).
    # Default 2 rejects vol-only entries. Set via algo.min_signal_count to override.
    MIN_SIGNAL_COUNT              = 2

    def __init__(self, algorithm):
        self.algo = algorithm

    # ------------------------------------------------------------------
    # Primary entry: returns (score, components_dict)
    # ------------------------------------------------------------------
    def calculate_scalp_score(self, crypto):
        """
        Calculate the aggregate scalp score using 3 core signals + ADX filter.

        Returns
        -------
        (score, components) where score ∈ [0, 0.60] and components maps each
        signal name to its individual contribution (0.20 max each).

        Gate 1: vol_ignition >= 0.10 required in trending regime; in choppy/sideways,
                mean_reversion >= 0.10 is required instead.
        Gate 2: at least MIN_SIGNAL_COUNT active components (each >= 0.10) required.
                Rejects weak entries; allows vol+mean_rev, vol+vwap, mean_rev+vwap (in choppy), or all three.
        ADX filter: ADX > 25 AND DI- > DI+ × 1.2 → return (0.0, components).
        EMA momentum cap: ema_ultra_short < ema_short → cap score at 0.30.
        Max possible score: 0.20 (vol) + 0.20 (mean_rev) + 0.20 (vwap) = 0.60.
        """
        components = {
            'vol_ignition':   0.0,
            'mean_reversion': 0.0,
            'vwap_signal':    0.0,
        }
        _downtrend = False

        try:
            # ----------------------------------------------------------
            # GATE: Volume Ignition (must fire for any entry)
            # Current volume surge vs adaptive rolling baseline (Apr-Oct fix).
            # Uses 24h long-term average when available instead of a fixed
            # 20-bar (20-minute) window, so thresholds stay relevant during
            # low-volatility summer periods.
            # ADX regime filter: lower thresholds in choppy markets (ADX < 25)
            # to fire more trades when trend continuation is unlikely.
            # ----------------------------------------------------------
            if len(crypto['volume']) >= 20:
                current_vol = float(crypto['volume'][-1])   # deque supports -1 indexing
                # Use precomputed running sums for O(1) volume baseline (avoid list + np.mean)
                vol_long = crypto.get('volume_long', [])
                if len(vol_long) >= 60:
                    vol_baseline = crypto.get('_vol_long_sum', 0.0) / len(vol_long)
                elif (crypto.get('_vol20_window') is not None
                        and len(crypto['_vol20_window']) >= 20):
                    vol_baseline = crypto['_vol20_sum'] / 20
                else:
                    # Cold-start fallback: list conversion (rare after warmup)
                    vol_baseline = float(np.mean(list(crypto['volume'])[-20:]))
                # ADX regime filter: lower thresholds in choppy markets (ADX < 25)
                adx_indicator = crypto.get('adx')
                is_choppy = (adx_indicator is not None and adx_indicator.IsReady
                             and adx_indicator.Current.Value < 25)
                # Dollar-volume floor: ignore volume spikes on illiquid bars
                if len(crypto['prices']) >= 1:
                    bar_dollar_vol = current_vol * float(crypto['prices'][-1])
                else:
                    bar_dollar_vol = 0.0
                if bar_dollar_vol < 1000.0:
                    pass  # leave vol_ignition at 0.0 — spike is on an illiquid bar
                else:
                    vol_strong  = 2.5 if is_choppy else self.VOL_SURGE_STRONG   # was 1.8 in choppy
                    vol_partial = 1.5 if is_choppy else self.VOL_SURGE_PARTIAL  # was 1.0 in choppy
                    if vol_baseline > 0:
                        ratio = current_vol / vol_baseline
                        if ratio >= vol_strong:
                            components['vol_ignition'] = 0.20
                        elif ratio >= vol_partial:
                            # Partial credit for a meaningful volume spike
                            components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # ADX FILTER: reject strong bearish trends
            # Not additive to score — filter only.
            # ----------------------------------------------------------
            # SIGNAL: Mean Reversion
            # RSI oversold + price near lower BB when ADX is low (ranging/choppy).
            # Max contribution: 0.20 (strong), 0.15 (moderate), 0.10 (mild).
            # ----------------------------------------------------------
            adx_indicator = crypto.get('adx')
            if adx_indicator is not None and adx_indicator.IsReady:
                adx_val = adx_indicator.Current.Value
                di_plus = adx_indicator.PositiveDirectionalIndex.Current.Value
                di_minus = adx_indicator.NegativeDirectionalIndex.Current.Value
                if adx_val > 25 and di_minus > di_plus * 1.2:
                    # Strong bearish trend — mean reversion will fail, skip.
                    # Clear any partial vol_ignition so signal-attribution logging
                    # doesn't incorrectly record a 'vol' combo for a rejected bar.
                    components['vol_ignition'] = 0.0
                    return 0.0, components
                if adx_val <= self.ADX_STRONG_THRESHOLD:
                    rsi_ind = crypto.get('rsi')
                    bb_lower_data = crypto.get('bb_lower', [])
                    if (rsi_ind is not None and rsi_ind.IsReady
                            and len(bb_lower_data) >= 1 and len(crypto['prices']) >= 1):
                        rsi_val = rsi_ind.Current.Value
                        price = crypto['prices'][-1]
                        bb_lower = bb_lower_data[-1]
                        is_mild_oversold_ranging = (adx_val <= self.ADX_MODERATE_THRESHOLD
                                                    and rsi_val < self.RSI_MILDLY_OVERSOLD_THRESHOLD)
                        # BB compression check: only award full mean-rev credit when BB is compressed
                        bb_width_hist = crypto.get('bb_width', [])
                        bb_compressed = False
                        if len(bb_width_hist) >= 10:
                            width_list = list(bb_width_hist)
                            bb_compressed = width_list[-1] <= float(np.median(width_list[-20:]))
                        # Range-position gate: in sideways regime require price in lower 35% of range
                        range_pos = crypto.get('range_position', 0.5)
                        in_choppy_regime = (
                            getattr(self.algo, 'market_regime', '') == 'sideways'
                            or crypto.get('is_symbol_choppy', False)
                        )
                        range_ok = (not in_choppy_regime or range_pos <= 0.35)
                        if (self.algo.market_regime == 'sideways'
                                and bb_lower > 0 and price <= bb_lower * 1.005 and rsi_val < 35):
                            base_score = 0.20 if bb_compressed else 0.10
                            components['mean_reversion'] = base_score if range_ok else max(0.0, base_score - 0.10)
                        elif (adx_val <= self.ADX_MODERATE_THRESHOLD
                                and rsi_val < self.RSI_OVERSOLD_THRESHOLD
                                and bb_lower > 0
                                and price <= bb_lower * (1 + self.BB_NEAR_LOWER_PCT)):
                            base_score = 0.20 if bb_compressed else 0.10
                            components['mean_reversion'] = base_score if range_ok else max(0.0, base_score - 0.10)
                        elif is_mild_oversold_ranging:
                            base_score = 0.10
                            components['mean_reversion'] = base_score if range_ok else 0.0

            # EMA momentum filter: if ultra-short EMA is below short EMA (price trending down),
            # cap score to prevent weak entries in active downtrends.
            # Full mean-reversion entries (all 3 signals firing near max) can still pass at 0.30+.
            ema_us = crypto.get('ema_ultra_short')
            ema_sh = crypto.get('ema_short')
            _downtrend = (ema_us is not None and ema_sh is not None
                          and ema_us.IsReady and ema_sh.IsReady
                          and ema_us.Current.Value < ema_sh.Current.Value)

            # ----------------------------------------------------------
            # SIGNAL: VWAP Reclaim / SD Band Bounce
            # Price > rolling 20-bar VWAP → institutional buying support.
            # Price bouncing aggressively off -2 or -3 SD lower band →
            # extreme mean-reversion entry signal (higher score than bare
            # VWAP reclaim in some cases).
            # ----------------------------------------------------------
            vwap = crypto.get('vwap', 0.0)
            vwap_sd = crypto.get('vwap_sd', 0.0)
            vwap_sd2_lower = crypto.get('vwap_sd2_lower', 0.0)
            vwap_sd3_lower = crypto.get('vwap_sd3_lower', 0.0)
            ema5 = crypto.get('ema_5')
            ema6 = ema_sh  # reuse already-fetched ema_short indicator (EMA6)
            ema5_rising = (ema5 is not None and ema6 is not None
                           and ema5.IsReady and ema6.IsReady
                           and ema5.Current.Value > ema6.Current.Value)
            if vwap > 0 and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                if price > vwap * self.VWAP_BUFFER:
                    # Clearly above VWAP (0.15% buffer) — full credit only if EMA5 confirms uptrend
                    components['vwap_signal'] = 0.20 if ema5_rising else 0.10
                elif price > vwap:
                    # Marginally above VWAP — partial credit regardless of EMA
                    components['vwap_signal'] = 0.10
                elif (vwap_sd > 0 and vwap_sd3_lower > 0
                      and price >= vwap_sd3_lower * 1.005
                      and price < vwap_sd2_lower):
                    # Aggressive bounce off -3 SD lower band (extreme support)
                    components['vwap_signal'] = 0.20
                elif (vwap_sd > 0 and vwap_sd2_lower > 0
                      and price >= vwap_sd2_lower * 1.003):
                    # Bounce off -2 SD lower band (strong support)
                    components['vwap_signal'] = 0.15

        except Exception as e:
            self.algo.Debug(f"MicroScalpEngine.calculate_scalp_score error: {e}")

        score = sum(components.values())

        # Apply downtrend cap BEFORE gates: if ultra-short EMA is below short EMA,
        # cap score at 0.30 to block weak entries in active downtrends.
        # Full mean-reversion entries (all 3 signals near max) can still pass.
        if _downtrend:
            score = min(score, 0.30)

        # Gate 1 – Volume gate (regime-aware)
        # In choppy/sideways markets, allow mean_reversion+vwap entries without vol ignition.
        in_choppy_regime = (
            getattr(self.algo, 'market_regime', 'sideways') == 'sideways'
            or crypto.get('is_symbol_choppy', False)
        )
        vol_gate_required = not in_choppy_regime  # vol ignition mandatory only in trending markets

        if vol_gate_required and components['vol_ignition'] < 0.10:
            score = 0.0
        elif not vol_gate_required:
            # In choppy regime: require mean_reversion as primary gate instead of vol
            if components['mean_reversion'] < 0.10:
                score = 0.0
            else:
                # Still require MIN_SIGNAL_COUNT active signals (mean_rev + vwap = 2, passes)
                min_req = getattr(self.algo, 'min_signal_count', self.MIN_SIGNAL_COUNT)
                if sum(1 for v in components.values() if v >= 0.10) < min_req:
                    score = 0.0
        else:
            # Trending regime: Gate 2 – Multi-signal confirmation
            min_req = getattr(self.algo, 'min_signal_count', self.MIN_SIGNAL_COUNT)
            if sum(1 for v in components.values() if v >= 0.10) < min_req:
                score = 0.0

        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Conviction-scaled position sizing with vol-targeting.

        Base size scales linearly from 0.30 (at threshold) to 0.50 (at max score
        of 0.60), rewarding high-confluence entries without overriding the vol
        scalar that keeps per-trade risk constant across assets.
        """
        # Conviction scaling: higher score → larger base allocation.
        max_score = 0.60
        score_range = max_score - threshold
        if score_range > 0:
            conviction = max(0.0, min(1.0, (score - threshold) / score_range))
        else:
            conviction = 1.0
        size = 0.30 + 0.20 * conviction  # 0.30 at threshold → 0.50 at max score

        # Vol-targeting: scale down for volatile assets, keep size for calmer ones
        if asset_vol_ann is not None and asset_vol_ann > 0:
            target_vol = self.algo.target_position_ann_vol  # 0.50
            vol_scalar = min(target_vol / asset_vol_ann, 1.0)
            size *= max(vol_scalar, 0.8)  # Never reduce below 80% of base size

        return size
