# region imports
from AlgorithmImports import *
import numpy as np
# endregion


class MicroScalpEngine:
    """
    Micro-Scalping Signal Engine - v8.0.0

    Simplified 3-signal scoring system with ADX regime filter.
    Strips kitchen-sink approach down to the 3 signals with proven edge.

    Score: 0.0 – 0.60 max (3 signals × 0.20 each).
      Gate: vol_ignition >= 0.10 required for any entry.
      ADX filter: ADX > 25 AND DI- > DI+ × 1.2 → reject (strong bearish trend).

    Signals
    -------
    1. Volume Ignition (gate): 4× volume surge; 1.8× in choppy markets (ADX < 25)
    2. Mean Reversion: RSI oversold + price near lower BB when ADX is low (max 0.20)
    3. VWAP Reclaim / SD Band Bounce: price above VWAP or bouncing off -2/-3 SD (max 0.20)

    Removed: OBI, MTF trend, steady grind, ADX trend, CVD divergence, Kalman reversion, BB squeeze
    """

    # Tunable signal thresholds (easy to adjust for backtesting)
    VOL_SURGE_STRONG        = 4.0    # 4× average volume = strong ignition
    VOL_SURGE_PARTIAL       = 2.5    # 2.5× volume = moderate spike
    ADX_STRONG_THRESHOLD    = 14     # (kept for mean reversion gating)
    ADX_MODERATE_THRESHOLD  = 10     # moderate directional threshold
    VWAP_BUFFER             = 1.0005  # 0.05% above VWAP for confirmed reclaim
    # Ranging-market mean reversion thresholds (used when ADX < ADX_MODERATE_THRESHOLD)
    RSI_OVERSOLD_THRESHOLD        = 45   # RSI < 45 → ranging-market entry (mean reversion in sideways/choppy markets)
    RSI_MILDLY_OVERSOLD_THRESHOLD = 50   # RSI < 50 → mild ranging-market entry, partial credit
    BB_NEAR_LOWER_PCT             = 0.03  # within 3% of lower Bollinger Band = near support

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

        Gate: vol_ignition >= 0.10 required for any entry.
        ADX filter: ADX > 25 AND DI- > DI+ × 1.2 → return (0.0, components).
        Max possible score: 0.20 (vol) + 0.20 (mean_rev) + 0.20 (vwap) = 0.60.
        """
        components = {
            'vol_ignition':   0.0,
            'mean_reversion': 0.0,
            'vwap_signal':    0.0,
        }

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
                volumes = list(crypto['volume'])
                current_vol = volumes[-1]
                # Adaptive baseline: prefer long-term rolling average (up to 24h)
                vol_long = list(crypto.get('volume_long', []))
                if len(vol_long) >= 60:
                    vol_baseline = float(np.mean(vol_long))
                else:
                    vol_baseline = float(np.mean(volumes[-20:]))
                # ADX regime filter: lower thresholds in choppy markets (ADX < 25)
                adx_indicator = crypto.get('adx')
                is_choppy = (adx_indicator is not None and adx_indicator.IsReady
                             and adx_indicator.Current.Value < 25)
                vol_strong  = 1.8 if is_choppy else self.VOL_SURGE_STRONG
                vol_partial = 1.0 if is_choppy else self.VOL_SURGE_PARTIAL
                if vol_baseline > 0:
                    ratio = current_vol / vol_baseline
                    if ratio >= vol_strong:
                        components['vol_ignition'] = 0.20
                    elif ratio >= vol_partial:
                        # Partial credit for a meaningful volume spike
                        components['vol_ignition'] = 0.10

            # ----------------------------------------------------------
            # ADX FILTER: reject strong bearish trends
            # If ADX > 25 AND DI- > DI+ × 1.2 → mean reversion will fail, skip.
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
                    # Strong bearish trend — mean reversion will fail, skip
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
                        if (self.algo.market_regime == 'sideways'
                                and bb_lower > 0 and price <= bb_lower * 1.005 and rsi_val < 35):
                            components['mean_reversion'] = 0.20
                        elif (adx_val <= self.ADX_MODERATE_THRESHOLD
                                and rsi_val < self.RSI_OVERSOLD_THRESHOLD
                                and bb_lower > 0
                                and price <= bb_lower * (1 + self.BB_NEAR_LOWER_PCT)):
                            components['mean_reversion'] = 0.20
                        elif is_mild_oversold_ranging:
                            components['mean_reversion'] = 0.10

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
            if vwap > 0 and len(crypto['prices']) >= 1:
                price = crypto['prices'][-1]
                if price > vwap * self.VWAP_BUFFER:
                    # Price clearly above VWAP (0.1% buffer)
                    components['vwap_signal'] = 0.20
                elif price > vwap:
                    # Price marginally above VWAP
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

        # Volume gate: require at least partial volume ignition for any entry.
        # This replaces the old separate backtest vs live gate logic.
        if components['vol_ignition'] < 0.10:
            score = 0.0

        return min(score, 1.0), components

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def calculate_position_size(self, score, threshold, asset_vol_ann):
        """
        Simplified position sizing: fixed fraction with vol-targeting.
        Kelly removed — noisy at this sample size and signal quality.
        """
        # Fixed base size — same for all entries that pass threshold
        size = 0.50

        # Vol-targeting: scale down for volatile assets, keep size for calmer ones
        if asset_vol_ann is not None and asset_vol_ann > 0:
            target_vol = self.algo.target_position_ann_vol  # 0.35
            vol_scalar = min(target_vol / asset_vol_ann, 1.0)
            size *= max(vol_scalar, 0.6)  # Never reduce below 60% of base size

        return size
