"""
trade_quality.py — Trade Quality Architecture

Provides modular, feature-flagged quality enhancements for the micro-scalping
strategy. All features are opt-in via algo attributes so baseline behaviour is
fully preserved when they are disabled.

Modules
-------
1. Session / liquidity regime layer
   – Classifies UTC hour into named session buckets.
   – Returns conservative (threshold_adj, size_mult, spread_cap_mult) per session.

2. Adverse-selection filter
   – Rejects entries showing signs of late / overextended / thin-market entry:
     (A) spread widening vs recent median,
     (B) bar displacement too stretched vs ATR,
     (C) price too extended above VWAP,
     (D) breakout after excessive 1-bar move.

3. Bollinger Band compression context
   – Detects squeeze state from rolling bb_width.
   – Returns compression rank [0, 1] for use in ranking overlay.

4. RS-vs-BTC bounded rank modifier
   – Converts recent rs_vs_btc into a bounded ±cap rank adjustment.

5. MFE / MAE tracking
   – Per-trade maximum-favorable / maximum-adverse excursion.
   – Updated on every CheckExits bar; finalised and persisted on exit.

6. Setup archetype classification
   – Classifies each entry into a small, human-readable archetype label.

7. Rich per-trade metadata recording
   – Records session, spread, dollar-volume, ADX, RS, archetype buckets at entry.
   – Merges MFE/MAE at exit and routes PnL into per-dimension attribution dicts.
"""

# region imports
import math
import itertools
import numpy as np
# endregion


# ---------------------------------------------------------------------------
# 1.  Session / liquidity regime layer
# ---------------------------------------------------------------------------

# Named session buckets with their UTC hour ranges.
SESSION_BUCKETS = [
    ('asia_dead',  list(range(0,  5))),   # 00-04 UTC: very thin; widest spreads
    ('asia',       list(range(5,  8))),   # 05-07 UTC: low-medium volume
    ('eu_open',    list(range(8,  10))),  # 08-09 UTC: EU participants arriving
    ('eu_main',    list(range(10, 13))),  # 10-12 UTC: solid EU liquidity
    ('us_open',    list(range(13, 16))),  # 13-15 UTC: US-EU overlap, tightest spreads
    ('us_main',    list(range(16, 21))),  # 16-20 UTC: high-quality US session
    ('us_eve',     list(range(21, 24))),  # 21-23 UTC: tailing off
]

# Pre-build hour → bucket-name lookup (O(1) per call).
_SESSION_HOUR_MAP = {}
for _name, _hours in SESSION_BUCKETS:
    for _h in _hours:
        _SESSION_HOUR_MAP[_h] = _name


def get_session_bucket(hour_utc):
    """Return the session bucket name for a UTC hour (0–23)."""
    return _SESSION_HOUR_MAP.get(int(hour_utc) % 24, 'us_eve')


# Default session quality table: (threshold_adj, size_mult, spread_cap_mult).
# Conservative defaults: only asia_dead applies a meaningful penalty.
# All values are user-overridable via algo._session_thresh_*, etc.
_SESSION_QUALITY_DEFAULTS = {
    'asia_dead': ( 0.05,  0.70,  0.70),
    'asia':      ( 0.02,  0.85,  0.85),
    'eu_open':   ( 0.00,  1.00,  1.00),
    'eu_main':   ( 0.00,  1.00,  1.00),
    'us_open':   (-0.02,  1.05,  1.10),
    'us_main':   ( 0.00,  1.00,  1.00),
    'us_eve':    ( 0.02,  0.90,  0.90),
}


def get_session_quality(algo, hour_utc):
    """
    Return (threshold_adj, size_mult, spread_cap_mult) for the current UTC hour.

    All three values are user-overridable via algo attributes:
        algo._session_thresh_<bucket>, algo._session_size_<bucket>,
        algo._session_spread_<bucket>

    Set algo.session_layer_enabled = False to return neutral (0, 1, 1).
    """
    if not getattr(algo, 'session_layer_enabled', True):
        return 0.0, 1.0, 1.0

    bucket = get_session_bucket(hour_utc)
    defaults = _SESSION_QUALITY_DEFAULTS.get(bucket, (0.0, 1.0, 1.0))

    thresh_adj  = getattr(algo, f'_session_thresh_{bucket}',  defaults[0])
    size_mult   = getattr(algo, f'_session_size_{bucket}',    defaults[1])
    spread_mult = getattr(algo, f'_session_spread_{bucket}',  defaults[2])
    return float(thresh_adj), float(size_mult), float(spread_mult)


# ---------------------------------------------------------------------------
# 2.  Adverse-selection filter
# ---------------------------------------------------------------------------

def adverse_selection_filter(algo, symbol, crypto, components, current_price):
    """
    Reject entries that show signs of adverse selection / late / overextended entry.

    Returns (passes: bool, reason: str).  On error always returns (True, '').

    Checks (all configurable via algo attributes):
      A) Spread widening: current spread > median spread × asel_spread_widen_thresh
      C) VWAP extension: price above VWAP by more than asel_vwap_extension_sd_mult × SD

    Checks B (bar displacement vs ATR) and D (breakout after excessive displacement)
    have been removed — they are structurally antagonistic with vol-ignition, which by
    definition fires on high-displacement bars.

    Set algo.adverse_selection_enabled = False to disable entirely.
    """
    if not getattr(algo, 'adverse_selection_enabled', True):
        return True, ''

    try:
        from execution import get_spread_pct, _estimate_backtest_spread

        # A) Spread widening vs recent median
        spreads = crypto.get('spreads', [])
        if len(spreads) >= 4:
            current_sp = get_spread_pct(algo, symbol)
            if current_sp is None and not algo.LiveMode:
                current_sp = _estimate_backtest_spread(algo, symbol)
            if current_sp is not None:
                median_sp = float(np.median(list(spreads)[-12:]))
                widen_thresh = getattr(algo, 'asel_spread_widen_thresh', 2.5)
                if median_sp > 0 and current_sp > median_sp * widen_thresh:
                    return False, (f'spread_widening({current_sp:.3%}>'
                                   f'{median_sp:.3%}x{widen_thresh})')

        # C) VWAP extension: price too far above VWAP SD bands
        vwap   = crypto.get('vwap',   0.0)
        vwap_sd = crypto.get('vwap_sd', 0.0)
        if vwap > 0 and vwap_sd > 0 and current_price > 0:
            extension   = (current_price - vwap) / vwap
            sd_mult     = getattr(algo, 'asel_vwap_extension_sd_mult', 3.0)
            max_ext_pct = sd_mult * (vwap_sd / vwap)
            if extension > max_ext_pct:
                return False, (f'vwap_extension({extension:.3%}>'
                               f'{max_ext_pct:.3%})')

    except Exception:
        pass  # Never block an entry on error

    return True, ''


# ---------------------------------------------------------------------------
# 3.  Bollinger Band compression context
# ---------------------------------------------------------------------------

def get_bb_compression_state(crypto, compression_pct=None):
    """
    Return True if the current BB width is in the compressed portion of its
    rolling history (below the compression_pct-ile of rolling width).

    compression_pct: 0–1 fraction. Default 0.75 (bottom 75 %ile = compressed).
    """
    if compression_pct is None:
        compression_pct = 0.75
    bb_width = crypto.get('bb_width', [])
    if len(bb_width) < 10:
        return False
    width_list = list(bb_width)
    current_width = width_list[-1]
    if current_width <= 0:
        return False
    threshold = float(np.percentile(width_list, compression_pct * 100.0))
    return current_width <= threshold


def get_bb_compression_rank(crypto):
    """
    Return a 0–1 rank where 1 = maximally compressed and 0 = maximally wide.
    Used as a quality modifier in the ranking overlay.
    Returns 0.5 (neutral) when bb_width history is insufficient.
    """
    bb_width = crypto.get('bb_width', [])
    if len(bb_width) < 10:
        return 0.5
    width_list = list(bb_width)
    current_width = width_list[-1]
    if current_width <= 0:
        return 0.5
    min_w = float(np.min(width_list))
    max_w = float(np.max(width_list))
    if max_w - min_w <= 0:
        return 0.5
    return (max_w - current_width) / (max_w - min_w)


# ---------------------------------------------------------------------------
# 4.  RS-vs-BTC bounded rank modifier
# ---------------------------------------------------------------------------

def compute_rs_rank_modifier(algo, crypto):
    """
    Return a bounded RS-vs-BTC rank modifier in [-cap, +cap].

    Uses the average of the last 6 rs_vs_btc values scaled by rs_rank_scale,
    then clamped to ±rs_rank_cap.  Result integrates into compute_ranking_overlay
    without dominating the core score.

    Set algo.rs_rank_overlay_enabled = False to return 0.
    """
    if not getattr(algo, 'rs_rank_overlay_enabled', True):
        return 0.0

    rs_deque = crypto.get('rs_vs_btc', [])
    if len(rs_deque) < 3:
        return 0.0

    n = min(6, len(rs_deque))
    recent_rs = sum(itertools.islice(reversed(rs_deque), n)) / n

    scale = getattr(algo, 'rs_rank_scale', 3.0)   # e.g. 1% RS → ~0.03 adj
    cap   = getattr(algo, 'rs_rank_cap',   0.05)   # hard clamp ±5%
    raw   = recent_rs * scale
    return float(max(-cap, min(cap, raw)))


# ---------------------------------------------------------------------------
# 5.  MFE / MAE tracking
# ---------------------------------------------------------------------------

def init_trade_excursion(algo, symbol, entry_price):
    """
    Initialise per-trade excursion tracking on entry fill.
    Called from events.on_order_event when a Buy is Filled.
    """
    if not hasattr(algo, '_trade_excursion'):
        algo._trade_excursion = {}
    algo._trade_excursion[symbol] = {
        'entry_price': float(entry_price),
        'mfe': 0.0,
        'mae': 0.0,
        'bars_held': 0,
        'bars_to_mfe': 0,
        '_peak_pnl': 0.0,
    }


def update_trade_excursion(algo, symbol, current_price):
    """
    Update MFE/MAE for an open position with the latest price.
    Called from main._check_exit() on every CheckExits iteration.
    """
    if not hasattr(algo, '_trade_excursion'):
        return
    rec = algo._trade_excursion.get(symbol)
    if rec is None:
        return
    entry = rec['entry_price']
    if entry <= 0 or current_price <= 0:
        return
    pnl = (float(current_price) - entry) / entry
    rec['bars_held'] += 1
    if pnl > rec['_peak_pnl']:
        rec['_peak_pnl'] = pnl
        rec['bars_to_mfe'] = rec['bars_held']
    if pnl > rec['mfe']:
        rec['mfe'] = pnl
    if pnl < rec['mae']:
        rec['mae'] = pnl


def finalize_trade_excursion(algo, symbol):
    """
    Pop and return (mfe, mae, bars_to_mfe, bars_held) for a closed trade.
    Returns (0, 0, 0, 0) when no record exists.
    """
    if not hasattr(algo, '_trade_excursion'):
        return 0.0, 0.0, 0, 0
    rec = algo._trade_excursion.pop(symbol, None)
    if rec is None:
        return 0.0, 0.0, 0, 0
    return rec['mfe'], rec['mae'], rec['bars_to_mfe'], rec['bars_held']


# ---------------------------------------------------------------------------
# 6.  Setup archetype classification
# ---------------------------------------------------------------------------

def get_setup_archetype(components, bb_compressed, crypto):
    """
    Classify an entry into a short interpretable archetype label.

    Archetypes
    ----------
    full_confluence     – all 3 signals active
    compression_breakout – vol ignition after BB squeeze, no mean-reversion
    vwap_breakout       – vol ignition + VWAP, no mean-reversion
    vol_mean_rev        – vol ignition + mean-reversion, no VWAP
    mean_reversion      – mean-reversion dominant, no vol ignition
    momentum_only       – vol ignition alone (gated out by default)
    mixed               – 2+ signals, none of the above patterns
    """
    vol  = components.get('vol_ignition',   0) >= 0.10
    rev  = components.get('mean_reversion', 0) >= 0.10
    vwap = components.get('vwap_signal',    0) >= 0.10

    if vol and rev and vwap:
        return 'full_confluence'
    if vol and bb_compressed and not rev:
        return 'compression_breakout'
    if vol and vwap and not rev:
        return 'vwap_breakout'
    if vol and rev and not vwap:
        return 'vol_mean_rev'
    if rev and not vol:
        return 'mean_reversion'
    if vol and not rev and not vwap:
        return 'momentum_only'
    return 'mixed'


# ---------------------------------------------------------------------------
# 7.  Rich per-trade metadata helpers
# ---------------------------------------------------------------------------

def get_adx_bucket(crypto):
    """Return a human-readable ADX regime bucket label."""
    adx_ind = crypto.get('adx')
    if adx_ind is None or not adx_ind.IsReady:
        return 'unknown'
    v = adx_ind.Current.Value
    if v < 10:  return 'flat'
    if v < 20:  return 'mild'
    if v < 30:  return 'moderate'
    return 'strong'


def get_spread_bucket(spread_pct):
    """Return a spread bucket label for reporting."""
    if spread_pct is None:    return 'unknown'
    if spread_pct < 0.001:    return '<0.1%'
    if spread_pct < 0.003:    return '0.1-0.3%'
    if spread_pct < 0.005:    return '0.3-0.5%'
    if spread_pct < 0.010:    return '0.5-1.0%'
    return '>1.0%'


def get_dollar_volume_bucket(recent_dv):
    """Return a dollar-volume bucket label for reporting."""
    if recent_dv is None:         return 'unknown'
    if recent_dv < 5_000:         return '<$5k'
    if recent_dv < 25_000:        return '$5-25k'
    if recent_dv < 100_000:       return '$25-100k'
    if recent_dv < 500_000:       return '$100-500k'
    return '>$500k'


def get_rs_bucket(rs_val):
    """Return RS-vs-BTC bucket label for reporting."""
    if rs_val is None:    return 'unknown'
    if rs_val < -0.005:   return 'strong_neg'
    if rs_val < -0.001:   return 'mild_neg'
    if rs_val <  0.001:   return 'neutral'
    if rs_val <  0.005:   return 'mild_pos'
    return 'strong_pos'


def record_trade_metadata_on_entry(algo, symbol, components, crypto,
                                    spread_pct, recent_dv):
    """
    Record rich per-trade metadata dict at entry time.
    Stored in algo._trade_metadata[symbol].
    Requires algo._entry_signal_combos[symbol] to already be set.
    """
    if not hasattr(algo, '_trade_metadata'):
        algo._trade_metadata = {}

    rs_deque = crypto.get('rs_vs_btc', [])
    rs_last  = float(rs_deque[-1]) if len(rs_deque) >= 1 else None
    bb_comp  = get_bb_compression_state(crypto,
                   getattr(algo, 'bb_compression_pct', 0.75))

    algo._trade_metadata[symbol] = {
        'signal_combo':  algo._entry_signal_combos.get(symbol, 'unknown'),
        'session_bucket': get_session_bucket(algo.Time.hour),
        'spread_bucket':  get_spread_bucket(spread_pct),
        'dv_bucket':      get_dollar_volume_bucket(recent_dv),
        'adx_bucket':     get_adx_bucket(crypto),
        'rs_bucket':      get_rs_bucket(rs_last),
        'setup_archetype': get_setup_archetype(components, bb_comp, crypto),
        'bb_compressed':  bb_comp,
        'regime':         getattr(algo, 'market_regime',    'unknown'),
        'vol_regime':     getattr(algo, 'volatility_regime', 'normal'),
    }


def finalize_trade_metadata_on_exit(algo, symbol, pnl):
    """
    Merge MFE/MAE into trade metadata and distribute PnL into per-dimension
    attribution dicts (pnl_by_session, pnl_by_archetype, etc.).

    Returns the completed metadata dict (or empty dict if none recorded).
    Called from events.on_order_event on a completed Sell fill.
    """
    if not hasattr(algo, '_trade_metadata'):
        return {}
    meta = algo._trade_metadata.pop(symbol, None)
    if meta is None:
        return {}

    mfe, mae, bars_to_mfe, bars_held = finalize_trade_excursion(algo, symbol)
    meta.update({'mfe': mfe, 'mae': mae,
                 'bars_to_mfe': bars_to_mfe, 'bars_held': bars_held,
                 'pnl': pnl})

    def _record(attr, key, val, cap=200):
        if not hasattr(algo, attr):
            setattr(algo, attr, {})
        d = getattr(algo, attr)
        if key not in d:
            d[key] = []
        d[key].append(val)
        if len(d[key]) > cap:
            d[key] = d[key][-cap:]

    _record('pnl_by_session',      meta['session_bucket'],  pnl)
    _record('pnl_by_archetype',    meta['setup_archetype'], pnl)
    _record('pnl_by_adx_bucket',   meta['adx_bucket'],      pnl)
    _record('pnl_by_spread_bucket', meta['spread_bucket'],  pnl)
    _record('pnl_by_dv_bucket',    meta['dv_bucket'],       pnl)
    _record('pnl_by_rs_bucket',    meta['rs_bucket'],       pnl)

    # MFE/MAE per archetype
    arch = meta['setup_archetype']
    _record('mfe_by_archetype', arch, mfe)
    _record('mae_by_archetype', arch, mae)

    return meta
