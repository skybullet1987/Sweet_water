"""
regime_router.py — Market Regime Router

Classifies market state and determines which trading engine is active.

Regimes
-------
- 'trend':      Clear directional market (bull or bear). The existing
                MicroScalpEngine (trend engine) is allowed to open new trades.
                Conditions: BTC is in 'bull' or 'bear' SMA-regime AND
                median ADX across the universe is at or above TREND_ADX_FLOOR.

- 'chop':       Range-bound / sideways market. The ChopEngine is allowed to
                open new trades.
                Conditions: BTC is in 'sideways' SMA-regime AND median ADX
                across the universe is at or below CHOP_ADX_CAP AND
                market breadth sits between BREADTH_CHOP_MIN and BREADTH_CHOP_MAX
                (mixed directional bias — not fully collapsed, not strongly
                trending).

- 'transition': Regime is uncertain or was recently changed. Both engines are
                suppressed; existing open positions continue to be managed by
                their respective engine exit logic. This state is entered when:
                - BTC regime is 'unknown' (insufficient data), or
                - BTC is trending but median ADX is too low to confirm, or
                - BTC is sideways but ADX or breadth are outside chop bounds, or
                - a forced cooldown is active immediately after a regime flip.

Hysteresis
----------
Prevents rapid ping-ponging. A regime change requires REGIME_MIN_BARS
consecutive bars where the new candidate is stable before the router commits
to the switch. After any flip, a TRANSITION_HOLD-bar mandatory transition
period is enforced to allow positions to settle.
"""

from collections import deque


class RegimeRouter:
    """
    Classifies current market state and routes entries to the correct engine.

    Usage (called from main.py each bar, after update_market_context):
        router.update()          # refresh classification
        engine = router.route()  # returns 'trend', 'chop', or 'transition'
        if router.engine_allowed('chop'):
            ...
    """

    # ADX thresholds for regime classification.
    # TREND_ADX_FLOOR: median ADX must be >= this to confirm trend conditions.
    TREND_ADX_FLOOR  = 18  # relaxed slightly vs 20 so minor-trend markets are caught
    # CHOP_ADX_CAP: median ADX must be <= this to confirm choppy conditions.
    CHOP_ADX_CAP     = 22  # overlaps with TREND_ADX_FLOOR to create a transition band

    # Market breadth bounds for chop classification (fraction of symbols in uptrend).
    # Chop regime requires breadth in [CHOP_MIN, CHOP_MAX] — genuinely mixed market.
    BREADTH_CHOP_MIN = 0.30
    BREADTH_CHOP_MAX = 0.70

    # Minimum number of consecutive bars a candidate regime must be stable before
    # committing to a flip (hysteresis guard).
    REGIME_MIN_BARS  = 5

    # Forced transition bars after any regime flip — allows trades to settle.
    TRANSITION_HOLD  = 3

    # Minimum number of universe symbols with a ready ADX before we trust the
    # median (otherwise we stay in transition due to insufficient data).
    MIN_ADX_SYMBOLS  = 5

    def __init__(self, algo):
        self.algo = algo

        # Current committed regime ('trend', 'chop', or 'transition').
        self.current_regime   = "transition"

        # Candidate regime being evaluated for hysteresis promotion.
        self._candidate       = "transition"

        # Number of consecutive bars the candidate has been stable.
        self._hold_count      = 0

        # Remaining bars of forced transition after a flip.
        self._transition_hold = 0

        # Short history for debugging / reporting (last ~48 bars).
        self._regime_history  = deque(maxlen=48)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self):
        """
        Refresh regime classification. Call once per bar after
        update_market_context() has been called.
        """
        new_candidate = self._classify()

        # Accumulate stability count for the current candidate.
        if new_candidate == self._candidate:
            self._hold_count += 1
        else:
            # Candidate changed — restart stability counter.
            self._candidate  = new_candidate
            self._hold_count = 1

        # Commit to new regime once candidate has been stable long enough.
        if (self._hold_count >= self.REGIME_MIN_BARS
                and new_candidate != self.current_regime):
            old = self.current_regime
            self.current_regime   = new_candidate
            self._transition_hold = self.TRANSITION_HOLD
            self.algo.Debug(
                f"REGIME ROUTER FLIP: {old} → {self.current_regime} "
                f"(btc={self.algo.market_regime} "
                f"breadth={self.algo.market_breadth:.0%} "
                f"hold={self._hold_count})"
            )

        # During the post-flip transition hold, override the committed regime
        # to 'transition' (suppress new entries while positions settle).
        if self._transition_hold > 0:
            self._transition_hold -= 1
            self._regime_history.append("transition")
            return

        self._regime_history.append(self.current_regime)

    def route(self) -> str:
        """
        Return the currently active routing regime:
        - 'trend'      — MicroScalpEngine may open new trades
        - 'chop'       — ChopEngine may open new trades
        - 'transition' — no new entries from either engine
        """
        if self._transition_hold > 0:
            return "transition"
        return self.current_regime

    def engine_allowed(self, engine_name: str) -> bool:
        """Return True if the named engine ('trend' or 'chop') may enter new trades."""
        return self.route() == engine_name

    # ------------------------------------------------------------------ #
    #  Internal classification logic                                       #
    # ------------------------------------------------------------------ #

    def _classify(self) -> str:
        """
        Compute the raw (un-hysteresis'd) regime from current market signals.
        Returns 'trend', 'chop', or 'transition'.
        """
        algo        = self.algo
        btc_regime  = getattr(algo, 'market_regime',   'unknown')
        breadth     = getattr(algo, 'market_breadth',   0.5)

        # Cannot classify without BTC regime data.
        if btc_regime == 'unknown':
            return "transition"

        # Compute median ADX across the universe.
        median_adx = self._median_universe_adx()

        # Insufficient universe data → stay in transition.
        if median_adx is None:
            return "transition"

        # ── Trend conditions ───────────────────────────────────────────────
        # BTC is clearly trending (bull or bear) and ADX confirms directionality.
        if btc_regime in ('bull', 'bear'):
            if median_adx >= self.TREND_ADX_FLOOR:
                return "trend"
            # BTC trending but ADX too low — conflicting signals → transition.
            return "transition"

        # ── Chop conditions ────────────────────────────────────────────────
        # BTC is sideways AND ADX is low AND breadth is genuinely mixed.
        if btc_regime == 'sideways':
            if (median_adx <= self.CHOP_ADX_CAP
                    and self.BREADTH_CHOP_MIN <= breadth <= self.BREADTH_CHOP_MAX):
                return "chop"
            # Sideways but ADX elevated or breadth extreme → uncertain.
            return "transition"

        # Catch-all.
        return "transition"

    def _median_universe_adx(self):
        """
        Compute the median ADX value across all ready universe symbols.
        Returns None when fewer than MIN_ADX_SYMBOLS symbols have a ready ADX.
        Runs in O(n) without external libraries.
        """
        algo = self.algo
        adx_values = []
        for crypto in algo.crypto_data.values():
            adx_ind = crypto.get('adx')
            if adx_ind is not None and adx_ind.IsReady:
                adx_values.append(adx_ind.Current.Value)

        if len(adx_values) < self.MIN_ADX_SYMBOLS:
            return None

        adx_values.sort()
        n   = len(adx_values)
        mid = n // 2
        if n % 2 == 0:
            return (adx_values[mid - 1] + adx_values[mid]) / 2.0
        return adx_values[mid]
