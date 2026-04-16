from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from nextgen.core.types import FeatureOutput, RegimeState, SignalOutput
from .utils import clamp_signal


class CrossSectionalMomentumSleeve:
    """
    Cross-sectional momentum sleeve.

    Uses ``rank_universe()`` to compute z-scores across all symbols in the
    investment universe before each bar's ``generate()`` calls.  Each symbol's
    score is its vol-normalised momentum, ranked relative to peers.

    Workflow per bar
    ----------------
    1. ``sleeve.rank_universe(all_features)``  – O(N), build universe ranks
    2. ``sleeve.generate(feature, regime)``    – O(1), look up stored rank
    """

    name = "cross_sectional_momentum"

    def __init__(self) -> None:
        self._ranks: Dict[str, float] = {}  # symbol → z-scored rank in [~−3, ~+3]

    # ── universe ranking (call once per bar with full universe) ────────────

    def rank_universe(self, features: Sequence[FeatureOutput]) -> None:
        """
        Compute vol-normalised, cross-sectionally z-scored momentum ranks.

        Steps
        -----
        1. Vol-normalise each symbol's momentum:
               vol_norm_mom = momentum / max(realized_vol, floor)
        2. Z-score across the universe:
               z = (vol_norm_mom − mean) / max(std, floor)
        3. Store result in ``self._ranks`` keyed by symbol.

        A symbol with no ``realized_vol`` in its features falls back to raw
        momentum so the calculation is never blocked by missing data.
        """
        if not features:
            self._ranks = {}
            return

        vol_norm: List[tuple[str, float]] = []
        for f in features:
            mom = f.values.get("momentum_short", f.values.get("momentum", 0.0))
            rv = f.values.get("realized_vol_short", f.values.get("realized_vol", 0.0))
            # Vol-floor prevents extreme scores when vol is near zero at market open.
            vol_floor = 0.01  # 1% annualised vol minimum
            vol_norm_mom = mom / max(rv, vol_floor)
            vol_norm.append((f.symbol, vol_norm_mom))

        if not vol_norm:
            self._ranks = {}
            return

        scores = [v for _, v in vol_norm]
        mean = sum(scores) / len(scores)
        if len(scores) >= 2:
            var = sum((x - mean) ** 2 for x in scores) / len(scores)
            std = math.sqrt(max(var, 0.0))
        else:
            std = 0.0

        std_floor = 1e-6
        self._ranks = {
            sym: (score - mean) / max(std, std_floor)
            for sym, score in vol_norm
        }

    # ── per-symbol signal generation ──────────────────────────────────────

    def generate(self, feature: FeatureOutput, regime: RegimeState) -> SignalOutput:
        """
        Return a SignalOutput using the pre-computed cross-sectional rank.

        If ``rank_universe()`` has not been called yet (or called with an empty
        list), falls back to the raw momentum value so the sleeve is always safe
        to call in isolation.
        """
        rank_score = self._ranks.get(
            feature.symbol,
            feature.values.get("momentum_rank", feature.values.get("momentum", 0.0)),
        )
        # Normalise the z-score to [−1, +1] by dividing by a reasonable spread (2σ)
        normalised = clamp_signal(rank_score / 2.0)
        score = clamp_signal(normalised * regime.trend_confidence)
        confidence = max(0.0, min(1.0, abs(score)))
        return SignalOutput(
            self.name,
            feature.symbol,
            feature.timestamp,
            score,
            confidence,
            {"raw_rank": float(rank_score), "normalised_rank": float(normalised)},
        )
