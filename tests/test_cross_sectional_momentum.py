"""Tests for the cross-sectional z-score ranking in CrossSectionalMomentumSleeve (item 3)."""
from __future__ import annotations

import unittest
from datetime import datetime

from nextgen.core.models import FeatureOutput, RegimeState
from nextgen.signals.cross_sectional_momentum import CrossSectionalMomentumSleeve


def _regime(trend: float = 0.7) -> RegimeState:
    return RegimeState(datetime(2026, 1, 1), trend, 0.3, 0.1, 1.0, 0.7, "trend", 4)


def _feat(symbol: str, momentum: float, rv: float = 0.5) -> FeatureOutput:
    return FeatureOutput(
        symbol,
        datetime(2026, 1, 1),
        {
            "momentum_short": momentum,
            "momentum": momentum,
            "realized_vol_short": rv,
            "realized_vol": rv,
        },
    )


class CrossSectionalMomentumTest(unittest.TestCase):

    def setUp(self) -> None:
        self.sleeve = CrossSectionalMomentumSleeve()

    def test_rank_universe_stores_z_scores(self) -> None:
        features = [
            _feat("BTC", 0.10),
            _feat("ETH", 0.05),
            _feat("SOL", -0.05),
        ]
        self.sleeve.rank_universe(features)
        # BTC has highest momentum → positive rank
        self.assertGreater(self.sleeve._ranks["BTC"], 0.0)
        # SOL has lowest momentum → negative rank
        self.assertLess(self.sleeve._ranks["SOL"], 0.0)

    def test_z_scores_approximately_zero_mean(self) -> None:
        features = [_feat(f"SYM{i}", float(i) * 0.01) for i in range(10)]
        self.sleeve.rank_universe(features)
        mean_rank = sum(self.sleeve._ranks.values()) / len(self.sleeve._ranks)
        self.assertAlmostEqual(mean_rank, 0.0, places=10)

    def test_generate_uses_stored_rank(self) -> None:
        features = [
            _feat("BTC", 0.20),
            _feat("ETH", 0.00),
            _feat("SOL", -0.20),
        ]
        self.sleeve.rank_universe(features)
        regime = _regime(trend=1.0)  # full trend confidence

        btc_sig = self.sleeve.generate(features[0], regime)
        sol_sig = self.sleeve.generate(features[2], regime)

        self.assertGreater(btc_sig.score, 0.0, "BTC should have positive score")
        self.assertLess(sol_sig.score, 0.0, "SOL should have negative score")
        self.assertGreater(btc_sig.score, sol_sig.score)

    def test_rank_by_vol_normalised_momentum(self) -> None:
        """A high-momentum but high-vol symbol should rank lower than a calmer one."""
        # SYM_A: high momentum, very high vol → vol-normalised momentum = 0.5/2.0 = 0.25
        # SYM_B: lower momentum, low vol → vol-normalised momentum = 0.3/0.1 = 3.0
        features = [
            _feat("SYM_A", 0.5, rv=2.0),
            _feat("SYM_B", 0.3, rv=0.1),
        ]
        self.sleeve.rank_universe(features)
        self.assertGreater(self.sleeve._ranks["SYM_B"], self.sleeve._ranks["SYM_A"])

    def test_empty_universe_clears_ranks(self) -> None:
        self.sleeve.rank_universe([_feat("BTC", 0.1)])
        self.sleeve.rank_universe([])
        self.assertEqual(self.sleeve._ranks, {})

    def test_fallback_when_no_rank_computed(self) -> None:
        """generate() before rank_universe() falls back to raw momentum safely."""
        feat = _feat("BTC", 0.5)
        regime = _regime()
        sig = self.sleeve.generate(feat, regime)
        # Should not raise; score should be in [-1, 1]
        self.assertGreaterEqual(sig.score, -1.0)
        self.assertLessEqual(sig.score, 1.0)

    def test_confidence_non_negative(self) -> None:
        features = [_feat("BTC", 0.3), _feat("ETH", -0.1)]
        self.sleeve.rank_universe(features)
        for feat in features:
            sig = self.sleeve.generate(feat, _regime())
            self.assertGreaterEqual(sig.confidence, 0.0)

    def test_score_scaled_by_regime_trend_confidence(self) -> None:
        """Low trend_confidence regime should attenuate the score."""
        features = [_feat("BTC", 0.4), _feat("ETH", 0.0)]
        self.sleeve.rank_universe(features)

        high_regime = _regime(trend=1.0)
        low_regime = _regime(trend=0.1)
        sig_high = self.sleeve.generate(features[0], high_regime)
        sig_low = self.sleeve.generate(features[0], low_regime)
        self.assertGreater(abs(sig_high.score), abs(sig_low.score))


if __name__ == "__main__":
    unittest.main()
