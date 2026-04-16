import unittest
from datetime import datetime, timedelta

from nextgen.core.types import FeatureOutput
from nextgen.regime.engine import ProbabilisticRegimeEngine, RegimeEngineConfig


class RegimeEngineTests(unittest.TestCase):
    def test_regime_hysteresis_requires_persistence_before_switch(self) -> None:
        engine = ProbabilisticRegimeEngine(RegimeEngineConfig(smoothing=1.0, switch_threshold=0.05, min_persistence_steps=2))
        t0 = datetime(2026, 1, 1)

        trend = FeatureOutput("BTCUSD", t0, {"trend_strength": 0.8, "mean_reversion_score": 0.0, "realized_vol": 0.2, "liquidity": 0.8, "breadth": 0.7})
        first = engine.update([trend])
        self.assertEqual(first.active_regime, "trend")

        mean_rev = FeatureOutput("BTCUSD", t0 + timedelta(minutes=5), {"trend_strength": -0.8, "mean_reversion_score": 0.9, "realized_vol": 0.2, "liquidity": 0.8, "breadth": -0.7})
        second = engine.update([mean_rev])
        self.assertEqual(second.active_regime, "trend")

        third = engine.update([mean_rev])
        self.assertEqual(third.active_regime, "mean_reversion")


if __name__ == "__main__":
    unittest.main()
