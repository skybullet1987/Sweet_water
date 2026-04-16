import unittest
from datetime import datetime

from nextgen.core.types import RegimeState, SignalOutput
from nextgen.portfolio.allocator import AllocationConfig, SignalPortfolioAllocator


class PortfolioAllocatorTests(unittest.TestCase):
    def test_allocator_respects_position_cap(self) -> None:
        allocator = SignalPortfolioAllocator(AllocationConfig(max_position_weight=0.2, max_gross_target=1.0))
        now = datetime(2026, 1, 1)
        regime = RegimeState(now, 0.8, 0.2, 0.1, 1.0, 0.7, "trend", 4)
        signals = [
            SignalOutput("trend_breakout", "BTCUSD", now, 1.0, 1.0),
            SignalOutput("trend_breakout", "ETHUSD", now, -1.0, 1.0),
        ]
        targets = allocator.allocate(signals, regime)
        self.assertEqual(len(targets), 2)
        for target in targets:
            self.assertLessEqual(abs(target.target_weight), 0.2)

    def test_allocator_drops_low_confidence(self) -> None:
        allocator = SignalPortfolioAllocator(AllocationConfig(min_signal_confidence=0.3))
        now = datetime(2026, 1, 1)
        regime = RegimeState(now, 0.7, 0.3, 0.2, 0.8, 0.6, "trend", 2)
        signals = [SignalOutput("cross_sectional_momentum", "BTCUSD", now, 0.8, 0.1)]
        self.assertEqual(allocator.allocate(signals, regime), ())


if __name__ == "__main__":
    unittest.main()
