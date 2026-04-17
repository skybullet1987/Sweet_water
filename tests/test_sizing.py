import unittest

from sizing.kelly import fractional_kelly
from sizing.vol_target import VolTargetScaler


class SizingTests(unittest.TestCase):
    def test_fractional_kelly_closed_form(self):
        out = fractional_kelly(win_prob=0.6, win_loss_ratio=2.0, fraction=0.25, cap=0.20)
        self.assertAlmostEqual(out, 0.1, places=8)

    def test_kelly_clamps_to_zero(self):
        out = fractional_kelly(win_prob=0.3, win_loss_ratio=1.0)
        self.assertEqual(out, 0.0)

    def test_vol_target_scale_bounds(self):
        vt = VolTargetScaler()
        for _ in range(60):
            vt.update_returns(0.001)
        scale = vt.scale()
        self.assertGreaterEqual(scale, 0.5)
        self.assertLessEqual(scale, 2.0)


if __name__ == "__main__":
    unittest.main()
