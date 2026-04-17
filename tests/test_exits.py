import unittest

from exits.triple_barrier import build_barrier, check_barrier_hit


class TripleBarrierTests(unittest.TestCase):
    def test_take_profit_hit_first(self):
        barrier = build_barrier(entry_price=100, atr=5, side="long", entry_bar=0)
        hit = check_barrier_hit(barrier, bar_index=1, high=111, low=99, close=110)
        self.assertEqual(hit, "take_profit")

    def test_stop_loss_hit_first(self):
        barrier = build_barrier(entry_price=100, atr=5, side="long", entry_bar=0)
        hit = check_barrier_hit(barrier, bar_index=1, high=102, low=94, close=95)
        self.assertEqual(hit, "stop_loss")

    def test_time_stop_hit(self):
        barrier = build_barrier(entry_price=100, atr=5, side="long", entry_bar=0)
        hit = check_barrier_hit(barrier, bar_index=24, high=101, low=99, close=100)
        self.assertEqual(hit, "time_stop")


if __name__ == "__main__":
    unittest.main()
