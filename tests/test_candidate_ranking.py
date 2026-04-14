import unittest
from collections import deque

from candidate_ranking import rank_candidates


class _Sym:
    def __init__(self, value):
        self.Value = value


class _Router:
    def __init__(self, conf):
        self._conf = conf

    def get_confidence(self):
        return self._conf


class _Algo:
    def __init__(self):
        self.max_spread_pct = 0.005
        self.symbol_penalty_threshold = 3
        self.Time = 100
        self._regime_router = _Router(0.85)
        self._symbol_entry_cooldowns = {}
        self._symbol_performance = {
            'GOOD': deque([0.01, 0.015, 0.012], maxlen=50),
            'BAD': deque([-0.02, -0.02, -0.01], maxlen=50),
        }

    def _max_open_position_correlation(self, symbol):
        return 0.92 if symbol.Value == 'BAD' else 0.20


class CandidateRankingTests(unittest.TestCase):
    def test_ranking_prioritizes_higher_quality_candidate(self):
        algo = _Algo()
        ranked = rank_candidates(
            algo,
            [
                {
                    'symbol': _Sym('GOOD'),
                    'net_score': 0.42,
                    'spread_pct': 0.0012,
                    'expected_move_pct': 0.030,
                    'expected_cost_pct': 0.009,
                },
                {
                    'symbol': _Sym('BAD'),
                    'net_score': 0.44,  # higher base score but worse quality
                    'spread_pct': 0.0048,
                    'expected_move_pct': 0.012,
                    'expected_cost_pct': 0.010,
                },
            ],
            base_score_key='net_score',
            score_scale=1.0,
        )
        self.assertEqual(ranked[0]['symbol'].Value, 'GOOD')
        self.assertIn('rank_components', ranked[0])
        self.assertIn('edge_vs_cost', ranked[0]['rank_components'])


if __name__ == '__main__':
    unittest.main()
