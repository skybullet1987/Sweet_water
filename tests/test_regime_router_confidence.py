import unittest

from regime_router import RegimeRouter


class _Current:
    def __init__(self, value):
        self.Value = value


class _Adx:
    def __init__(self, value):
        self.IsReady = True
        self.Current = _Current(value)


class _Algo:
    def __init__(self, regime, breadth, adx_values):
        self.market_regime = regime
        self.market_breadth = breadth
        self.crypto_data = {f's{i}': {'adx': _Adx(v)} for i, v in enumerate(adx_values)}
        self.regime_confidence = 0.0
        self.regime_size_multiplier = 0.0
        self.active_router_regime = 'transition'

    def Debug(self, _msg):
        pass


class RegimeRouterConfidenceTests(unittest.TestCase):
    def test_trend_regime_high_confidence_enables_larger_size(self):
        algo = _Algo('bull', 0.85, [26, 27, 25, 28, 30, 24])
        router = RegimeRouter(algo)
        router.TRANSITION_HOLD = 0
        for _ in range(router.REGIME_MIN_BARS):
            router.update()
        self.assertEqual(router.route(), 'trend')
        self.assertGreaterEqual(router.get_confidence(), 0.70)
        self.assertGreaterEqual(router.get_size_multiplier('trend'), 0.80)

    def test_transition_regime_blocks_new_size(self):
        algo = _Algo('sideways', 0.95, [30, 31, 33, 29, 28, 34])
        router = RegimeRouter(algo)
        router.TRANSITION_HOLD = 0
        for _ in range(router.REGIME_MIN_BARS):
            router.update()
        self.assertEqual(router.route(), 'transition')
        self.assertEqual(router.get_size_multiplier('transition'), 0.0)


if __name__ == '__main__':
    unittest.main()
