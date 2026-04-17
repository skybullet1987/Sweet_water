import unittest

import numpy as np

from config.strategy_config import StrategyConfig
from regime.hmm import HMMRegime


class HMMRegimeTests(unittest.TestCase):
    def test_synthetic_two_regime_assignment(self):
        np.random.seed(7)
        cfg = StrategyConfig(hmm_train_window_bars=120, hmm_retrain_every_bars=40)
        model = HMMRegime(cfg)

        trend_hits = 0
        chop_hits = 0
        trend_eval = 0
        chop_eval = 0
        trend_total = 200
        chop_total = 200

        trend_returns = np.random.normal(loc=0.002, scale=0.001, size=trend_total)
        chop_returns = np.random.normal(loc=-0.01, scale=0.01, size=chop_total)

        for i, ret in enumerate(np.concatenate([trend_returns, chop_returns])):
            vol = 0.01 if i < trend_total else 0.10
            breadth = 0.8 if i < trend_total else 0.1
            model.update(ret, vol, breadth)
            state = model.current_state()
            if i < trend_total and i >= cfg.hmm_train_window_bars:
                trend_eval += 1
                if state == "risk_on":
                    trend_hits += 1
            if i >= trend_total and i >= cfg.hmm_train_window_bars:
                chop_eval += 1
                if state in {"chop", "risk_off"}:
                    chop_hits += 1

        self.assertGreaterEqual(trend_hits / trend_eval, 0.80)
        self.assertGreaterEqual(chop_hits / chop_eval, 0.80)


if __name__ == "__main__":
    unittest.main()
