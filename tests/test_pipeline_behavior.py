import unittest

import pandas as pd

from features.indicators import adx, ema
from regime.hmm import HMMRegime, build_hmm_features
from scoring.rule_scorer import score_symbol


class PipelineBehaviorTests(unittest.TestCase):
    def test_phase1_pipeline_components_interoperate(self):
        close = pd.Series([100 + i * 0.2 for i in range(200)])
        frame = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000,
            }
        )

        features = {
            "adx": float(adx(frame, 14).iloc[-1]),
            "ema20": float(ema(frame, 20).iloc[-1]),
            "ema50": float(ema(frame, 50).iloc[-1]),
            "macd_hist": 0.1,
            "aroon_osc": 20.0,
            "mfi": 70.0,
        }
        score = score_symbol(features, regime="risk_on")
        self.assertGreaterEqual(score, 0.4)

        breadth = pd.Series([0.7] * len(close))
        hmm_df = build_hmm_features(close, breadth)
        hmm = HMMRegime()
        for _, row in hmm_df.iterrows():
            hmm.update(row["btc_log_return"], row["btc_realized_vol"], row["breadth"])
        self.assertIn(hmm.current_state(), {"risk_on", "risk_off", "chop"})


if __name__ == "__main__":
    unittest.main()
