import unittest

from scoring.rule_scorer import score_symbol


class RuleScorerTests(unittest.TestCase):
    def test_risk_on_long_score_positive_and_thresholded(self):
        features = {
            "adx": 30,
            "ema20": 101,
            "ema50": 100,
            "macd_hist": 1,
            "aroon_osc": 10,
            "mfi": 60,
        }
        score = score_symbol(features, regime="risk_on", btc_context={"btc_trend": 0.01})
        self.assertGreaterEqual(score, 0.4)

    def test_chop_short_negative(self):
        features = {"cci": 150, "bb_pos": 0.95, "rsi": 80}
        score = score_symbol(features, regime="chop")
        self.assertLessEqual(score, -0.4)

    def test_risk_off_forces_zero(self):
        score = score_symbol({"adx": 99}, regime="risk_off")
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
