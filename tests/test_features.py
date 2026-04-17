import unittest

import numpy as np
import pandas as pd

from features.cross_sectional import rank_momentum, zscore_vs_universe
from features.microstructure import (
    amihud_illiquidity,
    kyle_lambda_proxy,
    ofi_proxy,
    realized_vol,
    roll_spread,
)


class FeatureTests(unittest.TestCase):
    def test_amihud_matches_manual_fixture(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01])
        dv = pd.Series([1000.0, 2000.0, 3000.0, 4000.0])
        out = amihud_illiquidity(returns, dv, window=2)
        expected_last = ((abs(0.03) / 3000.0) + (abs(-0.01) / 4000.0)) / 2.0
        self.assertAlmostEqual(float(out.iloc[-1]), expected_last, places=10)

    def test_roll_spread_nan_for_non_negative_cov(self):
        close = pd.Series([1, 2, 3, 4, 5], dtype=float)
        out = roll_spread(close, window=3)
        self.assertTrue(np.isnan(out.iloc[-1]))

    def test_kyle_lambda_proxy_slope(self):
        r = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        sv = pd.Series([1, 2, 3, 4, 5], dtype=float)
        out = kyle_lambda_proxy(r, sv, window=5)
        self.assertAlmostEqual(float(out.iloc[-1]), 0.01, places=6)

    def test_realized_vol_annualizes_1h(self):
        r = pd.Series([0.01, -0.01, 0.01, -0.01], dtype=float)
        out = realized_vol(r, window=4)
        expected = r.std(ddof=1) * np.sqrt(24 * 365)
        self.assertAlmostEqual(float(out.iloc[-1]), float(expected), places=10)

    def test_ofi_proxy_known_value(self):
        out = ofi_proxy(
            pd.Series([1.0]),
            pd.Series([11.0]),
            pd.Series([9.0]),
            pd.Series([10.5]),
            pd.Series([100.0]),
        )
        self.assertAlmostEqual(float(out.iloc[-1]), 25.0, places=8)

    def test_cross_sectional_zscore_and_rank_are_causal(self):
        rets = pd.DataFrame(
            {
                "A": [0.0, 0.1, 0.2],
                "B": [0.0, 0.2, 0.1],
                "C": [0.0, -0.1, 0.3],
            }
        )
        z = zscore_vs_universe(rets)
        self.assertAlmostEqual(float(z.iloc[1].mean()), 0.0, places=10)

        ranks = rank_momentum(rets, window=2)
        self.assertTrue(ranks.iloc[0].isna().all())
        self.assertTrue((ranks.iloc[-1] >= 1 / 3).all())


if __name__ == "__main__":
    unittest.main()
