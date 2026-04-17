from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_vs_universe(symbol_returns_df: pd.DataFrame) -> pd.DataFrame:
    row_mean = symbol_returns_df.mean(axis=1)
    row_std = symbol_returns_df.std(axis=1, ddof=0).replace(0.0, np.nan)
    return symbol_returns_df.sub(row_mean, axis=0).div(row_std, axis=0)


def rank_momentum(symbol_returns_df: pd.DataFrame, window: int) -> pd.DataFrame:
    cumulative = (1.0 + symbol_returns_df).rolling(window=window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0
    return cumulative.rank(axis=1, pct=True, method="average")
