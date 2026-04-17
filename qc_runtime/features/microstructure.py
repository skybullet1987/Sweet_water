from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series, window: int) -> pd.Series:
    ratio = returns.abs() / dollar_volume.replace(0.0, np.nan)
    return ratio.rolling(window=window, min_periods=window).mean()


def roll_spread(close: pd.Series, window: int) -> pd.Series:
    dp = close.diff()
    lag = dp.shift(1)
    rolling_cov = dp.rolling(window=window, min_periods=window).cov(lag)
    spread = pd.Series(np.nan, index=close.index, dtype=float)
    mask = rolling_cov < 0
    spread.loc[mask] = 2.0 * np.sqrt((-rolling_cov[mask]).clip(lower=0.0))
    return spread


def kyle_lambda_proxy(returns: pd.Series, signed_volume: pd.Series, window: int) -> pd.Series:
    cov = returns.rolling(window=window, min_periods=window).cov(signed_volume)
    var = signed_volume.rolling(window=window, min_periods=window).var()
    return cov / var.replace(0.0, np.nan)


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    annualization = np.sqrt(24 * 365)
    return returns.rolling(window=window, min_periods=window).std() * annualization


def ofi_proxy(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    eps = 1e-12
    midpoint = (high + low) / 2.0
    return volume * (close - midpoint) / ((high - low).abs() + eps)
