from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig


def hourly_returns(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "close" not in frame.columns:
        return pd.Series(dtype=float)
    close = frame["close"].astype(float)
    return close.pct_change().dropna()


def return_correlation(
    cache,
    tickers: list[str],
    *,
    lookback_hours: int | None = None,
    min_samples: int | None = None,
) -> pd.DataFrame:
    lookback = int(lookback_hours or CONFIG.corr_lookback_hours)
    min_n = int(min_samples or CONFIG.min_corr_samples)
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        frame = cache.frame(ticker)
        if frame is None or frame.empty:
            continue
        rets = hourly_returns(frame.tail(lookback))
        if len(rets) >= min_n:
            series_map[ticker] = rets
    if len(series_map) < 2:
        return pd.DataFrame()
    aligned = pd.DataFrame(series_map).dropna(how="any")
    if aligned.shape[0] < min_n:
        return pd.DataFrame()
    return aligned.corr()


def max_corr_to_selected(ticker: str, selected: list[str], corr: pd.DataFrame) -> float:
    if not selected or corr.empty or ticker not in corr.columns:
        return 0.0
    vals = []
    for other in selected:
        if other == ticker or other not in corr.columns:
            continue
        try:
            vals.append(abs(float(corr.loc[ticker, other])))
        except Exception:
            continue
    return max(vals) if vals else 0.0


def filter_uncorrelated_picks(
    ranked: list[tuple[str, float]],
    cache,
    *,
    top_k: int | None = None,
    max_corr: float | None = None,
    config: KrakenMaxConfig = CONFIG,
) -> list[str]:
    """Greedy decorrelation: highest score first, skip high-beta clones."""
    k = int(top_k or config.top_k)
    cap = float(max_corr if max_corr is not None else config.max_pairwise_corr)
    tickers = [t for t, _ in ranked]
    corr = return_correlation(cache, tickers)
    chosen: list[str] = []
    for ticker, _score in ranked:
        if len(chosen) >= k:
            break
        if corr.empty:
            chosen.append(ticker)
            continue
        if max_corr_to_selected(ticker, chosen, corr) <= cap:
            chosen.append(ticker)
    return chosen
