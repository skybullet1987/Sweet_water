from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig


def _covariance_matrix(cache, tickers: list[str], lookback_hours: int) -> tuple[pd.DataFrame, list[str]]:
    from correlation import hourly_returns

    series_map: dict[str, pd.Series] = {}
    for t in tickers:
        frame = cache.frame(t)
        if frame is None or frame.empty:
            continue
        rets = hourly_returns(frame.tail(lookback_hours))
        if len(rets) >= int(CONFIG.min_corr_samples):
            series_map[t] = rets
    if len(series_map) < 2:
        return pd.DataFrame(), []
    aligned = pd.DataFrame(series_map).dropna(how="any")
    if aligned.shape[0] < int(CONFIG.min_corr_samples):
        return pd.DataFrame(), []
    return aligned.cov(), list(aligned.columns)


def erc_weights(cov: pd.DataFrame, max_iter: int = 200, tol: float = 1e-8) -> dict[str, float]:
    """Equal risk contribution weights (long-only, sum=1)."""
    cols = list(cov.columns)
    n = len(cols)
    if n == 0:
        return {}
    if n == 1:
        return {cols[0]: 1.0}
    C = cov.values.astype(float)
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma_w = C @ w
        risk_contrib = w * sigma_w
        target = float(np.sum(risk_contrib)) / n
        grad = risk_contrib - target
        w = w - 0.01 * grad
        w = np.clip(w, 1e-6, 1.0)
        w = w / w.sum()
        if float(np.max(np.abs(grad))) < tol:
            break
    return {cols[i]: float(w[i]) for i in range(n)}


def allocate_erc_notionals(
    targets: list[str],
    cache,
    equity: float,
    deployment_cap: float,
    *,
    config: KrakenMaxConfig = CONFIG,
) -> dict[str, float]:
    if not targets:
        return {}
    cov, valid = _covariance_matrix(cache, targets, int(config.corr_lookback_hours))
    if not valid:
        per = equity * deployment_cap / len(targets)
        return {t: per for t in targets}
    sub = cov.loc[valid, valid] if set(valid).issubset(cov.index) else cov
    weights = erc_weights(sub)
    deployable = equity * deployment_cap
    out: dict[str, float] = {}
    for t in targets:
        if t in weights:
            out[t] = deployable * weights[t]
    if not out:
        per = deployable / len(targets)
        return {t: per for t in targets}
    return out
