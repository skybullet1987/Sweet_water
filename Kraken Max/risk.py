"""Kraken Max — risk, clusters, ERC (`risk.py`)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig

# --- from risk.py ---


from config import CONFIG, KrakenMaxConfig


@dataclass
class PositionRisk:
    entry_price: float
    entry_time: datetime
    entry_atr: float
    highest_close: float
    pyramid_count: int = 0
    predicted_score: float = 0.0
    strategy_owner: str = "momentum"  # momentum | scalper
    stop_price: float | None = None
    take_profit_price: float | None = None


class PortfolioRisk:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.peak_equity = 0.0
        self.halted_until: datetime | None = None
        self.orders_today = 0
        self._order_day = None

    def update_peak(self, equity: float) -> float:
        self.peak_equity = max(self.peak_equity, float(equity))
        if self.peak_equity <= 0:
            return 0.0
        return (float(equity) / self.peak_equity) - 1.0

    def drawdown_halted(self, now: datetime, drawdown: float) -> bool:
        if self.halted_until and now < self.halted_until:
            return True
        if drawdown <= float(self.config.drawdown_halt_pct):
            hours = int(self.config.drawdown_cooldown_hours)
            self.halted_until = now + timedelta(hours=hours)
            return True
        if self.halted_until and now >= self.halted_until:
            self.halted_until = None
        return False

    def can_place_order(self, now: datetime) -> bool:
        day = now.date()
        if self._order_day != day:
            self._order_day = day
            self.orders_today = 0
        return self.orders_today < int(self.config.max_orders_per_day)

    def record_order(self) -> None:
        self.orders_today += 1


def should_exit(
    state: PositionRisk,
    *,
    close: float,
    now: datetime,
    hard_stop_pct: float,
    catastrophic_stop_pct: float,
    tp_atr_mult: float,
    sl_atr_mult: float,
    chandelier_atr_mult: float,
    activate_trail_pct: float,
    time_stop_hours: float,
) -> tuple[bool, str]:
    if state.entry_price <= 0:
        return False, ""
    pnl_pct = (close / state.entry_price) - 1.0
    if pnl_pct <= catastrophic_stop_pct:
        return True, "catastrophic"
    if pnl_pct <= hard_stop_pct:
        return True, "hard_stop"
    atr = max(state.entry_atr, state.entry_price * 0.01)
    stop = state.entry_price - sl_atr_mult * atr
    if close <= stop:
        return True, "atr_stop"
    tp = state.entry_price + tp_atr_mult * atr
    if close >= tp:
        return True, "take_profit"
    state.highest_close = max(state.highest_close, close)
    if pnl_pct >= activate_trail_pct:
        trail = state.highest_close - chandelier_atr_mult * atr
        if close <= trail:
            return True, "chandelier"
    held = (now - state.entry_time).total_seconds() / 3600.0
    if held >= time_stop_hours and pnl_pct < 0.02:
        return True, "time_stop"
    return False, ""

# --- from cluster_risk.py ---


# Beta / narrative clusters for portfolio-level caps (v7)
DEFAULT_CLUSTERS: dict[str, tuple[str, ...]] = {
    "major": ("BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"),
    "l1": ("SOLUSD", "AVAXUSD", "DOTUSD", "ADAUSD", "ATOMUSD"),
    "defi": ("LINKUSD", "UNIUSD", "AAVEUSD"),
    "meme": ("DOGEUSD", "SHIBUSD", "PEPEUSD"),
    "legacy": ("XRPUSD", "XLMUSD", "ETCUSD"),
}


def cluster_for_ticker(ticker: str, clusters: dict[str, tuple[str, ...]] | None = None) -> str:
    mapping = clusters or DEFAULT_CLUSTERS
    for name, members in mapping.items():
        if ticker in members:
            return name
    return "other"


def filter_cluster_caps(
    ranked: list[tuple[str, float]],
    *,
    current_holdings: list[str] | None = None,
    max_per_cluster: int | None = None,
    clusters: dict[str, tuple[str, ...]] | None = None,
    config: KrakenMaxConfig = CONFIG,
) -> list[str]:
    """Greedy pick from ranked list respecting per-cluster position caps."""
    cap = int(max_per_cluster if max_per_cluster is not None else config.max_positions_per_cluster)
    counts: dict[str, int] = {}
    for t in current_holdings or []:
        c = cluster_for_ticker(t, clusters)
        counts[c] = counts.get(c, 0) + 1
    selected: list[str] = []
    for ticker, _score in ranked:
        c = cluster_for_ticker(ticker, clusters)
        if counts.get(c, 0) >= cap:
            continue
        selected.append(ticker)
        counts[c] = counts.get(c, 0) + 1
    return selected


def max_cluster_exposure(weights: dict[str, float], clusters: dict[str, tuple[str, ...]] | None = None) -> dict[str, float]:
    """Sum ERC/target weights by cluster for telemetry."""
    mapping = clusters or DEFAULT_CLUSTERS
    out: dict[str, float] = {}
    for ticker, w in weights.items():
        c = cluster_for_ticker(ticker, mapping)
        out[c] = out.get(c, 0.0) + float(w)
    return out

# --- from portfolio_optimizer.py ---


from config import CONFIG, KrakenMaxConfig


def _covariance_matrix(cache, tickers: list[str], lookback_hours: int) -> tuple[pd.DataFrame, list[str]]:
    from core import hourly_returns  # portfolio_optimizer

    lookback_bars = CONFIG.lookback_bars(lookback_hours)
    series_map: dict[str, pd.Series] = {}
    for t in tickers:
        frame = cache.frame(t)
        if frame is None or frame.empty:
            continue
        rets = hourly_returns(frame.tail(lookback_bars))
        if len(rets) >= int(CONFIG.min_corr_samples):
            series_map[t] = rets
    if len(series_map) < 2:
        return pd.DataFrame(), []
    aligned = pd.DataFrame(series_map).dropna(how="any")
    if aligned.shape[0] < int(CONFIG.min_corr_samples):
        return pd.DataFrame(), []
    return aligned.cov(), list(aligned.columns)


def shrink_covariance(cov: pd.DataFrame, intensity: float) -> pd.DataFrame:
    """Ledoit-style shrinkage toward diagonal (v5)."""
    if cov.empty:
        return cov
    k = float(intensity)
    k = max(0.0, min(1.0, k))
    target = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
    return (1.0 - k) * cov + k * target


def erc_weights(cov: pd.DataFrame, max_iter: int = 200, tol: float = 1e-8) -> dict[str, float]:
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


def _blend_weights(
    new_w: dict[str, float],
    prev_w: dict[str, float],
    penalty: float,
) -> dict[str, float]:
    if not prev_w or penalty <= 0:
        return new_w
    keys = set(new_w) | set(prev_w)
    blended = {}
    for k in keys:
        p = float(prev_w.get(k, 0.0))
        n = float(new_w.get(k, 0.0))
        blended[k] = (1.0 - penalty) * n + penalty * p
    s = sum(blended.values())
    if s <= 0:
        return new_w
    return {k: v / s for k, v in blended.items()}


def allocate_erc_notionals(
    targets: list[str],
    cache,
    equity: float,
    deployment_cap: float,
    *,
    config: KrakenMaxConfig = CONFIG,
    previous_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if not targets:
        return {}
    cov, valid = _covariance_matrix(cache, targets, int(config.corr_lookback_hours))
    if not valid:
        per = equity * deployment_cap / len(targets)
        return {t: per for t in targets}
    sub = cov.loc[valid, valid] if set(valid).issubset(cov.index) else cov
    sub = shrink_covariance(sub, float(config.erc_shrinkage))
    weights = erc_weights(sub)
    weights = _blend_weights(weights, previous_weights or {}, float(config.erc_turnover_penalty))
    deployable = equity * deployment_cap
    out = {t: deployable * weights[t] for t in targets if t in weights}
    if not out:
        per = deployable / len(targets)
        return {t: per for t in targets}
    return out