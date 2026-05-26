from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig
from ensemble import AlphaEnsemble
from ml_scorer import MLScorer
from regime_ensemble import _DEFAULT_REGIME_MAP, _WEIGHT_KEYS, load_regime_weights, normalize_regime_key
from walk_forward_engine import _sharpe, prepare_bar_panel, simulate_oos_fold


def regime_label_from_btc_row(row: pd.Series, *, vol_chaos: float = 1.05) -> str:
    ret7 = float(row.get("ret168", 0.0))
    trend = float(row.get("ema20", 0.0)) / max(float(row.get("ema50", 1.0)), 1e-9) - 1.0
    rv = float(row.get("rv", 0.2))
    if rv >= vol_chaos:
        return "chaos"
    if ret7 > 0.03 and trend > 0:
        return "bull"
    if ret7 < -0.03:
        return "bear"
    return "neutral"


def build_regime_index(panel: dict[str, pd.DataFrame], timestamps: list) -> list[str]:
    btc = panel.get("BTCUSD")
    labels: list[str] = []
    for i, _ts in enumerate(timestamps):
        if btc is None or i >= len(btc):
            labels.append("neutral")
            continue
        labels.append(regime_label_from_btc_row(btc.iloc[i]))
    return labels


def _weight_grid() -> list[dict[str, float]]:
    keys = list(_WEIGHT_KEYS)
    grid = {
        "w_momentum": [0.25, 0.35, 0.40],
        "w_breakout": [0.15, 0.25, 0.30],
        "w_dip": [0.10, 0.15],
        "w_ml": [0.20, 0.25, 0.30],
    }
    combos = []
    for vals in product(*[grid[k] for k in keys]):
        w = dict(zip(keys, vals))
        s = sum(w.values())
        if s <= 0:
            continue
        combos.append({k: v / s for k, v in w.items()})
    return combos


def _eval_regime_weights(
    panel: dict[str, pd.DataFrame],
    timestamps: list,
    regime_labels: list[str],
    target_regime: str,
    weights: dict[str, float],
    config: KrakenMaxConfig,
) -> float:
    cfg = replace(config, **{k: float(weights[k]) for k in _WEIGHT_KEYS if k in weights})
    ens = AlphaEnsemble(cfg, MLScorer())
    indices = [i for i, lab in enumerate(regime_labels) if lab == target_regime]
    if len(indices) < 40:
        return -1e9
    chunks: list[float] = []
    step = max(20, len(indices) // 8)
    for start in range(0, len(indices) - step, step):
        i0 = indices[start]
        i1 = indices[min(start + step, len(indices) - 1)]
        if i1 <= i0 + 5:
            continue
        rets, _, _ = simulate_oos_fold(
            panel,
            train_end_idx=i0,
            test_end_idx=i1,
            timestamps=timestamps,
            config=cfg,
            ensemble=ens,
        )
        if len(rets):
            chunks.append(_sharpe(np.array(rets), periods_per_year=24 * 365 * config.bph()))
    return float(np.mean(chunks)) if chunks else -1e9


def optimize_regime_weights(
    bars: pd.DataFrame,
    *,
    config: KrakenMaxConfig = CONFIG,
    regimes: tuple[str, ...] = ("bull", "neutral", "bear", "chaos"),
) -> dict[str, dict[str, float]]:
    """Grid-search ensemble weights per labeled BTC regime (v8)."""
    panel = prepare_bar_panel(bars, config)
    ts = sorted(set(pd.to_datetime(bars["timestamp"], utc=True)))
    labels = build_regime_index(panel, ts)
    defaults = load_regime_weights()
    out: dict[str, dict[str, float]] = {k: dict(defaults.get(k, _DEFAULT_REGIME_MAP.get(k, {}))) for k in regimes}
    grid = _weight_grid()
    for regime in regimes:
        if labels.count(regime) < int(config.regime_wf_min_bars):
            continue
        best_sh = -1e9
        best_w = out[regime]
        for combo in grid:
            sh = _eval_regime_weights(panel, ts, labels, regime, combo, config)
            if sh > best_sh:
                best_sh = sh
                best_w = combo
        if best_w:
            out[regime] = {k: float(best_w[k]) for k in _WEIGHT_KEYS if k in best_w}
    return out


def save_regime_weights(
    regimes: dict[str, dict[str, float]],
    path: Path | None = None,
    *,
    metrics: dict[str, Any] | None = None,
) -> Path:
    target = path or (Path(__file__).resolve().parent / str(CONFIG.regime_weights_path))
    payload = {
        "regimes": regimes,
        "metrics": metrics or {},
        "source": "regime_walk_forward_v8",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_regime_weights_merged(path: Path | None = None) -> dict[str, dict[str, float]]:
    base = load_regime_weights(path)
    target = path or (Path(__file__).resolve().parent / str(CONFIG.regime_weights_path))
    if not target.exists():
        return base
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
        if blob.get("source", "").startswith("regime_walk_forward"):
            regimes = blob.get("regimes") or {}
            for name, w in regimes.items():
                key = normalize_regime_key(name)
                base[key] = {k: float(w[k]) for k in _WEIGHT_KEYS if k in w}
    except Exception:
        pass
    return base
