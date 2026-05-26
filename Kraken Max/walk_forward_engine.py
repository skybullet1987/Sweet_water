from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig
from ensemble import AlphaEnsemble
from ml_scorer import MLScorer


@dataclass(frozen=True)
class WalkForwardResult:
    oos_sharpe: float
    oos_max_drawdown: float
    oos_total_return: float
    oos_trades: int
    oos_win_rate: float
    best_weights: dict[str, float]
    fold_metrics: list[dict[str, float]]


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-9) - 1.0
    return float(dd.min())


def _sharpe(returns: np.ndarray, periods_per_year: float = 24 * 365) -> float:
    if len(returns) < 3:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(periods_per_year)


def prepare_hourly_panel(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = bars.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["symbol", "timestamp"])
    out: dict[str, pd.DataFrame] = {}
    for sym, grp in data.groupby("symbol"):
        f = grp.reset_index(drop=True)
        f["ret"] = f["close"].pct_change().fillna(0.0)
        f["ret24"] = f["close"].pct_change(24).fillna(0.0)
        f["ret168"] = f["close"].pct_change(168).fillna(0.0)
        f["ema20"] = f["close"].ewm(span=20, adjust=False).mean()
        f["ema50"] = f["close"].ewm(span=50, adjust=False).mean()
        f["atr"] = (f["high"] - f["low"]).rolling(14, min_periods=1).mean().bfill()
        f["rv"] = f["ret"].rolling(24 * 21, min_periods=24).std() * math.sqrt(24 * 365)
        f["breakout"] = f["close"] / f["high"].rolling(20 * 24, min_periods=5).max() - 1.0
        out[sym] = f
    return out


def _score_row(row: pd.Series, rank24: float, rank_bo: float, breadth: float, ensemble: AlphaEnsemble) -> float:
    feats = {
        "mom_7d": float(row.get("ret168", 0.0) / 3.0),
        "mom_21d": float(row.get("ret168", 0.0)),
        "mom_accel": float(row.get("ret24", 0.0)) - float(row.get("ret168", 0.0)) / 6.0,
        "rv_21d": max(float(row.get("rv", 0.2)), 1e-6),
        "breakout_strength": float(row.get("breakout", 0.0)),
        "volume_surge": 0.0,
        "trend_quality": float(row.get("ema20", 0.0) / max(row.get("ema50", 1.0), 1e-9) - 1.0),
        "rsi_pullback": 0.0,
        "rv_21d_inv": 1.0 / max(float(row.get("rv", 0.2)), 1e-6),
    }
    comp = ensemble.score_symbol(feats, rank_mom_21=rank24, rank_breakout=rank_bo, breadth=breadth)
    return float(comp["final"])


def simulate_oos_fold(
    panel: dict[str, pd.DataFrame],
    *,
    train_end_idx: int,
    test_end_idx: int,
    timestamps: list,
    config: KrakenMaxConfig,
    ensemble: AlphaEnsemble,
) -> tuple[np.ndarray, list[float], int]:
    equity = float(config.starting_cash)
    curve = [equity]
    trade_pnls: list[float] = []
    trades = 0
    syms = [s for s in panel.keys() if s != "BTCUSD"] or list(panel.keys())

    for i in range(train_end_idx, test_end_idx):
        ts = timestamps[i]
        r24 = {}
        rbo = {}
        for s in syms:
            f = panel[s]
            if i >= len(f):
                continue
            r24[s] = float(f.iloc[i]["ret24"])
            rbo[s] = float(f.iloc[i]["breakout"])
        if not r24:
            curve.append(equity)
            continue
        rank24 = {s: (sorted(r24.values()).index(v) / max(len(r24) - 1, 1)) for s, v in r24.items()}
        rankbo = {s: (sorted(rbo.values()).index(v) / max(len(rbo) - 1, 1)) for s, v in rbo.items()}
        breadth = sum(1 for v in r24.values() if v > 0) / len(r24)

        best_sym = None
        best_sc = -1e9
        for s in syms:
            f = panel[s]
            if i + 24 >= len(f) or i >= len(f):
                continue
            sc = _score_row(f.iloc[i], rank24.get(s, 0.5), rankbo.get(s, 0.5), breadth, ensemble)
            if sc > best_sc:
                best_sc = sc
                best_sym = s
        if best_sym is None or best_sc < float(config.entry_score_threshold):
            curve.append(equity)
            continue

        f = panel[best_sym]
        entry = float(f.iloc[i]["close"])
        atr = max(float(f.iloc[i]["atr"]), entry * 0.01)
        exit_i = min(i + int(config.time_stop_hours), len(f) - 1)
        exit_px = float(f.iloc[exit_i]["close"])
        for j in range(i + 1, exit_i + 1):
            row = f.iloc[j]
            if float(row["low"]) <= entry - config.sl_atr_mult * atr:
                exit_px = entry - config.sl_atr_mult * atr
                break
            if float(row["high"]) >= entry + config.tp_atr_mult * atr:
                exit_px = entry + config.tp_atr_mult * atr
                break
        pnl = (exit_px / entry - 1.0) - float(config.expected_round_trip_fees)
        equity *= 1.0 + pnl * min(0.35, float(config.max_position_pct))
        trade_pnls.append(pnl)
        trades += 1
        curve.append(equity)

    rets = np.diff(np.array(curve)) / np.maximum(np.array(curve[:-1]), 1e-9)
    return rets, trade_pnls, trades


def walk_forward_optimize(
    bars: pd.DataFrame,
    *,
    n_folds: int = 4,
    config: KrakenMaxConfig = CONFIG,
    weight_grid: dict[str, list[float]] | None = None,
) -> WalkForwardResult:
    panel = prepare_hourly_panel(bars)
    if "BTCUSD" not in panel:
        raise ValueError("bars must include BTCUSD")
    ts = sorted(set(pd.to_datetime(bars["timestamp"], utc=True)))
    if len(ts) < 400:
        raise ValueError("need at least 400 hourly timestamps for walk-forward")

    grid = weight_grid or {
        "w_momentum": [0.30, 0.35, 0.40],
        "w_breakout": [0.20, 0.25, 0.30],
        "w_dip": [0.10, 0.15],
        "w_ml": [0.20, 0.25],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]

    fold_size = len(ts) // (n_folds + 1)
    best_combo = combos[0]
    best_sharpe = -1e9
    fold_metrics: list[dict[str, float]] = []

    for combo in combos:
        cfg = replace(config, **combo)
        ens = AlphaEnsemble(cfg, MLScorer())
        fold_sharpes = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_end = min(train_end + fold_size, len(ts) - 1)
            rets, _pnls, _tr = simulate_oos_fold(
                panel, train_end_idx=train_end, test_end_idx=test_end, timestamps=ts, config=cfg, ensemble=ens
            )
            fold_sharpes.append(_sharpe(rets))
        mean_sh = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        if mean_sh > best_sharpe:
            best_sharpe = mean_sh
            best_combo = combo

    final_cfg = replace(config, **best_combo)
    final_ens = AlphaEnsemble(final_cfg, MLScorer())
    all_rets: list[float] = []
    all_pnls: list[float] = []
    total_trades = 0
    equity_curve = [float(config.starting_cash)]

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end = min(train_end + fold_size, len(ts) - 1)
        rets, pnls, tr = simulate_oos_fold(
            panel, train_end_idx=train_end, test_end_idx=test_end, timestamps=ts, config=final_cfg, ensemble=final_ens
        )
        all_rets.extend(rets.tolist())
        all_pnls.extend(pnls)
        total_trades += tr
        if len(rets):
            equity_curve.append(equity_curve[-1] * float(np.prod(1.0 + rets)))
        fold_metrics.append({
            "fold": float(fold),
            "sharpe": _sharpe(rets),
            "trades": float(tr),
        })

    eq = np.array(equity_curve)
    return WalkForwardResult(
        oos_sharpe=_sharpe(np.array(all_rets)),
        oos_max_drawdown=_max_drawdown(eq),
        oos_total_return=float(eq[-1] / eq[0] - 1.0) if len(eq) else 0.0,
        oos_trades=total_trades,
        oos_win_rate=float(sum(1 for p in all_pnls if p > 0) / max(len(all_pnls), 1)),
        best_weights=best_combo,
        fold_metrics=fold_metrics,
    )


def save_ensemble_weights(result: WalkForwardResult, path: Path | None = None) -> Path:
    target = path or (Path(__file__).resolve().parent / "ensemble_weights.json")
    payload: dict[str, Any] = {
        "ensemble": result.best_weights,
        "metrics": {
            "oos_sharpe": result.oos_sharpe,
            "oos_max_drawdown": result.oos_max_drawdown,
            "oos_total_return": result.oos_total_return,
            "oos_trades": result.oos_trades,
            "oos_win_rate": result.oos_win_rate,
        },
        "folds": result.fold_metrics,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
