"""Kraken Max — walk-forward, validation, revalidation (`workflow.py`)."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig
from kraken_ops import BaselineManager, DriftMonitor
from regime import (
    _DEFAULT_REGIME_MAP,
    _WEIGHT_KEYS,
    load_regime_weights,
    normalize_regime_key,
)

# --- from bars_util.py ---



def consolidate_minute_ohlcv(df: pd.DataFrame, bar_minutes: int = 15) -> pd.DataFrame:
    """Consolidate minute OHLCV to N-minute bars per symbol (v8 native sub-hour)."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    rows: list[dict] = []
    rule = f"{int(bar_minutes)}min"
    for sym, grp in data.groupby("symbol"):
        g = grp.set_index("timestamp").sort_index()
        ohlc = pd.DataFrame(
            {
                "open": g["open"].resample(rule).first(),
                "high": g["high"].resample(rule).max(),
                "low": g["low"].resample(rule).min(),
                "close": g["close"].resample(rule).last(),
                "volume": g["volume"].resample(rule).sum(),
            }
        ).dropna(subset=["close"])
        for ts, row in ohlc.iterrows():
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
    return pd.DataFrame(rows)

# --- from walk_forward_engine.py ---


import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig


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


def periods_per_year(config: KrakenMaxConfig = CONFIG) -> float:
    return float(24 * 365 * config.bph())


def infer_bar_minutes(timestamps: list) -> int:
    if len(timestamps) < 2:
        return 60
    ts = pd.Series(pd.to_datetime(timestamps, utc=True)).sort_values()
    median_min = ts.diff().dropna().median().total_seconds() / 60.0
    if median_min <= 20:
        return 15
    if median_min <= 90:
        return 60
    return 60


def bar_step_hours(config: KrakenMaxConfig = CONFIG) -> float:
    return max(1.0 / config.bph(), float(config.resolution_minutes) / 60.0)


def prepare_bar_panel(bars: pd.DataFrame, config: KrakenMaxConfig = CONFIG) -> dict[str, pd.DataFrame]:
    """Feature panel at config bar frequency (15m default in v6, hourly if resolution_minutes=60)."""
    bph = config.bph()
    bars_24h = 24 * bph
    bars_7d = 24 * 7 * bph
    bars_20d = 20 * 24 * bph
    bars_21d = 24 * 21 * bph
    ema_fast = max(20 * bph, 20)
    ema_slow = max(50 * bph, 50)
    atr_win = max(14 * bph, 14)

    data = bars.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["symbol", "timestamp"])
    out: dict[str, pd.DataFrame] = {}
    ppy = periods_per_year(config)
    for sym, grp in data.groupby("symbol"):
        f = grp.reset_index(drop=True)
        f["ret"] = f["close"].pct_change().fillna(0.0)
        f["ret24"] = f["close"].pct_change(bars_24h).fillna(0.0)
        f["ret168"] = f["close"].pct_change(bars_7d).fillna(0.0)
        f["ema20"] = f["close"].ewm(span=ema_fast, adjust=False).mean()
        f["ema50"] = f["close"].ewm(span=ema_slow, adjust=False).mean()
        f["atr"] = (f["high"] - f["low"]).rolling(atr_win, min_periods=1).mean().bfill()
        f["rv"] = f["ret"].rolling(bars_21d, min_periods=max(bph, 24)).std() * math.sqrt(ppy)
        f["breakout"] = f["close"] / f["high"].rolling(bars_20d, min_periods=5).max() - 1.0
        out[sym] = f
    return out


def prepare_hourly_panel(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from dataclasses import replace

    hourly = replace(CONFIG, resolution_minutes=60, use_sub_hour_bars=False)
    return prepare_bar_panel(bars, hourly)


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
        hold_fwd = max(24 * config.bph(), 24)
        for s in syms:
            f = panel[s]
            if i + hold_fwd >= len(f) or i >= len(f):
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
        hold_bars = max(1, int(config.time_stop_hours * config.bph()))
        exit_i = min(i + hold_bars, len(f) - 1)
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


def resample_bars_to_minutes(bars: pd.DataFrame, bar_minutes: int = 15) -> pd.DataFrame:
    """Upsample hourly OHLCV to synthetic N-minute bars for local walk-forward smoke tests."""
    data = bars.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    rule = f"{int(bar_minutes)}min"
    out_rows: list[dict] = []
    for sym, grp in data.groupby("symbol"):
        g = grp.set_index("timestamp").sort_index()
        ohlc = g[["open", "high", "low", "close", "volume"]].resample(rule).ffill()
        ohlc = ohlc.dropna(subset=["close"])
        for ts, row in ohlc.iterrows():
            out_rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
    return pd.DataFrame(out_rows)


def walk_forward_optimize(
    bars: pd.DataFrame,
    *,
    n_folds: int = 4,
    config: KrakenMaxConfig = CONFIG,
    weight_grid: dict[str, list[float]] | None = None,
    bar_minutes: int | None = None,
) -> WalkForwardResult:
    ts = sorted(set(pd.to_datetime(bars["timestamp"], utc=True)))
    inferred = infer_bar_minutes(ts)
    use_minutes = int(bar_minutes) if bar_minutes is not None else inferred
    if bar_minutes is not None or int(config.resolution_minutes) != use_minutes:
        from dataclasses import replace

        config = replace(
            config,
            resolution_minutes=use_minutes,
            use_sub_hour_bars=use_minutes < 60,
        )
    panel = prepare_bar_panel(bars, config)
    if "BTCUSD" not in panel:
        raise ValueError("bars must include BTCUSD")
    min_bars = 400 if int(config.resolution_minutes) >= 60 else int(getattr(config, "walk_forward_min_bars", 1600))
    if len(ts) < min_bars:
        raise ValueError(f"need at least {min_bars} bar timestamps for walk-forward")

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
        from core import AlphaEnsemble
        from kraken_ml import MLScorer

        ens = AlphaEnsemble(cfg, MLScorer())
        fold_sharpes = []
        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_end = min(train_end + fold_size, len(ts) - 1)
            rets, _pnls, _tr = simulate_oos_fold(
                panel, train_end_idx=train_end, test_end_idx=test_end, timestamps=ts, config=cfg, ensemble=ens
            )
            fold_sharpes.append(_sharpe(rets, periods_per_year=periods_per_year(config)))
        mean_sh = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        if mean_sh > best_sharpe:
            best_sharpe = mean_sh
            best_combo = combo

    final_cfg = replace(config, **best_combo)
    from core import AlphaEnsemble
    from kraken_ml import MLScorer

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
            "sharpe": _sharpe(rets, periods_per_year=periods_per_year(config)),
            "trades": float(tr),
        })

    eq = np.array(equity_curve)
    ppy = periods_per_year(config)
    return WalkForwardResult(
        oos_sharpe=_sharpe(np.array(all_rets), periods_per_year=ppy),
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

# --- from backtest_validator.py ---


import pandas as pd

from config import CONFIG, KrakenMaxConfig


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    oos_sharpe: float
    oos_max_drawdown: float
    oos_total_return: float
    oos_trades: int
    oos_win_rate: float
    bar_minutes: int
    failures: tuple[str, ...]
    best_weights: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def validate_bars(
    bars: pd.DataFrame,
    *,
    config: KrakenMaxConfig = CONFIG,
    n_folds: int = 3,
    bar_minutes: int | None = None,
) -> ValidationReport:
    """Run walk-forward and check against v7 validation thresholds."""
    ts = sorted(set(pd.to_datetime(bars["timestamp"], utc=True)))
    inferred = infer_bar_minutes(ts)
    use_min = int(bar_minutes) if bar_minutes is not None else inferred
    result: WalkForwardResult = walk_forward_optimize(bars, n_folds=n_folds, config=config, bar_minutes=use_min)
    failures: list[str] = []
    if result.oos_sharpe < float(config.validation_min_sharpe):
        failures.append(f"sharpe {result.oos_sharpe:.3f} < {config.validation_min_sharpe}")
    if result.oos_max_drawdown < float(config.validation_max_drawdown):
        failures.append(f"max_dd {result.oos_max_drawdown:.3f} < {config.validation_max_drawdown}")
    if result.oos_trades < int(config.validation_min_trades):
        failures.append(f"trades {result.oos_trades} < {config.validation_min_trades}")
    if result.oos_win_rate < float(config.validation_min_win_rate):
        failures.append(f"win_rate {result.oos_win_rate:.3f} < {config.validation_min_win_rate}")
    return ValidationReport(
        passed=len(failures) == 0,
        oos_sharpe=float(result.oos_sharpe),
        oos_max_drawdown=float(result.oos_max_drawdown),
        oos_total_return=float(result.oos_total_return),
        oos_trades=int(result.oos_trades),
        oos_win_rate=float(result.oos_win_rate),
        bar_minutes=use_min,
        failures=tuple(failures),
        best_weights=dict(result.best_weights),
    )


def save_validation_report(report: ValidationReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path

# --- from regime_walk_forward.py ---


import numpy as np
import pandas as pd

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
    from core import AlphaEnsemble
    from kraken_ml import MLScorer

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


class AutoRevalidator:
    """Monthly walk-forward re-validation + baseline + regime weight refresh."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.baseline_mgr = BaselineManager(config)
        self._last_run = None

    def should_run(self, now) -> bool:
        if self._last_run is None:
            return True
        days = int(self.config.auto_revalidate_days)
        return (now - self._last_run) >= timedelta(days=days)

    def bars_from_dataframe(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        rename = {str(c): str(c).lower() for c in bars.columns}
        out = bars.rename(columns=rename)
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.sort_values(["symbol", "timestamp"])

    def run(
        self,
        bars: pd.DataFrame,
        algo=None,
        *,
        n_folds: int | None = None,
    ) -> dict[str, Any]:
        bars = self.bars_from_dataframe(bars)
        folds = int(n_folds or self.config.auto_revalidate_folds)
        validation: ValidationReport = validate_bars(bars, config=self.config, n_folds=folds)
        wf = walk_forward_optimize(bars, n_folds=folds, config=self.config)
        regime_w = optimize_regime_weights(bars, config=self.config)
        if algo is not None:
            self.baseline_mgr.apply_walk_forward_result(algo, wf, persist_ensemble=False)
            self._persist_object_store(algo, validation, wf, regime_w)
        self._last_run = getattr(algo, "Time", None) if algo is not None else self._last_run
        return {
            "validation_passed": validation.passed,
            "oos_sharpe": float(wf.oos_sharpe),
            "regime_weights": regime_w,
            "failures": list(validation.failures),
        }

    def _persist_object_store(
        self,
        algo,
        validation: ValidationReport,
        wf,
        regime_w: dict[str, dict[str, float]],
    ) -> None:
        if not hasattr(algo, "ObjectStore"):
            return
        try:
            key = str(self.config.revalidation_object_store_key)
            payload = {
                "validation": validation.to_dict(),
                "oos_sharpe": float(wf.oos_sharpe),
                "regimes": regime_w,
            }
            algo.ObjectStore.Save(key, json.dumps(payload, indent=2))
            rw_key = str(self.config.regime_weights_object_store_key)
            algo.ObjectStore.Save(rw_key, json.dumps({"regimes": regime_w}, indent=2))
        except Exception:
            return
