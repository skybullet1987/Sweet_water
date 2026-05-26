from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from config import CONFIG, KrakenMaxConfig
from walk_forward_engine import WalkForwardResult, infer_bar_minutes, walk_forward_optimize


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
