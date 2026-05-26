#!/usr/bin/env python3
"""Run walk-forward optimization on 15-minute bar CSV (v6)."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from workflow import save_ensemble_weights, walk_forward_optimize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kraken Max 15m walk-forward")
    parser.add_argument("--csv", type=Path, required=True, help="OHLCV CSV (15m or hourly with --resample)")
    parser.add_argument("--resample", action="store_true", help="Upsample hourly CSV to 15m")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "ensemble_weights.json")
    args = parser.parse_args()

    bars = pd.read_csv(args.csv)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    if args.resample:
        from workflow import resample_bars_to_minutes

        bars = resample_bars_to_minutes(bars, bar_minutes=15)

    cfg = replace(
        CONFIG,
        resolution_minutes=15,
        use_sub_hour_bars=True,
        walk_forward_min_bars=1600,
    )
    result = walk_forward_optimize(bars, n_folds=args.folds, config=cfg, bar_minutes=15)
    save_ensemble_weights(result, args.out)
    print(
        f"oos_sharpe={result.oos_sharpe:.3f} weights={result.best_weights} "
        f"trades={result.oos_trades} -> {args.out}"
    )


if __name__ == "__main__":
    main()
