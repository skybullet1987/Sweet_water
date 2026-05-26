#!/usr/bin/env python3
"""Walk-forward ensemble optimization — run locally or in QC Research."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from walk_forward_engine import save_ensemble_weights, walk_forward_optimize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="Hourly OHLCV CSV: symbol,timestamp,open,high,low,close,volume")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "ensemble_weights.json")
    args = parser.parse_args()

    bars = pd.read_csv(args.csv)
    result = walk_forward_optimize(bars, n_folds=args.folds)
    out = save_ensemble_weights(result, args.out)
    print(f"OOS Sharpe: {result.oos_sharpe:.3f}")
    print(f"OOS MaxDD:  {result.oos_max_drawdown:.2%}")
    print(f"OOS Return: {result.oos_total_return:.2%}")
    print(f"Best weights: {result.best_weights}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
