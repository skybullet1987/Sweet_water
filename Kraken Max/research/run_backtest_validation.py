#!/usr/bin/env python3
"""Kraken Max v7 — walk-forward validation gate on OHLCV CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow import save_validation_report, validate_bars  # noqa: E402
from config import CONFIG  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Kraken Max backtest validation (v7)")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bar-minutes", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "validation_report.json")
    args = parser.parse_args()

    bars = pd.read_csv(args.csv)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    report = validate_bars(bars, config=CONFIG, n_folds=args.folds, bar_minutes=args.bar_minutes)
    save_validation_report(report, args.out)
    status = "PASS" if report.passed else "FAIL"
    print(f"{status} sharpe={report.oos_sharpe:.3f} dd={report.oos_max_drawdown:.3f} trades={report.oos_trades}")
    if report.failures:
        for f in report.failures:
            print(f"  - {f}")
    print(f"report -> {args.out}")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
