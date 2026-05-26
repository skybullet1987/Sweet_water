#!/usr/bin/env python3
"""Kraken Max v8 — optimize per-regime ensemble weights from OHLCV CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from workflow import optimize_regime_weights, save_regime_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "regime_weights.json")
    args = parser.parse_args()
    bars = pd.read_csv(args.csv)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    weights = optimize_regime_weights(bars, config=CONFIG)
    save_regime_weights(weights, args.out)
    for regime, w in weights.items():
        print(regime, w)


if __name__ == "__main__":
    main()
