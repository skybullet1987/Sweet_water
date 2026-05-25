#!/usr/bin/env python3
"""
Offline helper: refit ml_weights.json from a CSV with columns:
time, symbol, forward_return_24h, mom_7d, mom_21d, ...

Uses numpy only (no sklearn) — suitable for local research before uploading to QC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def fit_logistic(df: pd.DataFrame, feature_cols: list[str], label: str, steps: int = 800, lr: float = 0.08):
    x = df[feature_cols].astype(float).values
    y = (df[label].astype(float).values > 0).astype(float)
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(steps):
        z = x @ w + b
        p = _sigmoid(z)
        grad = (p - y) / max(len(y), 1)
        w -= lr * (x.T @ grad)
        b -= lr * float(grad.sum())
    return b, {c: float(wi) for c, wi in zip(feature_cols, w)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "ml_weights.json")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    features = [
        "mom_7d",
        "mom_21d",
        "mom_accel",
        "breakout_strength",
        "volume_surge",
        "rsi_pullback",
        "trend_quality",
        "rv_21d_inv",
        "breadth",
        "btc_beta",
    ]
    features = [c for c in features if c in df.columns]
    if "forward_return_24h" not in df.columns:
        raise SystemExit("CSV must include forward_return_24h label column")
    bias, weights = fit_logistic(df, features, "forward_return_24h")
    blob = {"bias": bias, "weights": weights}
    args.out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
