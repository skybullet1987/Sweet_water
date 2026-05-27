"""
Built-in strategy parameters — single source of truth for QuantConnect cloud.

Do not rely on ``.json`` project files on QC; edit this module to change defaults.
Research scripts may still export JSON locally, but the live algorithm never reads them.
"""
from __future__ import annotations

from typing import Any

ML_FEATURE_NAMES: tuple[str, ...] = (
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
)

ML_WEIGHTS: dict[str, Any] = {
    "bias": 0.0,
    "weights": {name: 0.0 for name in ML_FEATURE_NAMES},
}

ENSEMBLE_WEIGHTS: dict[str, float] = {
    "w_momentum": 0.35,
    "w_breakout": 0.25,
    "w_dip": 0.15,
    "w_ml": 0.25,
}

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "bull": {"w_momentum": 0.40, "w_breakout": 0.30, "w_dip": 0.10, "w_ml": 0.20},
    "neutral": {"w_momentum": 0.35, "w_breakout": 0.25, "w_dip": 0.15, "w_ml": 0.25},
    "bear": {"w_momentum": 0.25, "w_breakout": 0.15, "w_dip": 0.20, "w_ml": 0.40},
    "chaos": {"w_momentum": 0.20, "w_breakout": 0.20, "w_dip": 0.10, "w_ml": 0.50},
}
