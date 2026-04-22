from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig


def test_strategy_defaults_momentum_low_frequency_profile():
    cfg = StrategyConfig()
    assert cfg.strategy_mode == "momentum"
    assert cfg.top_k == 3
    assert cfg.min_hold_hours == 168
    assert cfg.rebalance_cadence_hours == 168
    assert cfg.max_orders_per_day == 2
    assert cfg.edge_cost_multiplier == 3.0
    assert cfg.scalper_z_entry == -2.8
    assert cfg.scalper_breakout_z_entry == 3.0
    assert cfg.scalper_anti_churn_hours == 12.0
