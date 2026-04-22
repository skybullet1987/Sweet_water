from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from sizing import Sizer


def test_kelly_estimate_returns_safe_cold_start_value():
    sizer = Sizer()
    assert sizer._kelly_estimate() == 0.05


def test_kelly_estimate_respects_kelly_cap_when_no_losses():
    sizer = Sizer(StrategyConfig(kelly_cap=0.02))
    for _ in range(20):
        sizer.record_trade(0.01)
    assert sizer._kelly_estimate() == 0.02


def test_kelly_estimate_caps_large_fraction():
    sizer = Sizer(StrategyConfig(kelly_cap=0.25))
    for _ in range(19):
        sizer.record_trade(0.02)
    sizer.record_trade(-0.001)
    assert sizer._kelly_estimate() == 0.25
