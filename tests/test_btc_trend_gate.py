from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from regime import BTC_EMA_PERIOD_HOURS, RegimeEngine


def test_gates_pass_blocked_by_btc_ema30d_gate():
    engine = RegimeEngine()
    engine.vol_stress = 0.0
    engine._btc_below_ema = False
    engine._btc_above_ema30d = False
    assert engine.gates_pass(breadth=1.0) is False
    engine._btc_above_ema30d = True
    assert engine.gates_pass(breadth=1.0) is True


def test_btc_ema30d_defaults_false_until_warm_then_turns_true():
    engine = RegimeEngine()
    assert engine.btc_above_ema30d() is False
    for px in range(1, BTC_EMA_PERIOD_HOURS + 1):
        engine.update_btc_close(float(px))
    assert engine.btc_above_ema30d() is True
