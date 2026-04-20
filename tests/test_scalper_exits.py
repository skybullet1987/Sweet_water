from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from config import StrategyConfig
from scalper import evaluate_exit


def _eval(*, current_price: float, z: float = -1.0, hours_held: float = 1.0, btc_ret_1h: float = 0.0):
    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    entry_time = now - timedelta(hours=hours_held)
    return evaluate_exit(
        symbol="ETHUSD",
        feats={"z_20h": z},
        entry_price=100.0,
        entry_time=entry_time,
        current_time=now,
        current_price=current_price,
        btc_ret_1h=btc_ret_1h,
        config=StrategyConfig(),
    )


def test_exit_none_above_hard_stop():
    assert _eval(current_price=99.0) == (False, "")


def test_exit_hard_stop():
    assert _eval(current_price=98.4) == (True, "SL")


def test_exit_time_stop():
    assert _eval(current_price=100.0, hours_held=7.0) == (True, "TimeStop")


def test_exit_meanrev():
    assert _eval(current_price=101.0, z=0.5) == (True, "MeanRev")


def test_exit_overshoot():
    assert _eval(current_price=101.0, z=1.5) == (True, "Overshoot")


def test_exit_btc_panic():
    assert _eval(current_price=100.0, z=-1.0, btc_ret_1h=-0.03) == (True, "BTCPanic")
