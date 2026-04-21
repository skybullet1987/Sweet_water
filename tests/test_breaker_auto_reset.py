from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from risk import should_auto_reset_latches


def test_latched_flags_auto_reset_on_new_utc_day_when_kill_switch_not_active():
    assert should_auto_reset_latches(current_day=date(2025, 1, 16), last_reset_day=date(2025, 1, 15), kill_switch_active=False)
    assert not should_auto_reset_latches(current_day=date(2025, 1, 16), last_reset_day=date(2025, 1, 16), kill_switch_active=False)
    assert not should_auto_reset_latches(current_day=date(2025, 1, 16), last_reset_day=date(2025, 1, 15), kill_switch_active=True)

