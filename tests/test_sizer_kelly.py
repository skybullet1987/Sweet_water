from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from sizing import Sizer


def test_kelly_estimate_returns_safe_cold_start_value():
    sizer = Sizer()
    assert sizer._kelly_estimate() == 0.05
