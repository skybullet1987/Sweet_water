from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from main import SweetWaterPhase1


def test_dispersion_regime_flat_half_full():
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo._dispersion_history = deque([0.10 + 0.01 * i for i in range(40)], maxlen=60)
    algo._dispersion_history[-1] = 0.11
    assert algo._dispersion_regime() == "flat"
    algo._dispersion_history[-1] = 0.16
    assert algo._dispersion_regime() == "half"
    algo._dispersion_history[-1] = 0.35
    assert algo._dispersion_regime() == "full"
