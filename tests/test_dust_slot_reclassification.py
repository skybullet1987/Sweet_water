from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from signals import dust_is_effectively_flat


def test_dust_slot_reclassified_as_flat_when_below_both_min_qty_and_notional():
    assert dust_is_effectively_flat(qty=0.00002, price=4.4, min_qty=0.9, min_notional=3.9) is True

