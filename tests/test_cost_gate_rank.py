from __future__ import annotations

import sys
from pathlib import Path

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from core import AggressiveSizer  # noqa: E402


def test_rank_mode_negative_score_passes_cost_gate():
    sizer = AggressiveSizer(CONFIG)
    assert sizer.passes_cost_gate(-0.12, 50.0, algo=None)


def test_calibrated_gate_rejects_zero_score():
    from kraken_ops import CalibratedCostModel

    model = CalibratedCostModel(CONFIG)
    assert not model.passes_edge_gate(0.0, 50.0, None)
    assert model.passes_edge_gate(0.0, 50.0, None, rank_relative=True)


def test_bph_hourly_is_one():
    assert CONFIG.bph() == 1
    assert CONFIG.min_feature_bars() == 48
