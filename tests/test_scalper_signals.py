from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from scalper_signals import ret_pct, rsi_14, z_score_20h


def test_z_score_20h_flat_series_zero():
    assert z_score_20h([100.0] * 20, 100.0) == 0.0


def test_z_score_20h_known_std():
    closes = [100.0 + i for i in range(20)]
    z = z_score_20h(closes, closes[-1])
    assert z > 0.0
    assert abs(z - 1.6057930839841814) < 1e-12


def test_rsi_14_monotonic_increasing_is_100():
    closes = [100.0 + i for i in range(15)]
    assert rsi_14(closes) == 100.0


def test_rsi_14_flat_is_50():
    closes = [100.0] * 15
    assert rsi_14(closes) == 50.0


def test_ret_pct():
    assert abs(ret_pct(100.0, 110.0) - 0.10) < 1e-12
