"""Walk-forward fixture gates (same checks as .github/workflows/walk_forward.yml)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "qc_runtime"))

from config import CONFIG  # noqa: E402
from reporting import walk_forward_run  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "walk_forward_bars.csv"
BASELINE = REPO_ROOT / "tests" / "fixtures" / "walk_forward_baseline.json"


@pytest.fixture(scope="module")
def walk_forward_out() -> dict:
    return walk_forward_run(pd.read_csv(FIXTURE), CONFIG)


def test_walk_forward_fixture_gates(walk_forward_out: dict) -> None:
    out = walk_forward_out
    assert out["oos_sharpe"] >= 0.5, out
    assert out["oos_max_dd"] <= 0.27, out
    assert 30 <= out["oos_trade_count"] <= 300, out
    assert out["oos_avg_win_avg_loss"] >= 1.2, out
    assert out["oos_cancel_rate"] < 0.20, out
    exits = out["exit_tag_distribution"]
    assert exits.get("TP", 0) >= 5, exits
    assert exits.get("SL", 0) >= 3, exits
    assert exits.get("TimeStop", 0) < exits.get("TP", 0) + exits.get("SL", 0) + exits.get("Chandelier", 0), exits


def test_walk_forward_matches_baseline_within_tolerance(walk_forward_out: dict) -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    out = walk_forward_out
    tol = 0.05
    for key in ("oos_sharpe", "oos_max_dd", "oos_trade_count", "oos_avg_win_avg_loss", "oos_cancel_rate"):
        b = float(baseline[key])
        v = float(out[key])
        if b == 0:
            assert abs(v) <= tol, (key, v, b)
        else:
            assert abs(v - b) / abs(b) <= tol, (key, v, b)
