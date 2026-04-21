from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from reporting import Reporter


def test_daily_report_includes_expectancy_metrics():
    reporter = Reporter()
    reporter.daily_trade_count = 3
    reporter.daily_wins = [0.02, 0.01]
    reporter.daily_losses = [-0.005]

    out = reporter.daily_report()

    assert out["daily_trade_count"] == 3.0
    assert out["win_rate"] == 2 / 3
    assert "avg_win_pct" in out
    assert "avg_loss_pct" in out
    assert "expectancy_pct" in out
