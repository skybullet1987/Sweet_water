from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from risk import PortfolioRisk  # noqa: E402


def test_drawdown_halt_requires_recovery_after_cooldown():
    pr = PortfolioRisk(CONFIG)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert pr.drawdown_halted(t0, -0.30) is True
    t1 = t0 + timedelta(hours=int(CONFIG.drawdown_cooldown_hours) + 1)
    assert pr.drawdown_halted(t1, -0.25) is True
    assert pr.drawdown_halted(t1, -0.10) is False
