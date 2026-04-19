from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from risk import DrawdownCircuitBreaker


class _Algo:
    def __init__(self, equity: float):
        self.Portfolio = type("Portfolio", (), {"TotalPortfolioValue": float(equity)})()
        self._debug = []
        self.Debug = lambda msg: self._debug.append(str(msg))


def test_breaker_resets_after_max_triggered_bars_without_recovery():
    algo = _Algo(100.0)
    breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.10, recovery_pct=0.50, max_triggered_bars=3)

    breaker.update(algo)
    assert not breaker.is_triggered()

    algo.Portfolio.TotalPortfolioValue = 85.0
    breaker.update(algo)
    assert breaker.is_triggered()
    assert breaker._bars_triggered == 0

    algo.Portfolio.TotalPortfolioValue = 84.0
    breaker.update(algo)
    assert breaker.is_triggered()
    assert breaker._bars_triggered == 1

    algo.Portfolio.TotalPortfolioValue = 83.0
    breaker.update(algo)
    assert breaker.is_triggered()
    assert breaker._bars_triggered == 2

    algo.Portfolio.TotalPortfolioValue = 82.0
    breaker.update(algo)
    assert not breaker.is_triggered()
    assert breaker._bars_triggered == 0
