from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from main import SweetWaterPhase1


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


def test_invalid_order_event_clears_submitted_and_sets_cooldown():
    sym = _Symbol("SOLUSD")
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    algo.Time = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    algo.reporter = type("Reporter", (), {"on_order_event": lambda *_args, **_kwargs: None})()
    algo._submitted_orders = {sym: {"order_id": 1}}
    algo._failed_escalations = {}
    algo.Debug = lambda *_args, **_kwargs: None

    event = type("OrderEvent", (), {"Status": "Invalid", "Symbol": sym})()
    algo.OnOrderEvent(event)

    assert sym not in algo._submitted_orders
    assert algo._failed_escalations[sym] == algo.Time
