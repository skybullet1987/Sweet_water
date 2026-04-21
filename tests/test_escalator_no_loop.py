from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import execution as execution_module
from execution import escalate_stale_orders


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _Algo:
    def __init__(self, symbol):
        self.Time = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
        self.stale_order_bars = 3
        self._submitted_orders = {
            symbol: {"order_id": 1, "quantity": 0.04, "intent": "entry", "age_bars": 4},
        }
        self._failed_escalations = {}
        self.Transactions = type("T", (), {"CancelOrder": lambda *_args, **_kwargs: None})()
        self.Debug = lambda *_args, **_kwargs: None


def test_escalator_pops_on_failed_replacement(monkeypatch):
    sym = _Symbol("ETHUSD")
    algo = _Algo(sym)
    calls = []

    def _replacement(*_args, **_kwargs):
        calls.append("called")
        return None

    monkeypatch.setattr(execution_module, "place_limit_or_market", _replacement)

    out = escalate_stale_orders(algo)
    assert out == []
    assert sym not in algo._submitted_orders
    assert sym not in algo._failed_escalations
    assert len(calls) == 1

    out2 = escalate_stale_orders(algo)
    assert out2 == []
    assert len(calls) == 1
