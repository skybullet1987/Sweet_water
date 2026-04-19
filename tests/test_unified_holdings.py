from __future__ import annotations

import sys
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


class _Holding:
    def __init__(self, qty: float):
        self.Quantity = float(qty)


def test_current_holdings_includes_dust():
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    btc = _Symbol("BTCUSD")
    eth = _Symbol("ETHUSD")
    xrp = _Symbol("XRPUSD")
    algo.Portfolio = {btc: _Holding(1.0), eth: _Holding(0.0000001), xrp: _Holding(0.0)}
    out = algo._current_holdings()
    assert btc in out
    assert eth in out
    assert xrp not in out
