from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from features import FeatureEngine


def test_mom_21d_skip_excludes_last_five_days_jump():
    engine = FeatureEngine(signal_mode="cross_sectional_momentum")
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    closes = [100.0 + i for i in range(75)] + [1000.0] * 6
    for i, px in enumerate(closes):
        t = t0 + timedelta(days=i)
        engine.update({"symbol": "BTCUSD", "open": px, "high": px, "low": px, "close": px, "volume": 1000.0, "time": t})
    feats = engine.current_features("BTCUSD")
    assert float(feats.get("mom_21d", 0.0)) >= 3.0
    assert float(feats.get("mom_21d_skip", 0.0)) < 1.0
