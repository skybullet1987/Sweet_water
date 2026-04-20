from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

from features import FeatureEngine
from scalper_signals import ret_pct, rsi_14, z_score_20h


def test_scalper_features_populate_from_25_bars():
    engine = FeatureEngine(signal_mode="cross_sectional_momentum")
    closes = [100.0 + i for i in range(25)]
    for i, close in enumerate(closes):
        engine.update(
            {
                "symbol": "ETHUSD",
                "open": close - 0.2,
                "high": close + 0.2,
                "low": close - 0.4,
                "close": close,
                "volume": 1000.0 + i,
            }
        )
    feats = engine.current_features("ETHUSD")
    assert abs(float(feats["z_20h"]) - z_score_20h(closes[-25:], closes[-1])) < 1e-12
    assert abs(float(feats["rsi_14"]) - rsi_14(closes[-25:])) < 1e-12
    assert abs(float(feats["ret_1h"]) - ret_pct(closes[-2], closes[-1])) < 1e-12
    assert abs(float(feats["ret_6h"]) - ret_pct(closes[-7], closes[-1])) < 1e-12
