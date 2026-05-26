from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from kraken_ops import BaselineManager  # noqa: E402
from config import CONFIG  # noqa: E402
from data import CrossVenueLead  # noqa: E402
from kraken_ops import DriftMonitor  # noqa: E402
from execution import track_order_submit  # noqa: E402
from kraken_ops import FillTracker  # noqa: E402
from kraken_ops import TelemetryDashboard  # noqa: E402
from workflow import (  # noqa: E402
    periods_per_year,
    prepare_bar_panel,
    resample_bars_to_minutes,
    walk_forward_optimize,
)


class _FakeObjectStore:
    def __init__(self):
        self._data = {}

    def ContainsKey(self, key):
        return key in self._data

    def Read(self, key):
        return self._data[key]

    def Save(self, key, value):
        self._data[key] = value


class _FakeAlgo:
    def __init__(self):
        self.config = CONFIG
        self.ObjectStore = _FakeObjectStore()
        self.Portfolio = type("P", (), {"TotalPortfolioValue": 1500.0})()
        self.fill_tracker = FillTracker()
        self.drift_monitor = DriftMonitor(replace(CONFIG, baseline_sharpe=0.8))
        self.active_universe = ["BTCUSD", "ETHUSD"]
        self._erc_weights = {"BTCUSD": 0.6, "ETHUSD": 0.4}
        self.Time = datetime(2025, 6, 1, tzinfo=timezone.utc)


def test_v6_config_flags():
    assert CONFIG.enable_telemetry
    assert CONFIG.use_cross_venue_lead
    assert CONFIG.auto_refresh_baseline
    assert periods_per_year(CONFIG) == 24 * 365 * 4


def test_track_order_submit_wires_fill_tracker():
    algo = _FakeAlgo()
    ticket = type("T", (), {"OrderId": 42})()
    track_order_submit(algo, ticket, symbol="BTCUSD", qty=0.01, expected_price=50000.0)
    assert algo.fill_tracker.stats.limits_submitted == 1


def test_telemetry_persist_roundtrip():
    algo = _FakeAlgo()
    dash = TelemetryDashboard()
    snap = dash.build(algo, regime_name="bull", micro_regime="qc_bull", deployment_cap=0.9)
    dash.persist(algo, snap)
    loaded = dash.load_latest(algo)
    assert loaded["regime"] == "bull"
    assert loaded["equity"] == 1500.0


def test_cross_venue_lead_from_csv(tmp_path):
    rows = []
    ts = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    for sym in ("BTCUSDT", "ETHUSDT"):
        c = 100.0
        for t in ts:
            c *= 1.001
            rows.append({"symbol": sym, "timestamp": t.isoformat(), "close": c})
    csv = tmp_path / "lead.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    lead = CrossVenueLead(replace(CONFIG, cross_venue_lead_csv=str(csv)))
    assert lead.loaded
    boost = lead.score_adjustment("BTCUSD", ts[-1])
    assert -0.2 < boost < 0.2


def test_baseline_manager_refresh():
    algo = _FakeAlgo()
    mgr = BaselineManager()
    mgr.refresh_from_sharpe(algo, 1.25, drift=algo.drift_monitor)
    assert algo.drift_monitor.baseline_sharpe == 1.25
    blob = json.loads(algo.ObjectStore.Read(CONFIG.drift_object_store_key))
    assert blob["oos_sharpe"] == 1.25


def test_resample_and_15m_walk_forward_smoke():
    idx = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    rows = []
    rng = np.random.default_rng(7)
    for sym in ("BTCUSD", "ETHUSD"):
        c = 100.0
        for ts in idx:
            c *= 1 + rng.normal(0, 0.002)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": ts,
                    "open": c,
                    "high": c * 1.001,
                    "low": c * 0.999,
                    "close": c,
                    "volume": 1000.0,
                }
            )
    hourly = pd.DataFrame(rows)
    bars15 = resample_bars_to_minutes(hourly, bar_minutes=15)
    assert len(bars15) > len(hourly)
    panel = prepare_bar_panel(bars15, replace(CONFIG, resolution_minutes=15))
    assert "ret168" in panel["BTCUSD"].columns
    cfg = replace(CONFIG, resolution_minutes=15, walk_forward_min_bars=800)
    result = walk_forward_optimize(bars15, n_folds=2, config=cfg, bar_minutes=15)
    assert "w_momentum" in result.best_weights
