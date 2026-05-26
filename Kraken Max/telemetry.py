from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import CONFIG, KrakenMaxConfig


@dataclass
class TelemetrySnapshot:
    """Paper/live dashboard payload (v6) — persisted to QC ObjectStore."""

    ts: str
    equity: float
    regime: str
    micro_regime: str
    deployment_cap: float
    fill_rate: float
    avg_slippage_bps: float
    live_sharpe: float
    baseline_sharpe: float
    drift_ratio: float
    cross_venue_boost: float
    active_universe: list[str] = field(default_factory=list)
    erc_weights: dict[str, float] = field(default_factory=dict)
    version: str = "v6"


class TelemetryDashboard:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def build(
        self,
        algo,
        *,
        regime_name: str = "",
        micro_regime: str = "",
        deployment_cap: float = 0.0,
        cross_venue_boost: float = 0.0,
    ) -> TelemetrySnapshot:
        equity = float(getattr(algo.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        fill_rate = 1.0
        slip_bps = 0.0
        ft = getattr(algo, "fill_tracker", None)
        if ft is not None:
            fill_rate = float(ft.stats.fill_rate)
            slip_bps = float(ft.stats.avg_slippage_bps)
        live_sh = 0.0
        base_sh = float(self.config.baseline_sharpe)
        drift_ratio = 0.0
        dm = getattr(algo, "drift_monitor", None)
        if dm is not None:
            snap = dm.evaluate()
            live_sh = float(snap.live_sharpe)
            base_sh = float(snap.baseline_sharpe)
            drift_ratio = float(snap.ratio)
        now = getattr(algo, "Time", datetime.now(timezone.utc))
        ts = now.isoformat() if hasattr(now, "isoformat") else str(now)
        universe = list(getattr(algo, "active_universe", []) or [])
        erc = {k: float(v) for k, v in (getattr(algo, "_erc_weights", {}) or {}).items()}
        return TelemetrySnapshot(
            ts=ts,
            equity=equity,
            regime=str(regime_name),
            micro_regime=str(micro_regime),
            deployment_cap=float(deployment_cap),
            fill_rate=fill_rate,
            avg_slippage_bps=slip_bps,
            live_sharpe=live_sh,
            baseline_sharpe=base_sh,
            drift_ratio=drift_ratio,
            cross_venue_boost=float(cross_venue_boost),
            active_universe=universe,
            erc_weights=erc,
        )

    def persist(self, algo, snapshot: TelemetrySnapshot) -> None:
        key = str(self.config.telemetry_object_store_key)
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps(asdict(snapshot), indent=2))
        except Exception:
            return

    def load_latest(self, algo) -> dict[str, Any]:
        key = str(self.config.telemetry_object_store_key)
        try:
            if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
                return json.loads(algo.ObjectStore.Read(key))
        except Exception:
            pass
        return {}

    def summary_line(self, snapshot: TelemetrySnapshot) -> str:
        return (
            f"KM_TELEM eq={snapshot.equity:.0f} reg={snapshot.regime} "
            f"fill={snapshot.fill_rate:.0%} slip={snapshot.avg_slippage_bps:.0f}bps "
            f"sharpe={snapshot.live_sharpe:.2f}/{snapshot.baseline_sharpe:.2f} "
            f"drift={snapshot.drift_ratio:.2f} xvenue={snapshot.cross_venue_boost:+.3f}"
        )
