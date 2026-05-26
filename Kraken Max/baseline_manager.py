from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import CONFIG, KrakenMaxConfig
from drift_monitor import DriftMonitor
from walk_forward_engine import WalkForwardResult, save_ensemble_weights


class BaselineManager:
    """Auto-refresh walk-forward baseline Sharpe into ObjectStore + drift monitor (v6)."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def apply_walk_forward_result(
        self,
        algo,
        result: WalkForwardResult,
        *,
        drift: DriftMonitor | None = None,
        persist_ensemble: bool = True,
    ) -> float:
        sharpe = float(result.oos_sharpe)
        dm = drift or getattr(algo, "drift_monitor", None)
        if dm is not None:
            dm.baseline_sharpe = sharpe
            dm.save_baseline_to_object_store(algo, sharpe)
        if persist_ensemble and bool(self.config.auto_save_ensemble_weights):
            root = Path(__file__).resolve().parent
            save_ensemble_weights(result, root / str(self.config.ensemble_weights_path))
        self._save_meta(algo, sharpe, source="walk_forward", extra={"weights": result.best_weights})
        return sharpe

    def refresh_from_sharpe(
        self,
        algo,
        sharpe: float,
        *,
        source: str = "ml_retrain",
        drift: DriftMonitor | None = None,
    ) -> None:
        dm = drift or getattr(algo, "drift_monitor", None)
        if dm is None:
            return
        if not bool(self.config.auto_refresh_baseline):
            return
        dm.baseline_sharpe = float(sharpe)
        dm.save_baseline_to_object_store(algo, float(sharpe))
        self._save_meta(algo, float(sharpe), source=source)

    def _save_meta(self, algo, sharpe: float, *, source: str, extra: dict[str, Any] | None = None) -> None:
        key = str(self.config.baseline_meta_object_store_key)
        payload = {
            "oos_sharpe": float(sharpe),
            "source": source,
            **(extra or {}),
        }
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps(payload, indent=2))
        except Exception:
            return
