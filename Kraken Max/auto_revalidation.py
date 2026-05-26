from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_validator import ValidationReport, save_validation_report, validate_bars
from baseline_manager import BaselineManager
from config import CONFIG, KrakenMaxConfig
from regime_walk_forward import optimize_regime_weights, save_regime_weights
from walk_forward_engine import walk_forward_optimize


class AutoRevalidator:
    """v8 monthly walk-forward re-validation + baseline + regime weight refresh."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.baseline_mgr = BaselineManager(config)
        self._last_run = None

    def should_run(self, now) -> bool:
        if self._last_run is None:
            return True
        days = int(self.config.auto_revalidate_days)
        return (now - self._last_run) >= timedelta(days=days)

    def bars_from_dataframe(self, bars: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
        rename = {str(c): str(c).lower() for c in bars.columns}
        out = bars.rename(columns=rename)
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        return out.sort_values(["symbol", "timestamp"])

    def run(
        self,
        bars: pd.DataFrame,
        algo=None,
        *,
        n_folds: int | None = None,
    ) -> dict[str, Any]:
        bars = self.bars_from_dataframe(bars)
        folds = int(n_folds or self.config.auto_revalidate_folds)
        validation: ValidationReport = validate_bars(bars, config=self.config, n_folds=folds)
        wf = walk_forward_optimize(bars, n_folds=folds, config=self.config)
        regime_w = optimize_regime_weights(bars, config=self.config)
        root = Path(__file__).resolve().parent
        if bool(self.config.auto_save_ensemble_weights):
            from walk_forward_engine import save_ensemble_weights

            save_ensemble_weights(wf, root / str(self.config.ensemble_weights_path))
        save_regime_weights(
            regime_w,
            root / str(self.config.regime_weights_path),
            metrics={"oos_sharpe": wf.oos_sharpe, "validation_passed": validation.passed},
        )
        save_validation_report(validation, root / str(self.config.validation_report_path))
        if algo is not None:
            self.baseline_mgr.apply_walk_forward_result(algo, wf, persist_ensemble=False)
            self._persist_object_store(algo, validation, wf, regime_w)
        self._last_run = getattr(algo, "Time", None) if algo is not None else self._last_run
        return {
            "validation_passed": validation.passed,
            "oos_sharpe": float(wf.oos_sharpe),
            "regime_weights": regime_w,
            "failures": list(validation.failures),
        }

    def _persist_object_store(
        self,
        algo,
        validation: ValidationReport,
        wf,
        regime_w: dict[str, dict[str, float]],
    ) -> None:
        if not hasattr(algo, "ObjectStore"):
            return
        try:
            key = str(self.config.revalidation_object_store_key)
            payload = {
                "validation": validation.to_dict(),
                "oos_sharpe": float(wf.oos_sharpe),
                "regimes": regime_w,
            }
            algo.ObjectStore.Save(key, json.dumps(payload, indent=2))
            rw_key = str(self.config.regime_weights_object_store_key)
            algo.ObjectStore.Save(rw_key, json.dumps({"regimes": regime_w}, indent=2))
        except Exception:
            return
