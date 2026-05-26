from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from config import KrakenMaxConfig, CONFIG
from ml_scorer import MLScorer
from regime_ensemble import config_for_regime, load_regime_weights, load_regime_weights_from_object_store

try:
    from regime_walk_forward import load_regime_weights_merged
except ImportError:
    load_regime_weights_merged = load_regime_weights  # type: ignore

ENSEMBLE_WEIGHTS_PATH = Path(__file__).resolve().parent / "ensemble_weights.json"


def load_optimized_ensemble_weights(path: Path | None = None) -> dict[str, float]:
    target = path or ENSEMBLE_WEIGHTS_PATH
    if not target.exists():
        return {}
    blob = json.loads(target.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in (blob.get("ensemble") or {}).items()}


class AlphaEnsemble:
    def __init__(
        self,
        config: KrakenMaxConfig = CONFIG,
        ml: MLScorer | None = None,
        *,
        regime_weights: dict[str, dict[str, float]] | None = None,
        algo=None,
    ) -> None:
        opt = load_optimized_ensemble_weights()
        if opt:
            allowed = {"w_momentum", "w_breakout", "w_dip", "w_ml"}
            kwargs = {k: v for k, v in opt.items() if k in allowed}
            config = replace(config, **kwargs) if kwargs else config
        self.config = config
        self.ml = ml or MLScorer()
        self._regime_weights: dict[str, dict[str, float]] = {}
        if bool(config.use_regime_ensembles):
            if regime_weights is not None:
                self._regime_weights = regime_weights
            elif algo is not None:
                stored = load_regime_weights_from_object_store(algo)
                self._regime_weights = stored or (
                    load_regime_weights_merged() if bool(config.use_regime_wf_weights) else load_regime_weights()
                )
            else:
                self._regime_weights = (
                    load_regime_weights_merged() if bool(config.use_regime_wf_weights) else load_regime_weights()
                )

    def score_symbol(
        self,
        features: dict[str, float],
        *,
        rank_mom_21: float = 0.5,
        rank_breakout: float = 0.5,
        breadth: float = 0.5,
        btc_beta: float = 0.0,
        regime_name: str = "neutral",
    ) -> dict[str, float]:
        if not features:
            return {"final": -1e9, "momentum": 0.0, "breakout": 0.0, "dip": 0.0, "ml": 0.0}

        rv = max(float(features.get("rv_21d", 0.05)), 1e-6)
        momentum = (
            0.45 * float(features.get("mom_21d", 0.0)) / rv
            + 0.35 * float(features.get("mom_accel", 0.0)) / rv
            + 0.20 * (rank_mom_21 - 0.5)
        )
        breakout = (
            0.60 * float(features.get("breakout_strength", 0.0)) * 8.0
            + 0.25 * max(0.0, float(features.get("volume_surge", 0.0)))
            + 0.15 * (rank_breakout - 0.5)
        )
        dip = float(features.get("rsi_pullback", 0.0)) * max(0.0, float(features.get("trend_quality", 0.0)) * 5.0)

        ml_ctx = {
            "breadth": breadth,
            "btc_beta": btc_beta,
        }
        ml_score = self.ml.score(features, ml_ctx)

        cfg = config_for_regime(self.config, regime_name, self._regime_weights)
        w_m = float(cfg.w_momentum)
        w_b = float(cfg.w_breakout)
        w_d = float(cfg.w_dip)
        w_ml = float(cfg.w_ml)
        final = w_m * momentum + w_b * breakout + w_d * dip + w_ml * ml_score
        clip = float(self.config.score_clip)
        final = max(-clip, min(clip, final))
        return {
            "final": final,
            "momentum": momentum,
            "breakout": breakout,
            "dip": dip,
            "ml": ml_score,
        }
