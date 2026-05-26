from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from config import CONFIG, KrakenMaxConfig

_WEIGHT_KEYS = ("w_momentum", "w_breakout", "w_dip", "w_ml")
_DEFAULT_REGIME_MAP = {
    "bull": {"w_momentum": 0.40, "w_breakout": 0.30, "w_dip": 0.10, "w_ml": 0.20},
    "neutral": {"w_momentum": 0.35, "w_breakout": 0.25, "w_dip": 0.15, "w_ml": 0.25},
    "bear": {"w_momentum": 0.25, "w_breakout": 0.15, "w_dip": 0.20, "w_ml": 0.40},
    "chaos": {"w_momentum": 0.20, "w_breakout": 0.20, "w_dip": 0.10, "w_ml": 0.50},
}


def load_regime_weights(path: Path | None = None) -> dict[str, dict[str, float]]:
    target = path or (Path(__file__).resolve().parent / "regime_weights.json")
    if not target.exists():
        return {k: dict(v) for k, v in _DEFAULT_REGIME_MAP.items()}
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
        regimes = blob.get("regimes") or {}
        out: dict[str, dict[str, float]] = {}
        for name, weights in regimes.items():
            out[str(name)] = {k: float(weights[k]) for k in _WEIGHT_KEYS if k in weights}
        return out or {k: dict(v) for k, v in _DEFAULT_REGIME_MAP.items()}
    except Exception:
        return {k: dict(v) for k, v in _DEFAULT_REGIME_MAP.items()}


def normalize_regime_key(regime_name: str) -> str:
    key = str(regime_name or "neutral").lower()
    for token in ("bull", "bear", "neutral", "chaos"):
        if token in key:
            return token
    return "neutral"


def config_for_regime(base: KrakenMaxConfig, regime_name: str, regime_map: dict[str, dict[str, float]] | None = None) -> KrakenMaxConfig:
    if not bool(base.use_regime_ensembles):
        return base
    mapping = regime_map or load_regime_weights()
    key = normalize_regime_key(regime_name)
    weights = mapping.get(key) or mapping.get("neutral") or {}
    if not weights:
        return base
    kwargs = {k: float(v) for k, v in weights.items() if k in _WEIGHT_KEYS}
    return replace(base, **kwargs) if kwargs else base
