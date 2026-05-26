"""Kraken Max — regime detection & ensemble weights (`regime.py`)."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from config import CONFIG, KrakenMaxConfig
from data import SentimentSnapshot, adjust_deployment_cap

# --- from regime.py ---


from config import KrakenMaxConfig, CONFIG


@dataclass(frozen=True)
class RegimeState:
    name: str
    deployment_cap: float
    allow_new_entries: bool
    prefer_symbols: tuple[str, ...]
    allow_scalper: bool = False
    micro_regime: str = "unknown"


class RegimeEngine:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def classify(
        self,
        *,
        btc_features: dict[str, float],
        breadth: float,
        median_rv: float,
        sentiment: SentimentSnapshot | None = None,
    ) -> RegimeState:
        btc_trend = float(btc_features.get("trend_quality", 0.0))
        btc_mom = float(btc_features.get("mom_21d", 0.0))
        vol_stress = median_rv >= float(self.config.vol_stress_threshold)
        fg = float(sentiment.fear_greed) if sentiment else 0.5

        if vol_stress and breadth < 0.35:
            return RegimeState("chaos", 0.0, False, (), allow_scalper=False, micro_regime="chaos")
        if btc_trend > 0 and btc_mom > 0 and breadth >= float(self.config.breadth_bull_threshold):
            cap = float(self.config.total_deployment_cap)
            if sentiment:
                cap = adjust_deployment_cap(cap, sentiment, "bull", self.config)
            return RegimeState("bull", cap, True, (), allow_scalper=False, micro_regime="trend")
        if btc_trend < 0 or btc_mom < -0.05:
            cap = float(self.config.bear_deployment_cap)
            if sentiment and fg < float(self.config.fg_extreme_fear):
                cap *= 0.85
            return RegimeState(
                "bear",
                cap,
                True,
                tuple(self.config.bear_prefer),
                allow_scalper=False,
                micro_regime="bear",
            )
        cap = 0.75
        if sentiment:
            cap = adjust_deployment_cap(cap, sentiment, "neutral", self.config)
        ranging = abs(btc_mom) < 0.03 and breadth > 0.4 and breadth < 0.65
        return RegimeState(
            "neutral",
            cap,
            True,
            (),
            allow_scalper=ranging or fg < 0.45,
            micro_regime="ranging" if ranging else "neutral",
        )

# --- from advanced_regime.py ---


import numpy as np

from config import CONFIG, KrakenMaxConfig


class HurstRegimeModel:
    def __init__(self, window: int = 5000) -> None:
        self.window = int(window)
        self._rets: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._last_close: dict[str, float] = {}
        self._h: dict[str, float] = {}

    @staticmethod
    def hurst_rs(log_returns: np.ndarray) -> float:
        r = np.asarray(log_returns, dtype=float)
        n = len(r)
        if n < 20:
            return 0.5
        mean = float(np.mean(r))
        dev = r - mean
        z = np.cumsum(dev)
        R = float(np.max(z) - np.min(z))
        S = float(np.std(r, ddof=1))
        if S <= 0 or R <= 0:
            return 0.5
        return max(0.0, min(1.0, math.log(R / S) / math.log(n)))

    def update(self, ticker: str, close: float) -> None:
        if close <= 0:
            return
        prev = self._last_close.get(ticker)
        self._last_close[ticker] = close
        if prev and prev > 0:
            self._rets[ticker].append(math.log(close / prev))
        if len(self._rets[ticker]) >= 500:
            self._h[ticker] = self.hurst_rs(np.array(self._rets[ticker], dtype=float))

    def hurst(self, ticker: str) -> float:
        return float(self._h.get(ticker, 0.5))

    def regime(self, ticker: str) -> str:
        h = self.hurst(ticker)
        if h > 0.55:
            return "trend"
        if h < 0.45:
            return "meanrev"
        return "random"


class VarianceRatioRegimeModel:
    TREND_THRESHOLD = 1.05
    MEANREV_THRESHOLD = 0.95

    def __init__(self, window: int = 5000, min_samples: int = 500) -> None:
        self.window = int(window)
        self.min_samples = int(min_samples)
        self._rets: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._last_close: dict[str, float] = {}
        self._vr: dict[str, float] = {}

    @staticmethod
    def _variance_ratio(log_returns: np.ndarray, k: int = 24) -> float:
        r = np.asarray(log_returns, dtype=float)
        n = len(r)
        if n < k + 2:
            return 1.0
        var1 = float(np.var(r, ddof=1))
        if var1 <= 0:
            return 1.0
        rolling = np.convolve(r, np.ones(k, dtype=float), mode="valid")
        vark = float(np.var(rolling, ddof=1)) if len(rolling) > 1 else var1
        return vark / (k * var1)

    def update(self, ticker: str, close: float) -> None:
        if close <= 0:
            return
        prev = self._last_close.get(ticker)
        self._last_close[ticker] = close
        if prev and prev > 0:
            self._rets[ticker].append(math.log(close / prev))
        if len(self._rets[ticker]) >= self.min_samples:
            arr = np.array(self._rets[ticker], dtype=float)
            self._vr[ticker] = 0.5 * (self._variance_ratio(arr, 6) + self._variance_ratio(arr, 24))

    def regime(self, ticker: str) -> str:
        vr = float(self._vr.get(ticker, 1.0))
        if vr > self.TREND_THRESHOLD:
            return "trend"
        if vr < self.MEANREV_THRESHOLD:
            return "meanrev"
        return "random"


class AdvancedRegimeEngine(RegimeEngine):
    """v4: Hurst + variance-ratio micro-regime layered on v3 macro regime."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        super().__init__(config)
        self.hurst = HurstRegimeModel()
        self.vr = VarianceRatioRegimeModel()
        self._btc_below_ema200 = False

    def update_btc_bar(self, close: float, ema200: float) -> None:
        if close > 0 and ema200 > 0:
            self._btc_below_ema200 = close < ema200
        self.hurst.update("BTCUSD", close)
        self.vr.update("BTCUSD", close)

    def classify_advanced(
        self,
        *,
        btc_features: dict[str, float],
        breadth: float,
        median_rv: float,
        sentiment: SentimentSnapshot | None = None,
    ) -> RegimeState:
        base = self.classify(
            btc_features=btc_features,
            breadth=breadth,
            median_rv=median_rv,
            sentiment=sentiment,
        )
        h_reg = self.hurst.regime("BTCUSD")
        vr_reg = self.vr.regime("BTCUSD")

        cap = base.deployment_cap
        allow_scalper = base.allow_scalper
        micro = base.micro_regime

        if h_reg == "meanrev" and vr_reg == "meanrev":
            allow_scalper = True
            micro = "meanrev"
            if base.name == "bull":
                cap *= 0.92
        elif h_reg == "trend" and vr_reg == "trend" and base.name == "neutral":
            cap = min(0.99, cap * 1.05)
            micro = "trend"
            allow_scalper = False
        elif self._btc_below_ema200 and base.name != "chaos":
            cap *= 0.88
            micro = "below_ema200"

        if sentiment and base.name == "bull":
            cap = adjust_deployment_cap(cap, sentiment, "bull", self.config)

        return RegimeState(
            name=base.name,
            deployment_cap=max(0.0, min(0.99, cap)),
            allow_new_entries=base.allow_new_entries,
            prefer_symbols=base.prefer_symbols,
            allow_scalper=allow_scalper,
            micro_regime=f"{micro}|H{h_reg}|VR{vr_reg}",
        )

# --- from regime_bridge.py ---


from config import CONFIG, KrakenMaxConfig

_QC_RUNTIME = Path(__file__).resolve().parents[1] / "qc_runtime"
_qc_regime = None


def _load_qc_regime_engine():
    global _qc_regime
    if _qc_regime is not None:
        return _qc_regime
    saved = sys.path[:]
    try:
        sys.path = [str(_QC_RUNTIME)] + [p for p in sys.path if "Kraken Max" not in p]
        spec = importlib.util.spec_from_file_location("qc_regime_mod", _QC_RUNTIME / "regime.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _qc_regime = mod
        return mod
    finally:
        sys.path = saved


@dataclass
class _KmRegimeAdapter:
    """Minimal adapter so qc_runtime.RegimeEngine reads KrakenMax thresholds."""

    vol_stress_threshold: float
    breadth_threshold: float
    chop_return_threshold: float


class UnifiedRegimeEngine(AdvancedRegimeEngine):
    """v5: AdvancedRegime + qc_runtime vol/breadth/EMA30d gates."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        super().__init__(config)
        self._qc = None
        if bool(config.use_qc_regime_gates):
            try:
                mod = _load_qc_regime_engine()
                adapter = _KmRegimeAdapter(
                    vol_stress_threshold=float(config.vol_stress_threshold),
                    breadth_threshold=float(config.breadth_threshold),
                    chop_return_threshold=float(config.chop_return_threshold),
                )
                self._qc = mod.RegimeEngine(adapter)
            except Exception:
                self._qc = None

    def update_market(
        self,
        *,
        btc_close: float,
        btc_return: float,
        btc_vol: float,
        breadth: float,
        btc_above_ema200: bool,
        ema200: float | None = None,
    ) -> None:
        ema_ref = float(ema200 if ema200 is not None else btc_close)
        self.update_btc_bar(btc_close, ema_ref)
        if self._qc is not None:
            self._qc.update_btc_close(btc_close)
            self._qc.update(btc_return, btc_vol, breadth, btc_above_ema200=btc_above_ema200)

    def classify_unified(
        self,
        *,
        btc_features: dict[str, float],
        breadth: float,
        median_rv: float,
        sentiment: SentimentSnapshot | None = None,
        btc_return: float = 0.0,
        btc_vol: float = 0.0,
        btc_above_ema200: bool = True,
    ) -> RegimeState:
        reg = self.classify_advanced(
            btc_features=btc_features,
            breadth=breadth,
            median_rv=median_rv,
            sentiment=sentiment,
        )
        if self._qc is None:
            return reg
        if not self._qc.gates_pass(breadth):
            return RegimeState(
                "qc_risk_off",
                0.0,
                False,
                tuple(self.config.bear_prefer),
                allow_scalper=False,
                micro_regime=f"qc_{self._qc.current_state()}|{reg.micro_regime}",
            )
        qc_state = self._qc.current_state()
        cap = reg.deployment_cap
        if qc_state == "chop":
            cap *= 0.85
        elif qc_state == "risk_off":
            cap = min(cap, float(self.config.bear_deployment_cap))
        if sentiment:
            cap = adjust_deployment_cap(cap, sentiment, reg.name, self.config)
        return RegimeState(
            reg.name,
            max(0.0, min(0.99, cap)),
            reg.allow_new_entries,
            reg.prefer_symbols,
            allow_scalper=reg.allow_scalper and qc_state != "risk_off",
            micro_regime=f"qc_{qc_state}|{reg.micro_regime}",
        )

# --- from regime_ensemble.py ---


from config import CONFIG, KrakenMaxConfig

_WEIGHT_KEYS = ("w_momentum", "w_breakout", "w_dip", "w_ml")
_DEFAULT_REGIME_MAP = {
    "bull": {"w_momentum": 0.40, "w_breakout": 0.30, "w_dip": 0.10, "w_ml": 0.20},
    "neutral": {"w_momentum": 0.35, "w_breakout": 0.25, "w_dip": 0.15, "w_ml": 0.25},
    "bear": {"w_momentum": 0.25, "w_breakout": 0.15, "w_dip": 0.20, "w_ml": 0.40},
    "chaos": {"w_momentum": 0.20, "w_breakout": 0.20, "w_dip": 0.10, "w_ml": 0.50},
}


def load_regime_weights_from_object_store(algo, key: str | None = None) -> dict[str, dict[str, float]] | None:
    store_key = key or str(CONFIG.regime_weights_object_store_key)
    try:
        if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(store_key):
            blob = json.loads(algo.ObjectStore.Read(store_key))
            regimes = blob.get("regimes") or {}
            out: dict[str, dict[str, float]] = {}
            for name, weights in regimes.items():
                rk = normalize_regime_key(name)
                out[rk] = {k: float(weights[k]) for k in _WEIGHT_KEYS if k in weights}
            return out if out else None
    except Exception:
        return None
    return None


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

def load_regime_weights_merged(path: Path | None = None) -> dict[str, dict[str, float]]:
    base = load_regime_weights(path)
    target = path or (Path(__file__).resolve().parent / str(CONFIG.regime_weights_path))
    if not target.exists():
        return base
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
        if blob.get("source", "").startswith("regime_walk_forward"):
            regimes = blob.get("regimes") or {}
            for name, w in regimes.items():
                key = normalize_regime_key(name)
                base[key] = {k: float(w[k]) for k in _WEIGHT_KEYS if k in w}
    except Exception:
        pass
    return base
