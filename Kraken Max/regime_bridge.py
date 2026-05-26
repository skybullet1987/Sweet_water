from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

from advanced_regime import AdvancedRegimeEngine
from config import CONFIG, KrakenMaxConfig
from regime import RegimeState
from sentiment import SentimentSnapshot, adjust_deployment_cap

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
