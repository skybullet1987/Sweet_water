from __future__ import annotations

import numpy as np

from config.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig

try:
    from arch import arch_model  # type: ignore

    HAS_ARCH = True
except Exception:  # pragma: no cover - env dependent
    arch_model = None
    HAS_ARCH = False


class VolTargetScaler:
    def __init__(self, config: StrategyConfig = DEFAULT_STRATEGY_CONFIG) -> None:
        self.config = config
        self._returns: list[float] = []
        self._fit_idx = -1
        self._model_fit = None

    def update_returns(self, ret: float) -> None:
        self._returns.append(float(ret))

    def _refit_if_due(self) -> None:
        idx = len(self._returns) - 1
        if idx < 60:
            return
        due = (self._fit_idx < 0) or ((idx - self._fit_idx) >= self.config.garch_refit_every_bars)
        if not due:
            return
        if HAS_ARCH:
            series = np.asarray(self._returns[-1000:], dtype=float) * 100.0
            model = arch_model(series, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
            self._model_fit = model.fit(disp="off")
        self._fit_idx = idx

    def forecast_annual_vol(self) -> float:
        if len(self._returns) < 30:
            return np.sqrt(24 * 365) * np.std(self._returns, ddof=1) if len(self._returns) > 1 else 0.15
        self._refit_if_due()
        if HAS_ARCH and self._model_fit is not None:
            f = self._model_fit.forecast(horizon=1, reindex=False)
            sigma = float(np.sqrt(f.variance.values[-1, 0]) / 100.0)
            return sigma * np.sqrt(24 * 365)
        rolling = np.asarray(self._returns[-30:], dtype=float)
        return float(np.std(rolling, ddof=1) * np.sqrt(24 * 365))

    def scale(self) -> float:
        forecast = max(self.forecast_annual_vol(), 1e-9)
        raw = self.config.target_annual_vol / forecast
        return float(min(2.0, max(0.5, raw)))
