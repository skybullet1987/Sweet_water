from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd

from config.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    HAS_HMMLEARN = True
except Exception:  # pragma: no cover - environment dependent
    GaussianHMM = None
    HAS_HMMLEARN = False

try:
    from sklearn.mixture import GaussianMixture

    HAS_SKLEARN_GMM = True
except Exception:  # pragma: no cover
    GaussianMixture = None
    HAS_SKLEARN_GMM = False

RegimeName = Literal["risk_on", "risk_off", "chop"]


@dataclass
class _ModelState:
    model: object | None = None
    labels: dict[int, RegimeName] | None = None
    vol_high: float | None = None


class HMMRegime:
    """3-state Gaussian HMM with GaussianMixture fallback when hmmlearn is unavailable."""

    def __init__(self, config: StrategyConfig = DEFAULT_STRATEGY_CONFIG) -> None:
        self.config = config
        self._features: list[list[float]] = []
        self._btc_returns: list[float] = []
        self._state = _ModelState()
        self._last_fit_idx = -1
        self._current_state: RegimeName = "risk_on"
        self._current_probs: Dict[str, float] = {"risk_on": 1.0, "risk_off": 0.0, "chop": 0.0}
        self.fallback_mode = (not HAS_HMMLEARN) and HAS_SKLEARN_GMM

    def update(self, btc_log_return: float, btc_realized_vol: float, breadth: float) -> None:
        self._features.append([float(btc_log_return), float(btc_realized_vol), float(breadth)])
        self._btc_returns.append(float(btc_log_return))
        idx = len(self._features) - 1

        can_fit = len(self._features) >= self.config.hmm_train_window_bars
        due_refit = (self._last_fit_idx < 0) or ((idx - self._last_fit_idx) >= self.config.hmm_retrain_every_bars)
        if can_fit and due_refit:
            self._fit()
            self._last_fit_idx = idx

        if self._state.model is None or self._state.labels is None:
            self._current_state = "risk_on"
            self._current_probs = {"risk_on": 1.0, "risk_off": 0.0, "chop": 0.0}
            return

        x = np.asarray(self._features[-1:], dtype=float)
        labels = self._state.labels
        if HAS_HMMLEARN and isinstance(self._state.model, GaussianHMM):
            probs = self._state.model.predict_proba(x)[0]
        else:
            probs = self._state.model.predict_proba(x)[0]

        mapped = {"risk_on": 0.0, "risk_off": 0.0, "chop": 0.0}
        for state_idx, prob in enumerate(probs):
            mapped[labels[state_idx]] += float(prob)
        vol_high = self._state.vol_high
        if vol_high is not None and btc_realized_vol >= vol_high:
            if self.fallback_mode:
                mapped["risk_off"] = max(mapped["risk_off"], mapped["risk_on"])
                mapped["risk_on"] *= 0.2
            else:
                mapped["chop"] = max(mapped["chop"], mapped["risk_on"])
                mapped["risk_on"] *= 0.2
        if vol_high is not None and btc_realized_vol <= (1.1 * vol_high) and btc_log_return > 0:
            mapped["risk_on"] = max(mapped["risk_on"], mapped["risk_off"], mapped["chop"])
            mapped["risk_off"] *= 0.5
            mapped["chop"] *= 0.5
        self._current_probs = mapped
        self._current_state = max(mapped, key=mapped.get)  # type: ignore[arg-type]

    def _fit(self) -> None:
        window = self.config.hmm_train_window_bars
        x = np.asarray(self._features[-window:], dtype=float)
        y = np.asarray(self._btc_returns[-window:], dtype=float)

        if HAS_HMMLEARN:
            try:
                model = GaussianHMM(
                    n_components=3,
                    covariance_type="diag",
                    min_covar=1e-5,
                    n_iter=200,
                    random_state=7,
                )
                model.fit(x)
                states = model.predict(x)
                labels = self._label_states(states, y, 3)
                self._state = _ModelState(model=model, labels=labels, vol_high=float(np.quantile(x[:, 1], 0.67)))
                return
            except Exception:
                pass

        if HAS_SKLEARN_GMM:
            model = GaussianMixture(n_components=2, covariance_type="full", random_state=7)
            model.fit(x)
            states = model.predict(x)
            labels = self._label_states(states, y, 2)
            self._state = _ModelState(model=model, labels=labels, vol_high=float(np.quantile(x[:, 1], 0.67)))
            return

        self._state = _ModelState(model=None, labels=None)

    @staticmethod
    def _label_states(states: np.ndarray, btc_returns: np.ndarray, n_states: int) -> dict[int, RegimeName]:
        sharpe_by_state: dict[int, float] = {}
        for s in range(n_states):
            vals = btc_returns[states == s]
            if len(vals) < 2:
                sharpe_by_state[s] = -np.inf
                continue
            std = float(np.std(vals, ddof=1))
            sharpe_by_state[s] = (float(np.mean(vals)) / std) if std > 1e-12 else -np.inf

        ranked = sorted(sharpe_by_state.items(), key=lambda kv: kv[1])
        label_map: dict[int, RegimeName] = {}
        if n_states == 2:
            label_map[ranked[0][0]] = "risk_off"
            label_map[ranked[1][0]] = "risk_on"
        else:
            label_map[ranked[0][0]] = "risk_off"
            label_map[ranked[1][0]] = "chop"
            label_map[ranked[2][0]] = "risk_on"
        return label_map

    def current_state(self) -> RegimeName:
        return self._current_state

    def current_state_probs(self) -> Dict[str, float]:
        return dict(self._current_probs)


def build_hmm_features(
    btc_close: pd.Series,
    breadth: pd.Series,
    rv_window: int = 24,
) -> pd.DataFrame:
    log_ret = np.log(btc_close).diff().fillna(0.0)
    rv = log_ret.rolling(window=rv_window, min_periods=rv_window).std().fillna(0.0)
    return pd.DataFrame({"btc_log_return": log_ret, "btc_realized_vol": rv, "breadth": breadth.fillna(0.5)})
