from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CONFIG, StrategyConfig

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    HAS_HMM = True
except Exception:  # pragma: no cover
    GaussianHMM = None
    HAS_HMM = False

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover
    GaussianMixture = None

HMM_COMPONENTS = 3
HMM_MAX_ITER = 200
HMM_RANDOM_STATE = 7
GMM_COMPONENTS = 2


@dataclass
class _ModelState:
    model: object | None = None
    labels: dict[int, str] | None = None


class RegimeEngine:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self._x: list[list[float]] = []
        self._ret: list[float] = []
        self._model = _ModelState()
        self._last_refit = -1
        self._state = "risk_on"
        self._probs = {"risk_on": 1.0, "risk_off": 0.0, "chop": 0.0}
        self.vol_stress = 0.0
        self._btc_below_ema = False

    def _label_states(self, states: np.ndarray, returns: np.ndarray, n_states: int) -> dict[int, str]:
        sharpe_by_state: dict[int, float] = {}
        for idx in range(n_states):
            vals = returns[states == idx]
            if len(vals) < 2:
                sharpe_by_state[idx] = -np.inf
                continue
            sigma = float(np.std(vals, ddof=1))
            sharpe_by_state[idx] = (float(np.mean(vals)) / sigma) if sigma > 1e-12 else -np.inf
        ranked = [k for k, _ in sorted(sharpe_by_state.items(), key=lambda kv: kv[1])]
        if n_states == 2:
            return {ranked[0]: "risk_off", ranked[1]: "risk_on"}
        return {ranked[0]: "risk_off", ranked[1]: "chop", ranked[2]: "risk_on"}

    def _fit(self) -> None:
        window = self.config.hmm_train_window_bars
        x = np.asarray(self._x[-window:], dtype=float)
        y = np.asarray(self._ret[-window:], dtype=float)
        if HAS_HMM:
            try:
                model = GaussianHMM(
                    n_components=HMM_COMPONENTS,
                    covariance_type="diag",
                    n_iter=HMM_MAX_ITER,
                    random_state=HMM_RANDOM_STATE,
                )
                model.fit(x)
                states = model.predict(x)
                self._model = _ModelState(model=model, labels=self._label_states(states, y, HMM_COMPONENTS))
                return
            except Exception:
                pass
        if GaussianMixture is not None:
            model = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="full", random_state=HMM_RANDOM_STATE)
            model.fit(x)
            states = model.predict(x)
            self._model = _ModelState(model=model, labels=self._label_states(states, y, GMM_COMPONENTS))
            return
        self._model = _ModelState(model=None, labels=None)

    def update(self, btc_return: float, btc_vol: float, breadth: float, btc_above_ema200: bool = True) -> None:
        self._x.append([float(btc_return), float(btc_vol), float(breadth)])
        self._ret.append(float(btc_return))
        self._btc_below_ema = not bool(btc_above_ema200)
        idx = len(self._x) - 1
        if len(self._x) >= 30:
            vols = np.asarray([row[1] for row in self._x[-30:]], dtype=float)
            mu = float(np.mean(vols))
            sigma = float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
            z = (float(btc_vol) - mu) / max(sigma, 1e-9)
            self.vol_stress = float(1.0 / (1.0 + np.exp(-z)))
        else:
            self.vol_stress = 0.0
        if len(self._x) >= self.config.hmm_train_window_bars:
            due = self._last_refit < 0 or (idx - self._last_refit) >= self.config.hmm_retrain_every_bars
            if due:
                self._fit()
                self._last_refit = idx
        if self._model.model is None or self._model.labels is None:
            self._state = "risk_on"
            self._probs = {"risk_on": 1.0, "risk_off": 0.0, "chop": 0.0}
            return
        latest = np.asarray(self._x[-1:], dtype=float)
        probs = self._model.model.predict_proba(latest)[0]
        mapped = {"risk_on": 0.0, "risk_off": 0.0, "chop": 0.0}
        for state_idx, prob in enumerate(probs):
            mapped[self._model.labels[state_idx]] += float(prob)
        self._probs = mapped
        self._state = max(mapped, key=mapped.get)

    def current_state(self) -> str:
        return self._state

    def current_state_probs(self) -> dict[str, float]:
        return dict(self._probs)

    def gates_pass(self, breadth: float) -> bool:
        if self.vol_stress > self.config.vol_stress_threshold:
            return False
        if breadth < self.config.breadth_threshold:
            return False
        if self._btc_below_ema:
            return False
        return True
