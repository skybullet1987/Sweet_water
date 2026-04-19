from __future__ import annotations

from collections import deque

import numpy as np

from config import CONFIG, StrategyConfig

try:
    from arch import arch_model  # type: ignore

    HAS_ARCH = True
except Exception:  # pragma: no cover
    arch_model = None
    HAS_ARCH = False


class Sizer:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.trade_outcomes: deque[float] = deque(maxlen=60)
        self._returns: list[float] = []
        self._fit_idx = -1
        self._fit = None

    def record_trade(self, pnl_fraction: float) -> None:
        self.trade_outcomes.append(float(pnl_fraction))

    def update_returns(self, ret: float) -> None:
        self._returns.append(float(ret))

    def _fractional_kelly(self) -> float:
        if len(self.trade_outcomes) < 10:
            p = self.config.default_win_rate
            r = self.config.default_win_loss_ratio
        else:
            wins = [x for x in self.trade_outcomes if x > 0]
            losses = [-x for x in self.trade_outcomes if x < 0]
            p = len(wins) / len(self.trade_outcomes)
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            r = (avg_win / avg_loss) if avg_loss > 1e-12 else self.config.default_win_loss_ratio
        raw = p - (1.0 - p) / max(r, 1e-9)
        return max(0.0, min(self.config.kelly_cap, raw * self.config.kelly_fraction))

    def _forecast_annual_vol(self) -> float:
        if len(self._returns) < 30:
            return self.config.target_annual_vol
        idx = len(self._returns) - 1
        due = self._fit_idx < 0 or (idx - self._fit_idx) >= self.config.garch_refit_every_bars
        if due and HAS_ARCH:
            try:
                series = np.asarray(self._returns[-1000:], dtype=float) * 100.0
                model = arch_model(series, vol="Garch", p=1, q=1, mean="Zero", dist="normal")
                self._fit = model.fit(disp="off")
                self._fit_idx = idx
            except Exception:
                self._fit = None
        if HAS_ARCH and self._fit is not None:
            forecast = self._fit.forecast(horizon=1, reindex=False)
            sigma = float(np.sqrt(forecast.variance.values[-1, 0]) / 100.0)
            return sigma * np.sqrt(24 * 365)
        rolling = np.asarray(self._returns[-60:], dtype=float)
        return float(np.std(rolling, ddof=1) * np.sqrt(24 * 365))

    def size_for_trade(self, symbol: str, score: float, current_portfolio_state: dict[str, float]) -> float:
        _ = symbol
        equity = max(float(current_portfolio_state.get("equity", 1.0)), 1.0)
        gross = max(float(current_portfolio_state.get("gross_exposure", 0.0)), 0.0)
        if abs(score) <= 0:
            return 0.0
        kelly = self._fractional_kelly()
        vol_forecast = max(self._forecast_annual_vol(), 1e-9)
        vol_scale = self.config.target_annual_vol / vol_forecast
        budget = max(0.0, 1.0 - gross)
        sized = min(self.config.kelly_cap, kelly * max(0.5, min(2.0, vol_scale)) * min(1.0, budget))
        _ = equity
        return max(0.0, min(self.config.kelly_cap, sized))


    def passes_cost_gate(self, symbol: str, score: float, notional: float, fee_model, is_limit: bool = True) -> bool:
        """Reject entries where estimated round-trip fees exceed edge floor (0.4 * |score|)."""
        _ = symbol
        if notional <= 0 or fee_model is None:
            return True
        estimate_fn = getattr(fee_model, 'estimate_round_trip_cost', None)
        if estimate_fn is None:
            return True
        est_cost = float(estimate_fn(symbol, notional, is_limit=is_limit))
        edge_floor = 0.4 * abs(float(score)) * abs(float(notional))
        return est_cost <= edge_floor
