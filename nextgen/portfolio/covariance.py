from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, Mapping, Optional


class DiagonalCovarianceEstimator:
    """
    Diagonal covariance estimator for a multi-symbol portfolio.

    Uses per-symbol rolling realised return standard deviation as the
    volatility estimate.  The off-diagonal correlation terms are ignored
    (diagonal approximation), which is conservative and avoids requiring
    a long shared history for all symbol pairs.

    Portfolio volatility estimate
    ------------------------------
    Under the diagonal assumption:

        σ_portfolio ≈ Σ |w_i| · σ_i

    This over-estimates true vol when positions are partially negating each
    other, so it functions as a conservative (safety-first) cap.

    Parameters
    ----------
    vol_window : int
        Number of return observations used for the rolling std.
        Default: 20 (≈ 100 minutes at 5-minute bars).
    annualization_factor : int
        Multiplier to annualise per-bar standard deviation.
        Default: 105,120  (12 bars/h × 24 h × 365 d, i.e. 5-minute bars).
    """

    _BARS_PER_YEAR_5MIN = 12 * 24 * 365

    def __init__(
        self,
        vol_window: int = 20,
        annualization_factor: Optional[int] = None,
    ) -> None:
        self.vol_window = vol_window
        self.annualization_factor = (
            annualization_factor
            if annualization_factor is not None
            else self._BARS_PER_YEAR_5MIN
        )
        self._sqrt_ann = math.sqrt(self.annualization_factor)
        self._returns: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.vol_window)
        )

    # ── data ingestion ────────────────────────────────────────────────────

    def update(self, symbol: str, return_: float) -> None:
        """Append a single bar return for *symbol*."""
        self._returns[symbol].append(return_)

    def update_price(self, symbol: str, close: float, prev_close: float) -> None:
        """Convenience: compute return from consecutive closes and call update()."""
        if prev_close > 0:
            self.update(symbol, (close - prev_close) / prev_close)

    # ── query API ─────────────────────────────────────────────────────────

    def realized_vol(self, symbol: str) -> float:
        """
        Annualised realised volatility for *symbol*.

        Returns 0.0 if fewer than 2 observations are available.
        """
        rets = self._returns.get(symbol)
        if rets is None or len(rets) < 2:
            return 0.0
        n = len(rets)
        mean = sum(rets) / n
        variance = sum((r - mean) ** 2 for r in rets) / (n - 1)
        return math.sqrt(max(variance, 0.0)) * self._sqrt_ann

    def portfolio_vol_estimate(
        self,
        weights: Mapping[str, float],
        vol_floor: float = 0.05,
    ) -> float:
        """
        Conservative portfolio volatility estimate (diagonal approximation).

        Parameters
        ----------
        weights : symbol → weight (positive or negative)
        vol_floor : minimum per-asset annual vol used when history is sparse

        Returns
        -------
        Estimated portfolio annualised volatility (≥ 0).
        """
        total = 0.0
        for symbol, w in weights.items():
            vol = self.realized_vol(symbol)
            if vol < vol_floor:
                vol = vol_floor
            total += abs(w) * vol
        return total

    def known_symbols(self) -> list[str]:
        """Return symbols for which at least one return has been recorded."""
        return list(self._returns.keys())
