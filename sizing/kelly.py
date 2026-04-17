from __future__ import annotations

from collections import deque


def fractional_kelly(win_prob: float, win_loss_ratio: float, fraction: float = 0.25, cap: float = 0.20) -> float:
    if win_loss_ratio <= 0:
        return 0.0
    kelly = win_prob - (1.0 - win_prob) / win_loss_ratio
    scaled = kelly * fraction
    return max(0.0, min(cap, scaled))


class RollingKellyEstimator:
    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.by_regime: dict[str, deque[float]] = {}
        self.global_outcomes: deque[float] = deque(maxlen=window)

    def record_trade(self, pnl_fraction: float, regime: str | None = None) -> None:
        self.global_outcomes.append(float(pnl_fraction))
        if regime:
            bucket = self.by_regime.setdefault(regime, deque(maxlen=self.window))
            bucket.append(float(pnl_fraction))

    def estimate(self, regime: str | None = None) -> tuple[float, float]:
        series = self.by_regime.get(regime or "", deque())
        data = list(series) if len(series) >= 20 else list(self.global_outcomes)
        if len(data) < 10:
            return 0.5, 1.5

        wins = [x for x in data if x > 0]
        losses = [-x for x in data if x < 0]
        win_prob = len(wins) / len(data)
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        ratio = (avg_win / avg_loss) if avg_loss > 1e-12 else 1.5
        return win_prob, max(0.1, ratio)
