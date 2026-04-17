from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    bar_resolution: str = "Hour"
    max_positions: int = 3
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.20
    target_annual_vol: float = 0.15
    tp_atr_mult: float = 2.0
    stop_atr_mult: float = 1.0
    time_stop_bars: int = 24
    score_threshold: float = 0.40
    universe_size: int = 8
    hmm_train_window_bars: int = 4320
    hmm_retrain_every_bars: int = 720
    garch_refit_every_bars: int = 168


DEFAULT_STRATEGY_CONFIG = StrategyConfig()
