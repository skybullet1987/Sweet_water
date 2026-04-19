from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    start_year: int = 2025
    start_month: int = 1
    start_day: int = 1
    end_year: int = 2026
    end_month: int = 1
    end_day: int = 1
    starting_cash: float = 500.0
    bar_resolution: str = "Hour"
    warmup_bars: int = 200
    max_positions: int = 3
    score_threshold: float = 0.40
    universe_size: int = 8
    hmm_train_window_bars: int = 24 * 180
    hmm_retrain_every_bars: int = 24 * 30
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.20
    target_annual_vol: float = 0.15
    garch_refit_every_bars: int = 24 * 7
    tp_atr_mult: float = 2.0
    stop_atr_mult: float = 1.0
    time_stop_bars: int = 24
    drawdown_halt_pct: float = 0.10
    max_position_weight: float = 0.35
    correlation_throttle: float = 0.70
    default_win_rate: float = 0.50
    default_win_loss_ratio: float = 1.5
    monthly_rebalance_day: int = 1
    feature_min_bars: int = 60
    cross_section_window: int = 24
    regime_vol_window: int = 24
    stale_order_bars: int = 3
    fee_maker_ratio: float = 0.70


CONFIG = StrategyConfig()
