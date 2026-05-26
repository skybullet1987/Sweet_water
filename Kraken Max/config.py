from __future__ import annotations

from dataclasses import dataclass, field

ENABLE_SHORTS = False


@dataclass(frozen=True)
class KrakenMaxConfig:
    """Aggressive Kraken spot config — v2 with execution, ML retrain, correlation, scalper."""

    start_year: int = 2022
    start_month: int = 1
    start_day: int = 1
    end_year: int = 2026
    end_month: int = 5
    end_day: int = 1
    starting_cash: float = 1000.0
    account_currency: str = "USD"
    bar_resolution: str = "Hour"
    warmup_bars: int = 24 * 14

    universe_size: int = 24
    top_k: int = 4
    max_positions: int = 4
    rebalance_hours: int = 12
    min_hold_hours: int = 6

    total_deployment_cap: float = 0.98
    max_position_pct: float = 0.45
    pyramid_add_pct: float = 0.15
    pyramid_min_unrealized_pct: float = 0.05
    target_annual_vol: float = 0.85
    kelly_cap: float = 0.55
    min_position_floor_usd: float = 25.0

    w_momentum: float = 0.35
    w_breakout: float = 0.25
    w_dip: float = 0.15
    w_ml: float = 0.25
    entry_score_threshold: float = 0.42
    replace_score_delta: float = 0.12

    btc_trend_ema: int = 100
    vol_stress_threshold: float = 1.05
    breadth_bull_threshold: float = 0.55
    bear_deployment_cap: float = 0.35
    bear_prefer: tuple[str, ...] = ("BTCUSD", "ETHUSD")

    hard_stop_pct: float = -0.08
    catastrophic_stop_pct: float = -0.12
    tp_atr_mult: float = 4.5
    sl_atr_mult: float = 1.2
    chandelier_atr_mult: float = 2.5
    activate_trail_above_pct: float = 0.10
    time_stop_hours: float = 72.0

    drawdown_halt_pct: float = -0.28
    drawdown_cooldown_hours: int = 24
    max_orders_per_day: int = 12
    post_breaker_cooldown_hours: int = 12

    expected_round_trip_fees: float = 0.0052
    edge_cost_multiplier: float = 1.8
    edge_scale: float = 0.04
    assumed_spread_bps: float = 18.0
    assumed_slippage_bps: float = 12.0
    min_rebalance_weight_delta: float = 0.02

    feature_min_bars: int = 48
    score_clip: float = 4.0
    enable_shorts: bool = ENABLE_SHORTS

    # --- v2: limit execution + participation ---
    use_limit_orders: bool = True
    max_participation_rate: float = 0.12
    limit_order_timeout_seconds: int = 45
    stale_price_minutes: int = 90
    cash_safety_factor: float = 0.97
    failed_esc_cooldown_hours: float = 6.0
    stale_order_bars: int = 3

    # --- v3: external data + walk-forward weights ---
    use_external_sentiment: bool = True
    use_qc_fear_greed_index: bool = True
    ensemble_weights_path: str = "ensemble_weights.json"

    # --- v2: correlation filter ---
    corr_lookback_hours: int = 24 * 7
    max_pairwise_corr: float = 0.82
    min_corr_samples: int = 48

    # --- v2: sentiment / dominance regime ---
    fg_extreme_fear: float = 0.25
    fg_extreme_greed: float = 0.78
    btc_dom_high: float = 0.58
    btc_dom_low: float = 0.38
    sentiment_greed_boost: float = 0.08
    sentiment_fear_cut: float = 0.20

    # --- v2: walk-forward ML retrain ---
    ml_retrain_days: int = 30
    ml_min_samples: int = 80
    ml_train_steps: int = 600
    ml_learning_rate: float = 0.06
    ml_object_store_key: str = "kraken_max_ml_weights.json"

    # --- v2: scalper sleeve (6h mean-reversion in range) ---
    enable_scalper: bool = True
    scalper_cadence_hours: int = 6
    scalper_max_positions: int = 2
    scalper_position_pct: float = 0.12
    scalper_z_entry: float = -2.2
    scalper_relaxed_z_entry: float = -1.5
    scalper_overshoot_z: float = 0.5
    scalper_meanrev_z: float = -0.3
    scalper_rsi_min: float = 18.0
    scalper_rsi_max: float = 42.0
    scalper_rsi_long_max: float = 32.0
    scalper_time_stop_hours: float = 6.0
    scalper_hard_stop_pct: float = -0.025
    scalper_tp_r: float = 1.8
    scalper_anti_churn_hours: float = 4.0
    scalper_btc_1h_floor: float = -0.02
    scalper_btc_6h_floor: float = -0.045
    scalper_adx_range_max: float = 22.0

    min_qty_fallback: dict[str, float] = field(default_factory=lambda: {
        "BTCUSD": 0.0001,
        "ETHUSD": 0.001,
        "SOLUSD": 0.05,
        "XRPUSD": 2.0,
        "ADAUSD": 10.0,
        "LINKUSD": 0.5,
        "DOTUSD": 1.0,
        "AVAXUSD": 0.2,
        "LTCUSD": 0.05,
        "BCHUSD": 0.01,
    })


CONFIG = KrakenMaxConfig()
