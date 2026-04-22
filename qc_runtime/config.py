from __future__ import annotations

from dataclasses import dataclass, field

ENABLE_SHORTS = False  # Cash-account / Canada compliance — no shorting.


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
    warmup_bars: int = 24
    universe_size: int = 40
    top_k: int = 3
    signal_mode: str = "cross_sectional_momentum"
    strategy_mode: str = "momentum"  # "scalper" | "momentum"

    # Sizing
    target_annual_vol: float = 0.30
    kelly_cap: float = 0.25
    min_position_floor_usd: float = 5.0
    max_position_pct: float = 0.20
    total_deployment_cap: float = 0.85
    max_positions: int = 10
    post_breaker_cooldown_hours: int = 48

    # Costs
    expected_round_trip_fees: float = 0.0065
    cost_gate_multiplier: float = 2.5
    # cost_gate_multiplier is retained for legacy paths; edge_cost_multiplier is used by the new cross-sectional momentum gate.
    edge_cost_multiplier: float = 3.0
    edge_scale: float = 0.025
    assumed_spread_bps: float = 12.0
    assumed_slippage_bps: float = 8.0
    min_rebalance_weight_delta: float = 0.01

    # Exits (ATR-scaled)
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    # Tier-7 names used by manage_open_positions; legacy *_mult fields are retained for backward compatibility.
    tp_atr_multiplier: float = 3.0
    sl_atr_multiplier: float = 1.5
    chandelier_atr_mult: float = 3.0
    activate_trailing_above_pct: float = 0.005
    time_stop_bars: int = 24
    time_stop_hours: float = 120.0

    # Regime gates
    vol_stress_threshold: float = 0.85
    breadth_threshold: float = 0.30
    btc_trend_ema: int = 200
    chop_return_threshold: float = 0.002

    # Scoring
    score_threshold: float = 0.0
    chop_threshold_multiplier: float = 1.5
    cross_section_weight: float = 0.40
    micro_entry_threshold: float = 0.45
    micro_flatten_threshold: float = 0.06
    max_orders_per_day: int = 2
    min_hold_hours: int = 168
    rebalance_cadence_hours: int = 168
    max_replacements_per_rebalance: int = 4
    sig_hold_log_every_bars: int = 24
    # Legacy knobs still consumed in risk/tests
    hmm_train_window_bars: int = 24 * 120
    hmm_retrain_every_bars: int = 24 * 30
    drawdown_halt_pct: float = 0.10
    max_position_weight: float = 0.35
    correlation_throttle: float = 0.70
    default_win_rate: float = 0.50
    default_win_loss_ratio: float = 1.5
    garch_refit_every_bars: int = 24 * 7
    feature_min_bars: int = 60
    regime_vol_window: int = 24
    stale_order_bars: int = 3
    stale_price_minutes: int = 90
    failed_esc_cooldown_hours: float = 6.0
    cash_safety_factor: float = 0.97
    rebalance_invalid_retry_cap: int = 3
    min_notional_usd: float = 5.0
    score_clip_value: float = 3.0
    min_rv_floor: float = 1e-4
    score_mom21_weight: float = 0.6
    score_mom63_weight: float = 0.4
    score_dd_penalty: float = 0.3

    # Scalper signal thresholds
    scalper_z_entry: float = -2.8
    scalper_meanrev_z: float = 0.0
    scalper_overshoot_z: float = 1.0
    scalper_rsi_min: float = 20.0
    scalper_rsi_max: float = 40.0
    scalper_rv_max: float = 1.5

    # BTC guardrails
    scalper_btc_1h_floor: float = -0.015
    scalper_btc_6h_floor: float = -0.04
    scalper_btc_panic_threshold: float = -0.025
    scalper_use_btc_ema_gate: bool = False
    scalper_adx_range_max: float = 20.0
    scalper_adx_trend_min: float = 30.0
    scalper_breakout_z_entry: float = 3.0
    scalper_breakout_volume_mult: float = 1.2

    # Position management
    scalper_position_size_pct: float = 0.2375
    scalper_max_concurrent: int = 4
    scalper_hard_sl_pct: float = -0.015
    scalper_time_stop_hours: float = 6.0
    scalper_anti_churn_hours: float = 12.0
    scalper_consecutive_loss_brake: int = 3
    scalper_daily_loss_brake: float = -0.01
    scalper_universe_size: int = 20
    scalper_loss_cooldown_after: int = 1
    scalper_loss_cooldown_hours: float = 6.0
    scalper_risk_per_trade_pct: float = 0.0075
    scalper_max_symbol_exposure_pct: float = 0.25
    scalper_max_gross_exposure_pct: float = 0.95
    scalper_corr_threshold: float = 0.85
    scalper_corr_block_count: int = 1
    scalper_max_hold_hours: float = 36.0
    scalper_stuck_hold_bars: int = 24
    scalper_meanrev_rsi_confirm_max: float = 35.0
    scalper_meanrev_rsi_long_max: float = 30.0
    scalper_meanrev_rsi_short_min: float = 70.0
    scalper_stop_atr_mult: float = 1.5
    scalper_chandelier_atr_mult: float = 2.0
    scalper_tight_trail_atr_mult: float = 0.5
    scalper_tp1_r: float = 1.0
    scalper_tp1_partial_pct: float = 0.5
    scalper_tp2_r: float = 2.5
    scalper_mr_max_hold_bars: int = 8
    scalper_mom_max_hold_bars: int = 16
    scalper_max_bars_held: int = 24
    scalper_stale_position_bars: int = 24
    scalper_breaker_symbol_loss_pct: float = 0.005
    scalper_beta_cap: float = 2.25
    scalper_beta_lookback_hours: int = 24 * 7
    scalper_tp1_atr: float = 1.5
    scalper_hard_stop_r_multiple: float = -1.5
    scalper_catastrophic_stop_pct: float = -0.03
    scalper_mr_relaxed_z_entry: float = -1.6
    scalper_mr_relaxed_vol_cone_max_pct: float = 0.40
    enable_shorts: bool = ENABLE_SHORTS
    scalper_momentum_short_z_entry: float = -2.0

    min_qty_fallback: dict[str, float] = field(default_factory=lambda: {
        'AXSUSD': 5.0, 'SANDUSD': 10.0, 'MANAUSD': 10.0, 'ADAUSD': 10.0,
        'MATICUSD': 10.0, 'DOTUSD': 1.0, 'LINKUSD': 0.5, 'AVAXUSD': 0.2,
        'ATOMUSD': 0.5, 'NEARUSD': 1.0, 'SOLUSD': 0.05,
        'ALGOUSD': 10.0, 'XLMUSD': 30.0, 'TRXUSD': 50.0, 'ENJUSD': 10.0,
        'BATUSD': 10.0, 'CRVUSD': 5.0, 'SNXUSD': 3.0, 'COMPUSD': 0.1,
        'AAVEUSD': 0.05, 'MKRUSD': 0.01, 'YFIUSD': 0.001, 'UNIUSD': 1.0,
        'SUSHIUSD': 5.0, '1INCHUSD': 5.0, 'GRTUSD': 10.0, 'FTMUSD': 10.0,
        'IMXUSD': 5.0, 'APEUSD': 2.0, 'GMTUSD': 10.0, 'OPUSD': 5.0,
        'LDOUSD': 5.0, 'ARBUSD': 5.0, 'LPTUSD': 5.0, 'KTAUSD': 10.0,
        'GUNUSD': 50.0, 'BANANAS31USD': 500.0, 'CHILLHOUSEUSD': 500.0,
        'PHAUSD': 50.0, 'MUSD': 50.0, 'ICNTUSD': 50.0,
        'SHIBUSD': 50000.0, 'XRPUSD': 2.0,
    })
    min_notional_fallback: dict[str, float] = field(default_factory=lambda: {
        'EWTUSD': 2.0, 'SANDUSD': 8.0, 'CTSIUSD': 18.0, 'MKRUSD': 0.01,
        'AUDUSD': 10.0, 'LPTUSD': 0.3, 'OXTUSD': 40.0, 'ENJUSD': 15.0,
        'UNIUSD': 0.5, 'LSKUSD': 3.0, 'BCHUSD': 1.0,
    })


CONFIG = StrategyConfig()
