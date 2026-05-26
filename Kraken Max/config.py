from __future__ import annotations

from dataclasses import dataclass, field

ENABLE_SHORTS = False


def bars_per_hour(resolution_minutes: int) -> int:
    return max(1, 60 // max(1, int(resolution_minutes)))


@dataclass(frozen=True)
class KrakenMaxConfig:
    """Kraken Max config (v8). QC limit: keep each .py module under ~63,000 characters."""

    start_year: int = 2022
    start_month: int = 1
    start_day: int = 1
    end_year: int = 2026
    end_month: int = 5
    end_day: int = 1
    starting_cash: float = 1000.0
    account_currency: str = "USD"
    bar_resolution: str = "Hour"
    resolution_minutes: int = 15
    use_sub_hour_bars: bool = False  # False=hourly (first QC backtest). True needs Kraken Minute data.
    warmup_bars: int = 24 * 7  # ~7 days hourly (14d+50 symbols can look "stuck" on QC)
    warmup_bars_sub_hour: int = 24 * 7 * 4  # ~7 days of 15m bars when use_sub_hour_bars=True
    seed_subscribe_symbols: tuple[str, ...] = (
        "BTCUSD",
        "ETHUSD",
        "LTCUSD",
        "BCHUSD",
        "SOLUSD",
        "XRPUSD",
        "LINKUSD",
        "ADAUSD",
    )
    subscribe_all_universe_on_init: bool = False  # True=subscribes ~50 pairs (slow QC warmup)

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
    breadth_threshold: float = 0.30
    bear_deployment_cap: float = 0.35
    bear_prefer: tuple[str, ...] = ("BTCUSD", "ETHUSD")
    chop_return_threshold: float = 0.002

    hard_stop_pct: float = -0.08
    catastrophic_stop_pct: float = -0.12
    tp_atr_mult: float = 4.5
    sl_atr_mult: float = 1.2
    chandelier_atr_mult: float = 2.5
    activate_trail_above_pct: float = 0.10
    time_stop_hours: float = 72.0

    drawdown_halt_pct: float = -0.28
    drawdown_cooldown_hours: int = 24
    max_orders_per_day: int = 16
    post_breaker_cooldown_hours: int = 12

    expected_round_trip_fees: float = 0.0052
    edge_cost_multiplier: float = 1.8
    edge_scale: float = 0.04
    assumed_spread_bps: float = 18.0
    assumed_slippage_bps: float = 12.0
    min_rebalance_weight_delta: float = 0.02

    feature_min_bars: int = 48 * 4
    score_clip: float = 4.0
    enable_shorts: bool = ENABLE_SHORTS

    use_limit_orders: bool = True
    max_participation_rate: float = 0.12
    limit_order_timeout_seconds: int = 45
    stale_price_minutes: int = 90
    cash_safety_factor: float = 0.97
    failed_esc_cooldown_hours: float = 6.0
    stale_order_bars: int = 3

    use_external_sentiment: bool = True
    use_qc_fear_greed_index: bool = True
    ensemble_weights_path: str = "ensemble_weights.json"

    enable_brackets: bool = True
    use_erc_sizing: bool = True
    use_advanced_regime: bool = True
    use_qc_regime_gates: bool = True
    enable_live_alerts: bool = True
    alert_on_drawdown_halt: bool = True
    alert_on_rebalance: bool = False
    alert_on_drift: bool = True
    funding_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

    corr_lookback_hours: int = 24 * 7
    max_pairwise_corr: float = 0.82
    min_corr_samples: int = 48
    erc_shrinkage: float = 0.35
    erc_turnover_penalty: float = 0.20

    fg_extreme_fear: float = 0.25
    fg_extreme_greed: float = 0.78
    btc_dom_high: float = 0.58
    btc_dom_low: float = 0.38
    sentiment_greed_boost: float = 0.08
    sentiment_fear_cut: float = 0.20

    ml_retrain_days: int = 30
    ml_min_samples: int = 80
    ml_train_steps: int = 600
    ml_learning_rate: float = 0.06
    ml_object_store_key: str = "kraken_max_ml_weights.json"

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

    enable_fill_tracking: bool = True
    fill_rate_alert_threshold: float = 0.55
    slippage_alert_bps: float = 35.0

    enable_drift_monitor: bool = True
    drift_window_hours: int = 24 * 30
    baseline_sharpe: float = 0.50
    drift_sharpe_ratio_threshold: float = 0.50
    drift_object_store_key: str = "kraken_max_baseline_sharpe.json"
    baseline_meta_object_store_key: str = "kraken_max_baseline_meta.json"
    auto_refresh_baseline: bool = True
    auto_save_ensemble_weights: bool = False

    enable_telemetry: bool = True
    telemetry_object_store_key: str = "kraken_max_telemetry.json"
    telemetry_cadence_hours: int = 1

    use_cross_venue_lead: bool = True
    cross_venue_lead_csv: str = "data/binance_spot_lead.csv"
    cross_venue_boost_per_z: float = 0.04
    cross_venue_max_boost: float = 0.12
    cross_venue_z_window: int = 96

    walk_forward_bar_minutes: int = 15
    walk_forward_min_bars: int = 1600

    use_regime_ensembles: bool = True
    regime_weights_path: str = "regime_weights.json"
    enable_cluster_risk: bool = True
    max_positions_per_cluster: int = 2

    use_calibrated_costs: bool = True
    cost_calibration_min_fills: int = 5

    enable_scorecard: bool = True
    scorecard_object_store_key: str = "kraken_max_scorecard.json"
    scorecard_max_points: int = 24 * 90 * 4
    scorecard_max_trades: int = 500
    paper_min_days: float = 30.0
    paper_min_sharpe: float = 0.35
    paper_max_drawdown: float = -0.35
    paper_min_trades: int = 12
    alert_on_paper_gate_fail: bool = True

    validation_min_sharpe: float = 0.15
    validation_max_drawdown: float = -0.45
    validation_min_trades: int = 8
    validation_min_win_rate: float = 0.38
    validation_report_path: str = "validation_report.json"

    use_regime_wf_weights: bool = True
    regime_wf_min_bars: int = 80
    regime_weights_object_store_key: str = "kraken_max_regime_weights.json"

    enable_auto_revalidation: bool = False  # True runs heavy walk-forward on MonthStart (can freeze backtests)
    auto_revalidate_days: int = 30
    auto_revalidate_folds: int = 3
    auto_revalidate_lookback_days: int = 120
    revalidation_object_store_key: str = "kraken_max_revalidation.json"

    enable_dashboard_digest: bool = True
    dashboard_digest_hour: int = 8
    dashboard_text_key: str = "kraken_max_dashboard.txt"
    dashboard_html_key: str = "kraken_max_dashboard.html"
    alert_on_dashboard: bool = True

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

    def bph(self) -> int:
        return bars_per_hour(self.resolution_minutes)

    def lookback_bars(self, hours: int) -> int:
        return max(2, int(hours) * self.bph())


CONFIG = KrakenMaxConfig()
