from __future__ import annotations

from dataclasses import dataclass, field


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
    universe_size: int = 80
    top_k: int = 8
    signal_mode: str = "cross_sectional_momentum"

    # Sizing
    target_annual_vol: float = 0.30
    kelly_cap: float = 0.25
    min_position_floor_usd: float = 5.0
    max_position_pct: float = 0.30
    max_positions: int = 8

    # Costs
    expected_round_trip_fees: float = 0.0065
    cost_gate_multiplier: float = 2.5
    # cost_gate_multiplier is retained for legacy paths; edge_cost_multiplier is used by the new cross-sectional momentum gate.
    edge_cost_multiplier: float = 2.5
    edge_scale: float = 0.02
    assumed_spread_bps: float = 12.0
    assumed_slippage_bps: float = 8.0
    min_rebalance_weight_delta: float = 0.03

    # Exits (ATR-scaled)
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    chandelier_atr_mult: float = 3.0
    activate_trailing_above_pct: float = 0.005
    time_stop_bars: int = 24

    # Regime gates
    vol_stress_threshold: float = 0.7
    breadth_threshold: float = 0.30
    btc_trend_ema: int = 200
    chop_return_threshold: float = 0.002

    # Scoring
    score_threshold: float = 0.40
    chop_threshold_multiplier: float = 1.5
    cross_section_weight: float = 0.40
    micro_entry_threshold: float = 0.45
    micro_flatten_threshold: float = 0.06
    max_orders_per_day: int = 6
    min_hold_hours: int = 6
    rebalance_cadence_hours: int = 6
    max_replacements_per_rebalance: int = 2
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
    score_clip_value: float = 5.0
    min_rv_floor: float = 1e-4
    score_mom21_weight: float = 0.6
    score_mom63_weight: float = 0.4
    score_dd_penalty: float = 0.3

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
