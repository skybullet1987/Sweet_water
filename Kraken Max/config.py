from __future__ import annotations

from dataclasses import dataclass, field

# Long-only, cash account — Canada / Kraken spot compliance.
ENABLE_SHORTS = False


@dataclass(frozen=True)
class KrakenMaxConfig:
    """Aggressive Kraken spot config targeting high convexity (high risk)."""

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

    # Aggressive deployment (long-only, no margin)
    total_deployment_cap: float = 0.98
    max_position_pct: float = 0.45
    pyramid_add_pct: float = 0.15
    pyramid_min_unrealized_pct: float = 0.05
    target_annual_vol: float = 0.85
    kelly_cap: float = 0.55
    min_position_floor_usd: float = 25.0

    # Ensemble weights (momentum / breakout / dip-in-trend / ML)
    w_momentum: float = 0.35
    w_breakout: float = 0.25
    w_dip: float = 0.15
    w_ml: float = 0.25
    entry_score_threshold: float = 0.42
    replace_score_delta: float = 0.12

    # Regime
    btc_trend_ema: int = 100
    vol_stress_threshold: float = 1.05
    breadth_bull_threshold: float = 0.55
    bear_deployment_cap: float = 0.35
    bear_prefer: tuple[str, ...] = ("BTCUSD", "ETHUSD")

    # Exits
    hard_stop_pct: float = -0.08
    catastrophic_stop_pct: float = -0.12
    tp_atr_mult: float = 4.5
    sl_atr_mult: float = 1.2
    chandelier_atr_mult: float = 2.5
    activate_trail_above_pct: float = 0.10
    time_stop_hours: float = 72.0

    # Portfolio risk (still risky — wide halt vs conservative stack)
    drawdown_halt_pct: float = -0.28
    drawdown_cooldown_hours: int = 24
    max_orders_per_day: int = 8
    post_breaker_cooldown_hours: int = 12

    # Costs / gates (looser than qc_runtime — accepts more churn for upside)
    expected_round_trip_fees: float = 0.0052
    edge_cost_multiplier: float = 1.8
    edge_scale: float = 0.04
    assumed_spread_bps: float = 18.0
    assumed_slippage_bps: float = 12.0
    min_rebalance_weight_delta: float = 0.02

    feature_min_bars: int = 48
    score_clip: float = 4.0
    enable_shorts: bool = ENABLE_SHORTS

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
        "SHIBUSD": 50000.0,
    })


CONFIG = KrakenMaxConfig()
