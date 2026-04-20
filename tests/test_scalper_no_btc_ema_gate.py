from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
QC_RUNTIME = REPO_ROOT / "qc_runtime"
if str(QC_RUNTIME) not in sys.path:
    sys.path.insert(0, str(QC_RUNTIME))

import main as main_module
from main import SweetWaterPhase1


class _Symbol:
    def __init__(self, value: str):
        self.Value = value

    def __hash__(self):
        return hash(self.Value)

    def __eq__(self, other):
        return isinstance(other, _Symbol) and self.Value == other.Value


class _FeatureEngine:
    def __init__(self, mapping):
        self.mapping = mapping

    def current_features(self, key):
        return dict(self.mapping.get(key, {}))


def test_scalper_mode_ignores_btc_ema_gate(monkeypatch):
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    now = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    sym = _Symbol("ETHUSD")
    algo.Time = now
    algo.Debug = lambda *_args, **_kwargs: None
    algo.config = SimpleNamespace(
        strategy_mode="scalper",
        scalper_use_btc_ema_gate=False,
        scalper_daily_loss_brake=-0.03,
        scalper_max_concurrent=4,
        scalper_universe_size=20,
        scalper_position_size_pct=0.15,
        min_position_floor_usd=5.0,
        scalper_btc_1h_floor=-0.015,
        scalper_btc_6h_floor=-0.04,
        scalper_btc_panic_threshold=-0.025,
        scalper_z_entry=-2.0,
        scalper_rsi_min=20.0,
        scalper_rsi_max=40.0,
        scalper_rv_max=1.5,
        scalper_anti_churn_hours=3.0,
        scalper_consecutive_loss_brake=3,
    )
    algo.Portfolio = type(
        "_P",
        (),
        {
            "TotalPortfolioValue": 500.0,
            "Cash": 500.0,
            "CashBook": {"USD": SimpleNamespace(Amount=500.0)},
            "__getitem__": lambda _self, _symbol: SimpleNamespace(AveragePrice=100.0),
        },
    )()
    algo.Securities = {sym: SimpleNamespace(Price=100.0)}
    algo.symbol_by_ticker = {"BTCUSD": _Symbol("BTCUSD")}
    algo.feature_engine = _FeatureEngine(
        {
            "BTCUSD": {"ret_1h": 0.0, "ret_6h": 0.0},
            "ETHUSD": {"z_20h": -2.6, "rsi_14": 30.0, "rv_21d": 0.3},
        }
    )
    algo.regime_engine = SimpleNamespace(btc_above_ema30d=lambda: False)
    algo._current_holdings = lambda: []
    algo.symbols = [sym]
    algo.position_state = {}
    algo._scalper_last_trade_time = {}
    algo._scalper_consec_losses = {}
    algo._scalper_daily_pnl = 0.0
    algo._scalper_daily_anchor_equity = 500.0
    algo._scalper_daily_anchor_date = now.date()
    algo._scalper_session_brake_until = None
    algo._scalper_recent_pnls = main_module.deque(maxlen=6)
    algo._failed_escalations = {}
    algo._breaker_liquidated = False

    submitted = []
    monkeypatch.setattr(main_module, "round_quantity", lambda _algo, _symbol, qty: qty)
    monkeypatch.setattr(
        main_module,
        "place_entry",
        lambda _algo, _symbol, _qty, tag="", signal_score=0.0: submitted.append(tag) or object(),
    )

    algo._scalper_on_data(SimpleNamespace())
    assert submitted


def test_momentum_mode_respects_btc_ema_gate():
    algo = SweetWaterPhase1.__new__(SweetWaterPhase1)
    now = datetime(2025, 1, 2, 16, tzinfo=timezone.utc)
    algo.Time = now
    algo.Debug = lambda *_args, **_kwargs: None
    algo.config = SimpleNamespace()
    algo._score_candidates = lambda _data, btc_ret=None, breadth=None: ("risk_on", [(_Symbol("ETHUSD"), 1.0, {})])
    algo._dispersion_history = []
    algo._rebalance_due = lambda: True
    calls = []
    algo._rebalance_portfolio = lambda _scored, risk_scale=1.0: calls.append(risk_scale)
    algo.regime_engine = SimpleNamespace(btc_above_ema30d=lambda: False)
    algo._last_scored = []
    algo._last_force_exit_date = None
    algo._last_dispersion_log_date = None
    algo._bar_count = 1
    algo._last_trade_bar = 1
    algo._last_no_trade_log_bar = 0
    algo.reporter = SimpleNamespace(tick=lambda _state: None)

    algo._momentum_on_data(SimpleNamespace(), btc_ret=0.0, breadth=0.5)
    assert calls == []
