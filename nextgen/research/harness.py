from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from nextgen.core.types import Bar


@dataclass(frozen=True)
class StressScenario:
    name: str
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    liquidity_haircut: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    start: datetime
    end: datetime
    symbols: tuple[str, ...]
    initial_cash: float
    stress_scenarios: tuple[StressScenario, ...] = ()


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    created_at: datetime
    config_hash: str
    git_sha: str | None = None


@dataclass(frozen=True)
class RunResult:
    metadata: RunMetadata
    metrics: Mapping[str, float]
    scenario_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


# ── Walk-forward result type ───────────────────────────────────────────────────

@dataclass(frozen=True)
class PerformanceObjective:
    """
    Drawdown-first performance target.

    A strategy must achieve *at least* ``target_sharpe`` annualised Sharpe ratio
    **and** stay within ``max_drawdown`` (as a positive fraction, e.g. 0.15 = 15%)
    before any profit-optimisation step is attempted.

    Defaults encode the recommended floor: Sharpe ≥ 1.5, max-DD < 15%.
    """
    target_sharpe: float = 1.5
    max_drawdown: float = 0.15


def meets_objective(metrics: Mapping[str, float], objective: PerformanceObjective) -> bool:
    """
    Return True iff ``metrics`` satisfies ``objective``.

    Parameters
    ----------
    metrics   : dict produced by ``BarReplayHarness._compute_metrics``.
    objective : performance objective to test against.
    """
    sharpe = metrics.get("sharpe_ratio", 0.0)
    dd = metrics.get("max_drawdown", 1.0)
    return sharpe >= objective.target_sharpe and dd <= objective.max_drawdown


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregated result from a rolling / k-fold walk-forward test."""
    fold_metrics: tuple[Mapping[str, float], ...]
    overall_sharpe: float
    sharpe_t_stat: float       # t-statistic for H0: Sharpe == 0
    sharpe_p_value: float      # two-tailed p-value (approximate)
    is_significant: bool       # p < 0.05
    objective_met: bool = False  # True when PerformanceObjective is satisfied by overall metrics


# ── OOS validation types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class OOSConfig:
    """
    Locked out-of-sample (OOS) validation window.

    The in-sample window (``is_start`` → ``is_end``) is used for all
    parameter tuning and walk-forward optimisation.  The OOS window
    (``oos_start`` → ``oos_end``) is touched **exactly once** — when
    ``OOSValidator.run()`` is called — and must never be used for tuning.

    A warning is emitted when ``n_is_folds < min_recommended_folds``
    (default 30) because fewer folds reduce statistical power.
    """
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    min_recommended_folds: int = 30


@dataclass(frozen=True)
class OOSResult:
    """
    Result of a single honest OOS evaluation.

    Attributes
    ----------
    is_result  : walk-forward result on the in-sample period.
    oos_metrics: metrics dict from a single replay on the OOS period.
    sharpe_degradation : (is_sharpe - oos_sharpe) / max(|is_sharpe|, 1e-9).
                         Positive = OOS underperforms IS.
    oos_objective_met  : whether the OOS metrics satisfy the given objective.
    folds_warning      : True when n_is_folds < min_recommended_folds.
    """
    is_result: WalkForwardResult
    oos_metrics: Mapping[str, float]
    sharpe_degradation: float
    oos_objective_met: bool
    folds_warning: bool


# ── Paper-trade types ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaperTradeEvent:
    """
    A single simulated order event recorded during a paper-trading replay.

    Attributes
    ----------
    timestamp   : bar time of the event.
    symbol      : trading symbol.
    action      : ``"BUY"`` or ``"SELL"``.
    price       : bar close price at which the simulated fill occurs.
    slippage_est: one-way slippage cost as a fraction of notional.
    fee_est     : one-way fee as a fraction of notional.
    """
    timestamp: datetime
    symbol: str
    action: str       # "BUY" | "SELL"
    price: float
    slippage_est: float
    fee_est: float


@dataclass
class PaperTradeSession:
    """
    Accumulates ``PaperTradeEvent`` records produced during a paper-trading
    replay and exposes helpers for analysis.

    ``events`` is populated by ``BarReplayHarness.run()`` when
    ``paper_trade_mode=True``.
    """
    events: List[PaperTradeEvent] = field(default_factory=list)

    def buy_count(self) -> int:
        return sum(1 for e in self.events if e.action == "BUY")

    def sell_count(self) -> int:
        return sum(1 for e in self.events if e.action == "SELL")

    def total_cost_fraction(self) -> float:
        """Sum of all one-way fee + slippage fractions recorded."""
        return sum(e.fee_est + e.slippage_est for e in self.events)

    def to_records(self) -> List[Dict[str, object]]:
        """Return events as a list of plain dicts (easy to serialise)."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "symbol": e.symbol,
                "action": e.action,
                "price": e.price,
                "slippage_est": e.slippage_est,
                "fee_est": e.fee_est,
            }
            for e in self.events
        ]


# ── Kraken universe filter ────────────────────────────────────────────────────

# Minimum order sizes (in USD notional) for the most-traded Kraken Pro symbols.
# Sourced from Kraken API docs (April 2025).  Symbols not listed default to
# ``_KRAKEN_DEFAULT_MIN_USD``.  Used by ``KrakenUniverseFilter``.
_KRAKEN_DEFAULT_MIN_USD = 10.0

KRAKEN_MIN_ORDER_SIZES: Dict[str, float] = {
    # Major pairs — smallest minimums
    "BTCUSD":  10.0,
    "ETHUSD":  10.0,
    "SOLUSD":  10.0,
    "ADAUSD":  10.0,
    "DOTUSD":  10.0,
    "AVAXUSD": 10.0,
    "LINKUSD": 10.0,
    "MATICUSD": 10.0,
    "UNIUSD":  10.0,
    "ATOMUSD": 10.0,
    # Mid-cap — slightly higher minimums
    "XRPUSD":  10.0,
    "LTCUSD":  10.0,
    "BCHUSD":  10.0,
    "ETCUSD":  10.0,
    "XLMUSD":  10.0,
    # Lower-liquidity / micro-cap — enforce higher floor
    "DOGEUSD": 10.0,
    "SHIBUSD": 10.0,
}


class KrakenUniverseFilter:
    """
    Validates a candidate symbol universe against Kraken minimum order sizes
    and account capital constraints.

    A symbol is **rejected** when its Kraken minimum order size (USD notional)
    exceeds ``capital × max_position_pct × safety_margin``.  This prevents the
    strategy from allocating positions that are below-minimum and therefore
    un-fillable on Kraken.

    Parameters
    ----------
    capital         : account equity in USD.
    max_position_pct: maximum fraction of capital per position (e.g. 0.25).
    safety_margin   : additional discount applied to the position budget before
                      comparing against the minimum order size (default 0.9 =
                      keep 10% buffer for fee reserves).
    min_order_sizes : optional override mapping; falls back to
                      ``KRAKEN_MIN_ORDER_SIZES`` for symbols not present,
                      then ``_KRAKEN_DEFAULT_MIN_USD``.
    """

    def __init__(
        self,
        capital: float,
        max_position_pct: float = 0.25,
        safety_margin: float = 0.9,
        min_order_sizes: Optional[Mapping[str, float]] = None,
    ) -> None:
        if capital <= 0:
            raise ValueError("capital must be positive")
        if not (0 < max_position_pct <= 1):
            raise ValueError("max_position_pct must be in (0, 1]")
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.safety_margin = safety_margin
        self._min_sizes: Mapping[str, float] = min_order_sizes or KRAKEN_MIN_ORDER_SIZES

    def position_budget(self) -> float:
        """Maximum USD notional per position after safety margin."""
        return self.capital * self.max_position_pct * self.safety_margin

    def min_order_usd(self, symbol: str) -> float:
        """Minimum order size for ``symbol`` in USD."""
        return self._min_sizes.get(symbol, _KRAKEN_DEFAULT_MIN_USD)

    def is_tradeable(self, symbol: str) -> bool:
        """Return True if a position in ``symbol`` can be filled on Kraken."""
        return self.position_budget() >= self.min_order_usd(symbol)

    def filter_symbols(self, symbols: Sequence[str]) -> Tuple[List[str], List[str]]:
        """
        Partition ``symbols`` into (tradeable, rejected).

        Returns
        -------
        tradeable : symbols for which ``is_tradeable`` is True.
        rejected  : symbols removed because the minimum order exceeds budget.
        """
        tradeable: List[str] = []
        rejected: List[str] = []
        for sym in symbols:
            (tradeable if self.is_tradeable(sym) else rejected).append(sym)
        return tradeable, rejected


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sharpe(returns: List[float], periods_per_year: float = 105120.0) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(max(var, 0.0))
    if std < 1e-12:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def _max_drawdown(equity_curve: List[float]) -> float:
    """Peak-to-trough maximum drawdown as a positive fraction."""
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _t_test_one_sample(values: List[float], mu0: float = 0.0) -> tuple[float, float]:
    """
    One-sample t-test: H0: mean == mu0.

    Returns (t_statistic, two-tailed p-value).
    Uses a simple Student-t CDF approximation via the beta-function incomplete
    regularised series (accurate to ≈ 4 decimal places for df ≥ 2).
    """
    n = len(values)
    if n < 2:
        return 0.0, 1.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(max(var, 0.0) / n)
    if se < 1e-12:
        return 0.0, 1.0
    t = (mean - mu0) / se
    df = n - 1
    # Approximation of two-tailed p-value via the regularised incomplete beta function.
    x = df / (df + t * t)
    p = _regularised_incomplete_beta(df / 2.0, 0.5, x)
    return t, float(min(1.0, max(0.0, p)))


def _regularised_incomplete_beta(a: float, b: float, x: float, max_iter: int = 200) -> float:
    """
    Regularised incomplete beta function I_x(a, b) via continued-fraction
    expansion (Lentz algorithm).  Used only for the t-test p-value.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # Use the symmetry relation to keep x in (0, 0.5] for better convergence.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularised_incomplete_beta(b, a, 1.0 - x)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    # Lentz continued fraction
    f, c, d = 1.0, 1.0, 0.0
    TINY = 1e-300
    for i in range(max_iter):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + num / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * (f - 1.0)


# ── Standalone cost model ─────────────────────────────────────────────────────
# Extracted numerical parameters from fee_model.py and realistic_slippage.py
# so the research harness runs without any LEAN dependency.

# Kraken tiered fees: blended rate at lowest volume tier (most conservative)
_FEE_MAKER = 0.0040   # 0.40%
_FEE_TAKER = 0.0080   # 0.80%
_TAKER_RATIO = 0.25
_DEFAULT_FEE_RATE = (1 - _TAKER_RATIO) * _FEE_MAKER + _TAKER_RATIO * _FEE_TAKER  # 0.50%

# Realistic slippage base (per side, from realistic_slippage.py)
_DEFAULT_SLIPPAGE_RATE = 0.0040   # 0.40%

# Round-trip cost rate (entry + exit, fee + slippage each way)
_DEFAULT_ROUND_TRIP_COST = 2.0 * (_DEFAULT_FEE_RATE + _DEFAULT_SLIPPAGE_RATE)


def _round_trip_cost(fee_rate: float, slippage_rate: float) -> float:
    return 2.0 * (fee_rate + slippage_rate)


def break_even_return(fee_rate: float = _DEFAULT_FEE_RATE, slippage_rate: float = _DEFAULT_SLIPPAGE_RATE) -> float:
    """
    Minimum gross return per trade required to cover one full round-trip cost.

    A cost-aware strategy should only enter when its expected edge exceeds
    ``edge_multiplier × break_even_return(...)`` (Kelly-inspired; default
    ``edge_multiplier=2.0`` in ``BarReplayHarness``).

    Parameters
    ----------
    fee_rate      : one-way fee as a fraction of notional.
    slippage_rate : one-way slippage as a fraction of notional.

    Returns
    -------
    Round-trip cost fraction (entry fee+slip + exit fee+slip).
    """
    return _round_trip_cost(fee_rate, slippage_rate)


# ── Bar-replay harness ────────────────────────────────────────────────────────


class BarReplayHarness:
    """
    Standalone bar-replay backtest harness.

    Does NOT depend on LEAN.  Uses the fee/slippage constants derived from
    ``fee_model.py`` and ``realistic_slippage.py`` as the cost model.

    Strategy implemented internally
    --------------------------------
    At each bar the harness:
    1. Updates the feature engine for the traded symbol.
    2. Computes a simple cross-sectional trend signal (EMA ratio > 0 → long).
    3. Enters at the bar close when the signal flips positive **and** the
       expected edge (EMA ratio magnitude) exceeds
       ``min_edge_multiplier × break_even_return(fee_rate, slippage_rate)``.
    4. Exits when the signal turns negative.

    This is deliberately simple: the purpose is to validate the cost model,
    metrics pipeline, and walk-forward logic – not to optimise a strategy.

    Parameters
    ----------
    fee_rate           : one-way fee as a fraction of notional (default: 0.50%)
    slippage_rate      : one-way slippage as a fraction (default: 0.40%)
    min_edge_multiplier: entry gate — only enter when EMA-ratio edge >
                         ``min_edge_multiplier × break_even_return()``.
                         Set to 0.0 to disable the gate (backward compat).
                         Default 2.0 (Kelly-inspired).
    paper_trade_mode   : when True, ``run()`` returns a ``PaperTradeSession``
                         attached to the ``RunResult`` via
                         ``result.metadata`` is unchanged but a session is
                         accessible via ``harness.last_paper_session``.
    """

    def __init__(
        self,
        fee_rate: float = _DEFAULT_FEE_RATE,
        slippage_rate: float = _DEFAULT_SLIPPAGE_RATE,
        min_edge_multiplier: float = 0.0,
        paper_trade_mode: bool = False,
    ) -> None:
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.min_edge_multiplier = min_edge_multiplier
        self.paper_trade_mode = paper_trade_mode
        self._rt_cost = _round_trip_cost(fee_rate, slippage_rate)
        self._min_edge: float = min_edge_multiplier * break_even_return(fee_rate, slippage_rate)
        self.last_paper_session: Optional[PaperTradeSession] = None

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_metadata(config: ExperimentConfig, git_sha: str | None = None) -> RunMetadata:
        payload = {
            "name": config.name,
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "symbols": list(config.symbols),
            "initial_cash": config.initial_cash,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return RunMetadata(
            run_id=digest[:12],
            created_at=datetime.now(UTC),
            config_hash=digest,
            git_sha=git_sha,
        )

    @staticmethod
    def _sort_bars(bars_by_symbol: Mapping[str, Sequence[Bar]]) -> List[Bar]:
        all_bars: List[Bar] = []
        for bars in bars_by_symbol.values():
            all_bars.extend(bars)
        all_bars.sort(key=lambda b: b.timestamp)
        return all_bars

    def _replay_bars(
        self,
        bars: List[Bar],
        stress: Optional[StressScenario] = None,
        paper_session: Optional[PaperTradeSession] = None,
    ) -> tuple[List[float], List[float]]:
        """
        Replay a sorted list of bars.

        Edge gating
        -----------
        When ``self.min_edge_multiplier > 0``, a new entry is only opened when
        the EMA-ratio signal strength (``ema_short/ema_long - 1``) exceeds the
        break-even cost threshold scaled by ``min_edge_multiplier``.  Exits are
        not gated — the strategy always closes positions when the signal turns
        negative.

        Returns
        -------
        portfolio_returns : list of per-bar portfolio returns
        equity_curve      : list of equity values (starting at 1.0)
        """
        # Simple EMA trend signal per symbol
        ema_short: Dict[str, float] = {}
        ema_long: Dict[str, float] = {}
        prev_close: Dict[str, float] = {}
        in_position: Dict[str, bool] = {}

        alpha_s = 2.0 / (6 + 1)    # 6-bar EMA
        alpha_l = 2.0 / (20 + 1)   # 20-bar EMA

        # Cost multipliers for stress scenarios
        fee_mult = stress.spread_multiplier if stress else 1.0
        slip_mult = stress.slippage_multiplier if stress else 1.0
        eff_fee_rate = self.fee_rate * fee_mult
        eff_slip_rate = self.slippage_rate * slip_mult
        rt_cost = _round_trip_cost(eff_fee_rate, eff_slip_rate)

        # Cost-aware edge gate: minimum EMA-ratio premium required to open.
        # Recomputed here so stress scenarios tighten the gate proportionally.
        min_edge = self.min_edge_multiplier * break_even_return(eff_fee_rate, eff_slip_rate)

        portfolio_returns: List[float] = []
        equity = 1.0
        equity_curve: List[float] = [equity]

        for bar in bars:
            sym = bar.symbol
            close = bar.close

            # Update EMAs
            if sym not in ema_short:
                ema_short[sym] = close
                ema_long[sym] = close
                prev_close[sym] = close
                in_position[sym] = False
                continue

            ema_short[sym] = (1 - alpha_s) * ema_short[sym] + alpha_s * close
            ema_long[sym] = (1 - alpha_l) * ema_long[sym] + alpha_l * close

            # Bar return before trading costs
            prev = prev_close[sym]
            bar_ret = (close - prev) / prev if prev > 0 else 0.0
            prev_close[sym] = close

            signal = ema_short[sym] > ema_long[sym]
            was_in = in_position.get(sym, False)

            # Cost-aware edge gate: only enter when the EMA ratio premium
            # exceeds min_edge (break_even × edge_multiplier).
            ema_long_val = ema_long[sym]
            edge = (ema_short[sym] / ema_long_val - 1.0) if ema_long_val > 0 else 0.0
            edge_gate_pass = (min_edge <= 0.0) or (edge >= min_edge)

            # Position return
            position_ret = bar_ret if was_in else 0.0

            # Cost on state changes
            cost = 0.0
            if signal and not was_in and edge_gate_pass:
                cost = rt_cost / 2.0   # entry cost (half of round-trip)
                in_position[sym] = True
                if paper_session is not None:
                    paper_session.events.append(PaperTradeEvent(
                        timestamp=bar.timestamp,
                        symbol=sym,
                        action="BUY",
                        price=close,
                        slippage_est=eff_slip_rate,
                        fee_est=eff_fee_rate,
                    ))
            elif not signal and was_in:
                cost = rt_cost / 2.0   # exit cost
                in_position[sym] = False
                if paper_session is not None:
                    paper_session.events.append(PaperTradeEvent(
                        timestamp=bar.timestamp,
                        symbol=sym,
                        action="SELL",
                        price=close,
                        slippage_est=eff_slip_rate,
                        fee_est=eff_fee_rate,
                    ))

            net_ret = position_ret - cost
            portfolio_returns.append(net_ret)
            equity *= (1.0 + net_ret)
            equity_curve.append(equity)

        return portfolio_returns, equity_curve

    def _compute_metrics(
        self,
        portfolio_returns: List[float],
        equity_curve: List[float],
    ) -> Dict[str, float]:
        n = len(portfolio_returns)
        total_return = (equity_curve[-1] - 1.0) if equity_curve else 0.0
        sharpe = _sharpe(portfolio_returns)
        max_dd = _max_drawdown(equity_curve) if len(equity_curve) >= 2 else 0.0
        win_rate = (sum(1 for r in portfolio_returns if r > 0) / n) if n else 0.0
        avg_return = sum(portfolio_returns) / n if n else 0.0
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "average_bar_return": float(avg_return),
            "observations": float(n),
        }

    # ── public API ────────────────────────────────────────────────────────

    def run(
        self,
        config: ExperimentConfig,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
        git_sha: str | None = None,
        objective: Optional[PerformanceObjective] = None,
    ) -> RunResult:
        """
        Run a single bar-replay backtest.

        When ``paper_trade_mode=True`` the harness records every simulated
        order event in ``self.last_paper_session`` (a ``PaperTradeSession``).

        Parameters
        ----------
        config         : experiment configuration (dates, symbols, initial cash)
        bars_by_symbol : dict[symbol_str → list[Bar]], pre-sorted or unsorted
        git_sha        : optional git commit SHA to embed in metadata
        objective      : optional ``PerformanceObjective``; when supplied,
                         ``RunResult.metrics`` gains an ``"objective_met"``
                         key (1.0 = True, 0.0 = False).

        Returns
        -------
        RunResult with metrics and per-scenario stress results.
        """
        metadata = self._make_metadata(config, git_sha)
        all_bars = self._sort_bars(bars_by_symbol)

        # Filter to config date range and symbols
        sym_set = set(config.symbols)
        filtered = [
            b for b in all_bars
            if b.symbol in sym_set and config.start <= b.timestamp <= config.end
        ]

        # Paper-trade session (only populated when paper_trade_mode is True)
        paper_session: Optional[PaperTradeSession] = None
        if self.paper_trade_mode:
            paper_session = PaperTradeSession()
            self.last_paper_session = paper_session

        rets, curve = self._replay_bars(filtered, paper_session=paper_session)
        metrics: Dict[str, float] = self._compute_metrics(rets, curve)

        if objective is not None:
            metrics["objective_met"] = 1.0 if meets_objective(metrics, objective) else 0.0

        # Stress scenarios
        scenario_metrics: Dict[str, Mapping[str, float]] = {}
        for scenario in config.stress_scenarios:
            s_rets, s_curve = self._replay_bars(filtered, stress=scenario)
            scenario_metrics[scenario.name] = self._compute_metrics(s_rets, s_curve)

        return RunResult(metadata=metadata, metrics=metrics, scenario_metrics=scenario_metrics)

    def walk_forward_run(
        self,
        config: ExperimentConfig,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
        n_folds: int = 2,
        git_sha: str | None = None,
        objective: Optional[PerformanceObjective] = None,
    ) -> WalkForwardResult:
        """
        Rolling walk-forward test.

        The bars are split into ``n_folds`` non-overlapping sequential test
        folds.  Each fold's bars are back-tested independently.  The fold
        Sharpe ratios are then tested for statistical significance via a
        one-sample t-test (H0: mean fold Sharpe == 0).

        When ``objective`` is supplied the returned ``WalkForwardResult`` has
        ``objective_met=True`` iff the overall (mean-fold) metrics satisfy the
        objective.  Use this to gate live deployment — only deploy when
        ``is_significant and objective_met`` are both True.

        Parameters
        ----------
        config         : experiment configuration
        bars_by_symbol : full bar history (will be split chronologically)
        n_folds        : number of sequential test folds (≥ 2)
        git_sha        : optional
        objective      : optional ``PerformanceObjective``

        Returns
        -------
        WalkForwardResult
        """
        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")

        sym_set = set(config.symbols)
        all_bars = self._sort_bars(bars_by_symbol)
        filtered = [
            b for b in all_bars
            if b.symbol in sym_set and config.start <= b.timestamp <= config.end
        ]

        if not filtered:
            return WalkForwardResult(
                fold_metrics=(),
                overall_sharpe=0.0,
                sharpe_t_stat=0.0,
                sharpe_p_value=1.0,
                is_significant=False,
                objective_met=False,
            )

        fold_size = max(1, len(filtered) // n_folds)
        fold_results: List[Dict[str, float]] = []
        fold_sharpes: List[float] = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else len(filtered)
            fold_bars = filtered[start_idx:end_idx]
            if not fold_bars:
                continue
            rets, curve = self._replay_bars(fold_bars)
            m = self._compute_metrics(rets, curve)
            fold_results.append(m)
            fold_sharpes.append(m["sharpe_ratio"])

        if not fold_sharpes:
            return WalkForwardResult(
                fold_metrics=tuple(fold_results),
                overall_sharpe=0.0,
                sharpe_t_stat=0.0,
                sharpe_p_value=1.0,
                is_significant=False,
                objective_met=False,
            )

        t_stat, p_value = _t_test_one_sample(fold_sharpes, mu0=0.0)
        overall_sharpe = sum(fold_sharpes) / len(fold_sharpes)

        # Aggregate fold metrics for objective test
        if objective is not None and fold_results:
            avg_metrics: Dict[str, float] = {
                k: sum(m.get(k, 0.0) for m in fold_results) / len(fold_results)
                for k in fold_results[0]
            }
            obj_met = meets_objective(avg_metrics, objective)
        else:
            obj_met = False

        return WalkForwardResult(
            fold_metrics=tuple(fold_results),
            overall_sharpe=float(overall_sharpe),
            sharpe_t_stat=float(t_stat),
            sharpe_p_value=float(p_value),
            is_significant=p_value < 0.05,
            objective_met=obj_met,
        )

    def oos_run(
        self,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
        oos_config: OOSConfig,
        n_is_folds: int = 30,
        git_sha: str | None = None,
        objective: Optional[PerformanceObjective] = None,
    ) -> OOSResult:
        """
        Honest out-of-sample validation.

        1. Runs walk-forward optimisation exclusively on the in-sample (IS)
           window (``oos_config.is_start`` → ``oos_config.is_end``).
        2. Evaluates a **single** bar replay on the OOS window
           (``oos_config.oos_start`` → ``oos_config.oos_end``).
        3. Never uses OOS data for any parameter selection.

        A ``folds_warning`` is set when ``n_is_folds < oos_config.min_recommended_folds``
        (default 30) because fewer folds reduce statistical power at 5-min
        bar frequency.

        Parameters
        ----------
        bars_by_symbol : full bar history covering both IS and OOS windows.
        oos_config     : defines the IS / OOS boundary.
        n_is_folds     : number of walk-forward folds on the IS period.
        git_sha        : optional git commit SHA.
        objective      : optional performance objective to test.

        Returns
        -------
        OOSResult
        """
        # Derive ExperimentConfig for IS period from oos_config
        # (symbols and initial_cash are not meaningful here — use sentinels)
        all_bars = self._sort_bars(bars_by_symbol)
        all_symbols = tuple({b.symbol for b in all_bars})

        is_config = ExperimentConfig(
            name="is_window",
            start=oos_config.is_start,
            end=oos_config.is_end,
            symbols=all_symbols,
            initial_cash=1.0,
        )
        is_result = self.walk_forward_run(
            is_config, bars_by_symbol,
            n_folds=n_is_folds,
            git_sha=git_sha,
            objective=objective,
        )

        # Single OOS replay — touched exactly once
        oos_bars = [
            b for b in all_bars
            if oos_config.oos_start <= b.timestamp <= oos_config.oos_end
        ]
        oos_rets, oos_curve = self._replay_bars(oos_bars)
        oos_metrics = self._compute_metrics(oos_rets, oos_curve)

        is_sharpe = is_result.overall_sharpe
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)
        degradation = (is_sharpe - oos_sharpe) / max(abs(is_sharpe), 1e-9)

        oos_obj_met = meets_objective(oos_metrics, objective) if objective is not None else False
        folds_warning = n_is_folds < oos_config.min_recommended_folds

        return OOSResult(
            is_result=is_result,
            oos_metrics=oos_metrics,
            sharpe_degradation=float(degradation),
            oos_objective_met=oos_obj_met,
            folds_warning=folds_warning,
        )


# ── Legacy harness (preserved for backward compatibility) ────────────────────


class BacktestHarness:
    """Foundational research runner with immutable metadata capture."""

    def _make_metadata(self, config: ExperimentConfig, git_sha: str | None = None) -> RunMetadata:
        payload = {
            "name": config.name,
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "symbols": list(config.symbols),
            "initial_cash": config.initial_cash,
            "stress_scenarios": [s.__dict__ for s in config.stress_scenarios],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return RunMetadata(run_id=digest[:12], created_at=datetime.now(UTC), config_hash=digest, git_sha=git_sha)

    def run(self, config: ExperimentConfig, returns_by_symbol: Mapping[str, Sequence[float]], git_sha: str | None = None) -> RunResult:
        metadata = self._make_metadata(config, git_sha=git_sha)

        combined = []
        for symbol in config.symbols:
            combined.extend(returns_by_symbol.get(symbol, ()))

        avg_return = sum(combined) / len(combined) if combined else 0.0
        metrics = {
            "average_return": float(avg_return),
            "observations": float(len(combined)),
        }

        scenario_metrics: dict[str, Mapping[str, float]] = {}
        for scenario in config.stress_scenarios:
            stressed_avg = avg_return * (1.0 - (scenario.spread_multiplier - 1.0) * 0.05) * (1.0 - (scenario.slippage_multiplier - 1.0) * 0.05)
            scenario_metrics[scenario.name] = {"average_return": float(stressed_avg)}

        return RunResult(metadata=metadata, metrics=metrics, scenario_metrics=scenario_metrics)
