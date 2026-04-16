from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Mapping, Optional, Sequence

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
class WalkForwardResult:
    """Aggregated result from a rolling / k-fold walk-forward test."""
    fold_metrics: tuple[Mapping[str, float], ...]
    overall_sharpe: float
    sharpe_t_stat: float       # t-statistic for H0: Sharpe == 0
    sharpe_p_value: float      # two-tailed p-value (approximate)
    is_significant: bool       # p < 0.05


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
    3. Enters at the bar close when the signal flips positive and exits when
       it turns negative.  Only one position (in one symbol at a time) is
       held to keep the backtest legible.

    This is deliberately simple: the purpose is to validate the cost model,
    metrics pipeline, and walk-forward logic – not to optimise a strategy.

    Parameters
    ----------
    fee_rate      : one-way fee as a fraction of notional (default: 0.50%)
    slippage_rate : one-way slippage as a fraction (default: 0.40%)
    """

    def __init__(
        self,
        fee_rate: float = _DEFAULT_FEE_RATE,
        slippage_rate: float = _DEFAULT_SLIPPAGE_RATE,
    ) -> None:
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self._rt_cost = _round_trip_cost(fee_rate, slippage_rate)

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
    ) -> tuple[List[float], List[float]]:
        """
        Replay a sorted list of bars.

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
        rt_cost = _round_trip_cost(
            self.fee_rate * fee_mult,
            self.slippage_rate * slip_mult,
        )

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

            # Position return
            position_ret = bar_ret if was_in else 0.0

            # Cost on state changes
            cost = 0.0
            if signal and not was_in:
                cost = rt_cost / 2.0   # entry cost (half of round-trip)
                in_position[sym] = True
            elif not signal and was_in:
                cost = rt_cost / 2.0   # exit cost
                in_position[sym] = False

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
    ) -> RunResult:
        """
        Run a single bar-replay backtest.

        Parameters
        ----------
        config         : experiment configuration (dates, symbols, initial cash)
        bars_by_symbol : dict[symbol_str → list[Bar]], pre-sorted or unsorted
        git_sha        : optional git commit SHA to embed in metadata

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

        rets, curve = self._replay_bars(filtered)
        metrics = self._compute_metrics(rets, curve)

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
    ) -> WalkForwardResult:
        """
        Rolling walk-forward test.

        The bars are split into ``n_folds`` non-overlapping sequential test
        folds.  Each fold's bars are back-tested independently.  The fold
        Sharpe ratios are then tested for statistical significance via a
        one-sample t-test (H0: mean fold Sharpe == 0).

        Parameters
        ----------
        config         : experiment configuration
        bars_by_symbol : full bar history (will be split chronologically)
        n_folds        : number of sequential test folds (≥ 2)
        git_sha        : optional

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
            )

        t_stat, p_value = _t_test_one_sample(fold_sharpes, mu0=0.0)
        overall_sharpe = sum(fold_sharpes) / len(fold_sharpes)

        return WalkForwardResult(
            fold_metrics=tuple(fold_results),
            overall_sharpe=float(overall_sharpe),
            sharpe_t_stat=float(t_stat),
            sharpe_p_value=float(p_value),
            is_significant=p_value < 0.05,
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
