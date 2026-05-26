from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from config import CONFIG, KrakenMaxConfig


@dataclass
class TradeRecord:
    ticker: str
    pnl_pct: float
    strategy: str
    when: str


@dataclass
class ScorecardSnapshot:
    ts: str
    equity: float
    peak_equity: float
    drawdown: float
    live_sharpe: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade_pct: float
    fill_rate: float
    avg_slippage_bps: float
    days_tracked: float
    version: str = "v7"


class PaperTradingScorecard:
    """v7 paper/live performance scorecard — ObjectStore + telemetry feed."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self._equity: deque[tuple[datetime, float]] = deque(maxlen=config.scorecard_max_points)
        self._trades: deque[TradeRecord] = deque(maxlen=config.scorecard_max_trades)
        self.peak_equity = 0.0
        self.started_at: datetime | None = None

    def record_equity(self, when: datetime, equity: float) -> None:
        eq = float(equity)
        if self.started_at is None:
            self.started_at = when
        self.peak_equity = max(self.peak_equity, eq)
        self._equity.append((when, eq))

    def record_trade(self, ticker: str, pnl_pct: float, *, strategy: str = "momentum", when: datetime | None = None) -> None:
        ts = when or datetime.now(timezone.utc)
        self._trades.append(
            TradeRecord(
                ticker=str(ticker),
                pnl_pct=float(pnl_pct),
                strategy=str(strategy),
                when=ts.isoformat(),
            )
        )

    def _returns(self) -> list[float]:
        pts = list(self._equity)
        rets = []
        for i in range(1, len(pts)):
            prev, cur = pts[i - 1][1], pts[i][1]
            if prev > 0:
                rets.append(cur / prev - 1.0)
        return rets

    def live_sharpe(self) -> float:
        rets = self._returns()
        if len(rets) < 8:
            return 0.0
        ppy = 24 * 365 * self.config.bph()
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
        sd = math.sqrt(max(var, 1e-12))
        if sd <= 0:
            return 0.0
        return (mu / sd) * math.sqrt(ppy)

    def win_rate(self) -> float:
        if not self._trades:
            return 0.0
        wins = sum(1 for t in self._trades if t.pnl_pct > 0)
        return wins / len(self._trades)

    def profit_factor(self) -> float:
        gains = sum(t.pnl_pct for t in self._trades if t.pnl_pct > 0)
        losses = -sum(t.pnl_pct for t in self._trades if t.pnl_pct < 0)
        if losses <= 1e-12:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    def current_drawdown(self, equity: float) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return float(equity) / self.peak_equity - 1.0

    def days_tracked(self) -> float:
        if self.started_at is None or not self._equity:
            return 0.0
        delta = self._equity[-1][0] - self.started_at
        return max(0.0, delta.total_seconds() / 86400.0)

    def build(self, algo, *, fill_tracker=None) -> ScorecardSnapshot:
        equity = float(getattr(algo.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        now = getattr(algo, "Time", datetime.now(timezone.utc))
        self.record_equity(now, equity)
        fill_rate = 1.0
        slip = 0.0
        ft = fill_tracker or getattr(algo, "fill_tracker", None)
        if ft is not None:
            fill_rate = float(ft.stats.fill_rate)
            slip = float(ft.stats.avg_slippage_bps)
        trades = list(self._trades)
        avg_pnl = sum(t.pnl_pct for t in trades) / len(trades) if trades else 0.0
        pf = self.profit_factor()
        if math.isinf(pf):
            pf = 99.0
        return ScorecardSnapshot(
            ts=now.isoformat() if hasattr(now, "isoformat") else str(now),
            equity=equity,
            peak_equity=self.peak_equity,
            drawdown=self.current_drawdown(equity),
            live_sharpe=self.live_sharpe(),
            win_rate=self.win_rate(),
            profit_factor=float(pf),
            n_trades=len(trades),
            avg_trade_pct=avg_pnl,
            fill_rate=fill_rate,
            avg_slippage_bps=slip,
            days_tracked=self.days_tracked(),
        )

    def persist(self, algo, snapshot: ScorecardSnapshot) -> None:
        key = str(self.config.scorecard_object_store_key)
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps(asdict(snapshot), indent=2))
        except Exception:
            return

    def summary_line(self, snap: ScorecardSnapshot) -> str:
        return (
            f"KM_SCORECARD eq={snap.equity:.0f} dd={snap.drawdown:.1%} "
            f"sharpe={snap.live_sharpe:.2f} wr={snap.win_rate:.0%} pf={snap.profit_factor:.2f} "
            f"trades={snap.n_trades} days={snap.days_tracked:.1f}"
        )

    def passes_paper_gate(self, snap: ScorecardSnapshot) -> tuple[bool, str]:
        if snap.days_tracked < float(self.config.paper_min_days):
            return False, f"paper_days={snap.days_tracked:.1f}<{self.config.paper_min_days}"
        if snap.live_sharpe < float(self.config.paper_min_sharpe):
            return False, f"sharpe={snap.live_sharpe:.2f}<{self.config.paper_min_sharpe}"
        if snap.drawdown < float(self.config.paper_max_drawdown):
            return False, f"dd={snap.drawdown:.1%}<{self.config.paper_max_drawdown:.0%}"
        if snap.n_trades < int(self.config.paper_min_trades):
            return False, f"trades={snap.n_trades}<{self.config.paper_min_trades}"
        return True, "paper_gate_pass"
