"""Kraken Max — monitoring, alerts, scorecard (`kraken_ops.py`)."""
from __future__ import annotations

import json
import math
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CONFIG, KrakenMaxConfig
from risk import max_cluster_exposure

# --- from fill_tracker.py ---


from config import CONFIG, KrakenMaxConfig


@dataclass
class FillStats:
    limits_submitted: int = 0
    limits_filled: int = 0
    market_filled: int = 0
    cancelled: int = 0
    slippage_bps: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def fill_rate(self) -> float:
        """Filled limits / non-cancelled submissions (cancels are intentional, not failed fills)."""
        attempted = self.limits_submitted - self.cancelled
        if attempted <= 0:
            return 1.0
        return float(self.limits_filled) / float(attempted)

    @property
    def avg_slippage_bps(self) -> float:
        if not self.slippage_bps:
            return 0.0
        return float(sum(self.slippage_bps)) / len(self.slippage_bps)


class FillTracker:
    """Track limit fill quality vs expected prices (v5)."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.stats = FillStats()
        self._pending: dict[int, dict] = {}

    def on_submit(self, order_id: int, *, is_limit: bool, expected_price: float, qty: float) -> None:
        if not is_limit:
            return
        self.stats.limits_submitted += 1
        self._pending[int(order_id)] = {
            "expected_price": float(expected_price),
            "qty": float(qty),
        }

    def on_fill(self, order_id: int, fill_price: float, *, is_limit: bool, tag: str = "") -> None:
        if is_limit:
            self.stats.limits_filled += 1
        else:
            self.stats.market_filled += 1
        meta = self._pending.pop(int(order_id), None)
        if meta and meta["expected_price"] > 0:
            exp = float(meta["expected_price"])
            slip_bps = abs(float(fill_price) / exp - 1.0) * 10_000.0
            self.stats.slippage_bps.append(slip_bps)

    def on_cancel(self, order_id: int) -> None:
        self.stats.cancelled += 1
        self._pending.pop(int(order_id), None)

    def should_alert(self) -> tuple[bool, str]:
        if self.stats.limits_submitted < 5:
            return False, ""
        if self.stats.fill_rate < float(self.config.fill_rate_alert_threshold):
            return True, f"low_fill_rate={self.stats.fill_rate:.2%}"
        if self.stats.avg_slippage_bps > float(self.config.slippage_alert_bps):
            return True, f"high_slippage_bps={self.stats.avg_slippage_bps:.1f}"
        return False, ""

    def summary(self) -> dict[str, float]:
        return {
            "fill_rate": self.stats.fill_rate,
            "avg_slippage_bps": self.stats.avg_slippage_bps,
            "limits_submitted": float(self.stats.limits_submitted),
            "limits_filled": float(self.stats.limits_filled),
        }

# --- from drift_monitor.py ---


from config import CONFIG, KrakenMaxConfig


@dataclass
class DriftSnapshot:
    live_sharpe: float
    baseline_sharpe: float
    ratio: float
    n_hours: int


class DriftMonitor:
    """Compare rolling live Sharpe to walk-forward baseline (v5)."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.baseline_sharpe = float(config.baseline_sharpe)
        self._equity: deque[tuple[datetime, float]] = deque(maxlen=config.drift_window_hours * 4)

    def load_baseline_from_object_store(self, algo) -> None:
        key = str(self.config.drift_object_store_key)
        try:
            if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
                blob = json.loads(algo.ObjectStore.Read(key))
                self.baseline_sharpe = float(blob.get("oos_sharpe", self.baseline_sharpe))
        except Exception:
            return

    def save_baseline_to_object_store(self, algo, sharpe: float) -> None:
        key = str(self.config.drift_object_store_key)
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps({"oos_sharpe": float(sharpe)}))
                self.baseline_sharpe = float(sharpe)
        except Exception:
            return

    def record_equity(self, when: datetime, equity: float) -> None:
        self._equity.append((when, float(equity)))

    def _hourly_returns(self) -> list[float]:
        if len(self._equity) < 3:
            return []
        points = list(self._equity)
        rets: list[float] = []
        for i in range(1, len(points)):
            prev = points[i - 1][1]
            cur = points[i][1]
            if prev > 0:
                rets.append(cur / prev - 1.0)
        return rets

    def live_sharpe(self, periods_per_year: float | None = None) -> float:
        rets = self._hourly_returns()
        if len(rets) < 8:
            return 0.0
        ppy = periods_per_year
        if ppy is None:
            ppy = 24 * 365 / max(1, self.config.resolution_minutes) * (60 / max(1, self.config.resolution_minutes))
            ppy = 24 * 365 * self.config.bph()
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)
        sd = math.sqrt(max(var, 1e-12))
        if sd <= 0:
            return 0.0
        return (mu / sd) * math.sqrt(ppy)

    def evaluate(self) -> DriftSnapshot:
        live = self.live_sharpe()
        base = max(self.baseline_sharpe, 1e-6)
        ratio = live / base if base > 0 else 0.0
        return DriftSnapshot(
            live_sharpe=live,
            baseline_sharpe=base,
            ratio=ratio,
            n_hours=len(self._equity),
        )

    def should_alert(self) -> tuple[bool, str]:
        snap = self.evaluate()
        if snap.n_hours < 24:
            return False, ""
        if snap.baseline_sharpe <= 0:
            return False, ""
        if snap.ratio < float(self.config.drift_sharpe_ratio_threshold):
            return (
                True,
                f"drift live_sharpe={snap.live_sharpe:.2f} baseline={snap.baseline_sharpe:.2f} "
                f"ratio={snap.ratio:.2f}",
            )
        return False, ""

# --- from cost_model.py ---


from config import CONFIG, KrakenMaxConfig

BPS = 10_000.0


@dataclass(frozen=True)
class CostBreakdown:
    fee_pct: float
    spread_pct: float
    slippage_pct: float

    @property
    def total_pct(self) -> float:
        return self.fee_pct + self.spread_pct + self.slippage_pct


class CalibratedCostModel:
    """v7: round-trip cost from FillTracker live stats with config fallbacks."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def from_fill_tracker(self, fill_tracker) -> CostBreakdown:
        fee = float(self.config.expected_round_trip_fees)
        spread = float(self.config.assumed_spread_bps) / BPS
        slip = float(self.config.assumed_slippage_bps) / BPS
        if fill_tracker is None:
            return CostBreakdown(fee_pct=fee, spread_pct=spread, slippage_pct=slip)
        stats = fill_tracker.stats
        n = int(stats.limits_filled) + int(stats.market_filled)
        if n >= int(self.config.cost_calibration_min_fills):
            slip = max(slip, float(stats.avg_slippage_bps) / BPS)
            fill_rate = float(stats.fill_rate)
            if fill_rate < 0.7:
                spread *= 1.0 + (0.7 - fill_rate)
        return CostBreakdown(fee_pct=fee, spread_pct=spread, slippage_pct=slip)

    def round_trip_pct(self, algo) -> float:
        ft = getattr(algo, "fill_tracker", None)
        return self.from_fill_tracker(ft).total_pct

    def passes_edge_gate(self, score: float, notional: float, algo) -> bool:
        if score <= 0 or notional <= 0:
            return False
        cost_pct = self.round_trip_pct(algo)
        edge = score * float(self.config.edge_scale)
        return edge > cost_pct * float(self.config.edge_cost_multiplier)

# --- from scorecard.py ---


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

# --- from telemetry.py ---


from config import CONFIG, KrakenMaxConfig


@dataclass
class TelemetrySnapshot:
    """Paper/live dashboard payload (v6) — persisted to QC ObjectStore."""

    ts: str
    equity: float
    regime: str
    micro_regime: str
    deployment_cap: float
    fill_rate: float
    avg_slippage_bps: float
    live_sharpe: float
    baseline_sharpe: float
    drift_ratio: float
    cross_venue_boost: float
    scorecard_sharpe: float = 0.0
    scorecard_win_rate: float = 0.0
    scorecard_trades: int = 0
    cluster_exposure: dict[str, float] = field(default_factory=dict)
    active_universe: list[str] = field(default_factory=list)
    erc_weights: dict[str, float] = field(default_factory=dict)
    version: str = "v7"


class TelemetryDashboard:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def build(
        self,
        algo,
        *,
        regime_name: str = "",
        micro_regime: str = "",
        deployment_cap: float = 0.0,
        cross_venue_boost: float = 0.0,
    ) -> TelemetrySnapshot:
        equity = float(getattr(algo.Portfolio, "TotalPortfolioValue", 0.0) or 0.0)
        fill_rate = 1.0
        slip_bps = 0.0
        ft = getattr(algo, "fill_tracker", None)
        if ft is not None:
            fill_rate = float(ft.stats.fill_rate)
            slip_bps = float(ft.stats.avg_slippage_bps)
        live_sh = 0.0
        base_sh = float(self.config.baseline_sharpe)
        drift_ratio = 0.0
        dm = getattr(algo, "drift_monitor", None)
        if dm is not None:
            snap = dm.evaluate()
            live_sh = float(snap.live_sharpe)
            base_sh = float(snap.baseline_sharpe)
            drift_ratio = float(snap.ratio)
        now = getattr(algo, "Time", datetime.now(timezone.utc))
        ts = now.isoformat() if hasattr(now, "isoformat") else str(now)
        sc_sharpe = 0.0
        sc_wr = 0.0
        sc_trades = 0
        sc = getattr(algo, "scorecard", None)
        if sc is not None and sc._trades:
            snap_sc = sc.build(algo, fill_tracker=ft)
            sc_sharpe = float(snap_sc.live_sharpe)
            sc_wr = float(snap_sc.win_rate)
            sc_trades = int(snap_sc.n_trades)
        universe = list(getattr(algo, "active_universe", []) or [])
        erc = {k: float(v) for k, v in (getattr(algo, "_erc_weights", {}) or {}).items()}
        clusters = {}
        if erc:
            try:
                clusters = max_cluster_exposure(erc)
            except Exception:
                clusters = {}
        return TelemetrySnapshot(
            ts=ts,
            equity=equity,
            regime=str(regime_name),
            micro_regime=str(micro_regime),
            deployment_cap=float(deployment_cap),
            fill_rate=fill_rate,
            avg_slippage_bps=slip_bps,
            live_sharpe=live_sh,
            baseline_sharpe=base_sh,
            drift_ratio=drift_ratio,
            cross_venue_boost=float(cross_venue_boost),
            scorecard_sharpe=sc_sharpe,
            scorecard_win_rate=sc_wr,
            scorecard_trades=sc_trades,
            cluster_exposure=clusters,
            active_universe=universe,
            erc_weights=erc,
        )

    def persist(self, algo, snapshot: TelemetrySnapshot) -> None:
        key = str(self.config.telemetry_object_store_key)
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps(asdict(snapshot), indent=2))
        except Exception:
            return

    def load_latest(self, algo) -> dict[str, Any]:
        key = str(self.config.telemetry_object_store_key)
        try:
            if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
                return json.loads(algo.ObjectStore.Read(key))
        except Exception:
            pass
        return {}

    def summary_line(self, snapshot: TelemetrySnapshot) -> str:
        return (
            f"KM_TELEM eq={snapshot.equity:.0f} reg={snapshot.regime} "
            f"fill={snapshot.fill_rate:.0%} slip={snapshot.avg_slippage_bps:.0f}bps "
            f"sharpe={snapshot.live_sharpe:.2f}/{snapshot.baseline_sharpe:.2f} "
            f"drift={snapshot.drift_ratio:.2f} xvenue={snapshot.cross_venue_boost:+.3f}"
        )

# --- from dashboard_digest.py ---


from config import CONFIG, KrakenMaxConfig


@dataclass
class DigestBundle:
    telemetry: dict[str, Any]
    scorecard: dict[str, Any]
    validation: dict[str, Any]
    revalidation: dict[str, Any]


def load_json_store(algo, key: str) -> dict[str, Any]:
    try:
        if hasattr(algo, "ObjectStore") and algo.ObjectStore.ContainsKey(key):
            return json.loads(algo.ObjectStore.Read(key))
    except Exception:
        pass
    return {}


def load_bundle(algo, config: KrakenMaxConfig = CONFIG) -> DigestBundle:
    return DigestBundle(
        telemetry=load_json_store(algo, str(config.telemetry_object_store_key)),
        scorecard=load_json_store(algo, str(config.scorecard_object_store_key)),
        validation=_load_local_validation(config),
        revalidation=load_json_store(algo, str(config.revalidation_object_store_key)),
    )


def _load_local_validation(config: KrakenMaxConfig) -> dict[str, Any]:
    _ = config
    return {}


def build_text_digest(bundle: DigestBundle) -> str:
    t = bundle.telemetry
    s = bundle.scorecard
    v = bundle.validation
    r = bundle.revalidation
    lines = [
        "Kraken Max Dashboard",
        "────────────────────",
        f"Equity: ${t.get('equity', 0):,.0f}  Regime: {t.get('regime', '?')} ({t.get('micro_regime', '')})",
        f"Deploy cap: {float(t.get('deployment_cap', 0)):.0%}  Fill: {float(t.get('fill_rate', 1)):.0%}  Slip: {float(t.get('avg_slippage_bps', 0)):.0f} bps",
        f"Drift Sharpe: {float(t.get('live_sharpe', 0)):.2f} / {float(t.get('baseline_sharpe', 0)):.2f} (ratio {float(t.get('drift_ratio', 0)):.2f})",
    ]
    if s:
        lines.append(
            f"Scorecard: Sharpe {float(s.get('live_sharpe', 0)):.2f}  WR {float(s.get('win_rate', 0)):.0%}  "
            f"PF {float(s.get('profit_factor', 0)):.2f}  Trades {int(s.get('n_trades', 0))}  DD {float(s.get('drawdown', 0)):.1%}"
        )
    if v:
        status = "PASS" if v.get("passed") else "FAIL"
        lines.append(
            f"Validation: {status}  OOS Sharpe {float(v.get('oos_sharpe', 0)):.2f}  "
            f"MaxDD {float(v.get('oos_max_drawdown', 0)):.1%}"
        )
    if r:
        lines.append(f"Last reval OOS Sharpe: {float(r.get('oos_sharpe', 0)):.2f}")
    clusters = t.get("cluster_exposure") or {}
    if clusters:
        parts = [f"{k}={float(v):.0%}" for k, v in clusters.items()]
        lines.append("Clusters: " + ", ".join(parts))
    return "\n".join(lines)


def build_html_digest(bundle: DigestBundle) -> str:
    t = bundle.telemetry
    s = bundle.scorecard
    v = bundle.validation
    passed = v.get("passed", None)
    badge = "pass" if passed else "fail" if passed is False else "na"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Kraken Max</title>
<style>
body{{font-family:system-ui;background:#0d1117;color:#e6edf3;padding:1.5rem}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:1rem;margin:.75rem 0}}
.badge{{display:inline-block;padding:.2rem .6rem;border-radius:4px;font-size:.85rem}}
.pass{{background:#238636}}.fail{{background:#da3633}}.na{{background:#6e7681}}
h1{{font-size:1.25rem}} table{{width:100%;border-collapse:collapse}} td{{padding:.35rem 0}}
</style></head><body>
<h1>Kraken Max v8 Dashboard</h1>
<div class="card"><span class="badge {badge}">validation</span>
<table>
<tr><td>Equity</td><td>${float(t.get('equity',0)):,.0f}</td></tr>
<tr><td>Regime</td><td>{t.get('regime','?')} / {t.get('micro_regime','')}</td></tr>
<tr><td>Fill rate</td><td>{float(t.get('fill_rate',1)):.0%}</td></tr>
<tr><td>Slippage</td><td>{float(t.get('avg_slippage_bps',0)):.0f} bps</td></tr>
<tr><td>Drift Sharpe</td><td>{float(t.get('live_sharpe',0)):.2f} / {float(t.get('baseline_sharpe',0)):.2f}</td></tr>
</table></div>
<div class="card"><h2>Paper scorecard</h2><table>
<tr><td>Live Sharpe</td><td>{float(s.get('live_sharpe',0)):.2f}</td></tr>
<tr><td>Win rate</td><td>{float(s.get('win_rate',0)):.0%}</td></tr>
<tr><td>Profit factor</td><td>{float(s.get('profit_factor',0)):.2f}</td></tr>
<tr><td>Trades</td><td>{int(s.get('n_trades',0))}</td></tr>
<tr><td>Drawdown</td><td>{float(s.get('drawdown',0)):.1%}</td></tr>
</table></div>
</body></html>"""


def persist_digest(algo, text: str, html: str, config: KrakenMaxConfig = CONFIG) -> None:
    if not hasattr(algo, "ObjectStore"):
        return
    try:
        algo.ObjectStore.Save(str(config.dashboard_text_key), text)
        algo.ObjectStore.Save(str(config.dashboard_html_key), html)
    except Exception:
        return

# --- from notifications.py ---



def _post_json(url: str, payload: dict[str, Any]) -> bool:
    if not url or not url.startswith("http"):
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "KrakenMax/4.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def send_telegram(webhook_url: str, message: str) -> bool:
    """Telegram bot API: pass full URL like https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<ID> as webhook_url with POST body, or use bot token + chat in algo params."""
    if "api.telegram.org" in webhook_url and "sendMessage" in webhook_url:
        return _post_json(webhook_url, {"text": message[:4000]})
    return _post_json(webhook_url, {"message": message[:4000]})


def send_discord(webhook_url: str, message: str) -> bool:
    return _post_json(webhook_url, {"content": message[:2000]})


class AlertManager:
    def __init__(self, algo) -> None:
        self.algo = algo
        self.telegram_url = ""
        self.discord_url = ""
        self._last_alert_key = ""
        try:
            self.telegram_url = str(algo.GetParameter("telegram_webhook") or "")
        except Exception:
            self.telegram_url = ""
        try:
            self.discord_url = str(algo.GetParameter("discord_webhook") or "")
        except Exception:
            self.discord_url = ""

    def notify(self, event: str, detail: str, *, dedupe_key: str | None = None) -> None:
        if dedupe_key and dedupe_key == self._last_alert_key:
            return
        self._last_alert_key = dedupe_key or ""
        msg = f"[Kraken Max] {event}\n{detail}"
        if hasattr(self.algo, "Debug"):
            self.algo.Debug(msg[:200])
        if self.telegram_url:
            send_telegram(self.telegram_url, msg)
        if self.discord_url:
            send_discord(self.discord_url, msg)

# --- from baseline_manager.py ---


from config import CONFIG, KrakenMaxConfig


class BaselineManager:
    """Auto-refresh walk-forward baseline Sharpe into ObjectStore + drift monitor (v6)."""

    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config

    def apply_walk_forward_result(
        self,
        algo,
        result,
        *,
        drift: DriftMonitor | None = None,
        persist_ensemble: bool = True,
    ) -> float:
        sharpe = float(result.oos_sharpe)
        dm = drift or getattr(algo, "drift_monitor", None)
        if dm is not None:
            dm.baseline_sharpe = sharpe
            dm.save_baseline_to_object_store(algo, sharpe)
        _ = persist_ensemble
        self._save_meta(algo, sharpe, source="walk_forward", extra={"weights": result.best_weights})
        return sharpe

    def refresh_from_sharpe(
        self,
        algo,
        sharpe: float,
        *,
        source: str = "ml_retrain",
        drift: DriftMonitor | None = None,
    ) -> None:
        dm = drift or getattr(algo, "drift_monitor", None)
        if dm is None:
            return
        if not bool(self.config.auto_refresh_baseline):
            return
        dm.baseline_sharpe = float(sharpe)
        dm.save_baseline_to_object_store(algo, float(sharpe))
        self._save_meta(algo, float(sharpe), source=source)

    def _save_meta(self, algo, sharpe: float, *, source: str, extra: dict[str, Any] | None = None) -> None:
        key = str(self.config.baseline_meta_object_store_key)
        payload = {
            "oos_sharpe": float(sharpe),
            "source": source,
            **(extra or {}),
        }
        try:
            if hasattr(algo, "ObjectStore"):
                algo.ObjectStore.Save(key, json.dumps(payload, indent=2))
        except Exception:
            return