from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nextgen.core.models import Bar
from nextgen.research.harness import BarReplayHarness, ExperimentConfig
from regime.hmm import HMMRegime
from config.strategy_config import StrategyConfig


def _parse_ts(v: str) -> datetime:
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    return datetime.fromisoformat(v)


def _load_bars(csv_path: Path) -> dict[str, list[Bar]]:
    bars_by_symbol: dict[str, list[Bar]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            symbol = row["symbol"]
            bars_by_symbol.setdefault(symbol, []).append(
                Bar(
                    symbol=symbol,
                    timestamp=_parse_ts(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    for bars in bars_by_symbol.values():
        bars.sort(key=lambda b: b.timestamp)
    return bars_by_symbol


def _deterministic_fixture() -> dict[str, list[Bar]]:
    np.random.seed(42)
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD"]
    bars_by_symbol: dict[str, list[Bar]] = {s: [] for s in symbols}
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    n = 5000

    base = {"BTCUSD": 40000.0, "ETHUSD": 2200.0, "SOLUSD": 90.0}
    for i in range(n):
        if i < 1000:
            drift, vol = -0.0005, 0.012
        elif i < 2000:
            drift, vol = 0.0, 0.009
        else:
            drift, vol = 0.0008, 0.003
        for sym in symbols:
            scale = 1.0 + (0.3 if sym == "SOLUSD" else (0.1 if sym == "ETHUSD" else 0.0))
            ret = np.random.normal(loc=drift, scale=vol * scale)
            close = max(0.01, base[sym] * (1.0 + ret))
            high = close * (1.0 + abs(np.random.normal(0.0008, 0.0004)))
            low = close * (1.0 - abs(np.random.normal(0.0008, 0.0004)))
            open_ = base[sym]
            volume = float(abs(np.random.normal(1000.0, 150.0)) * (1.5 if sym == "BTCUSD" else 1.0))
            bars_by_symbol[sym].append(
                Bar(symbol=sym, timestamp=t, open=open_, high=high, low=low, close=close, volume=volume)
            )
            base[sym] = close
        t += timedelta(hours=1)
    return bars_by_symbol


def _avg_win_loss_from_events(events) -> tuple[float, float]:
    open_by_symbol = {}
    pnls = []
    for e in events:
        if e.action == "BUY":
            open_by_symbol[e.symbol] = e.price
        elif e.action == "SELL" and e.symbol in open_by_symbol:
            entry = open_by_symbol.pop(e.symbol)
            pnls.append((e.price - entry) / entry)
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return avg_win, avg_loss


def _regime_distribution(bars_by_symbol: dict[str, list[Bar]]) -> dict[str, float]:
    btc = bars_by_symbol["BTCUSD"]
    cfg = StrategyConfig(hmm_train_window_bars=4320, hmm_retrain_every_bars=720)
    regime = HMMRegime(cfg)

    btc_df = pd.DataFrame(
        {
            "close": [b.close for b in btc],
            "timestamp": [b.timestamp for b in btc],
        }
    )
    btc_df["ret"] = np.log(btc_df["close"]).diff().fillna(0.0)
    btc_df["rv24"] = btc_df["ret"].rolling(24, min_periods=24).std().fillna(0.0)
    ema50 = btc_df["close"].ewm(span=50, adjust=False).mean()
    breadth = (btc_df["close"] > ema50).astype(float)

    counts = {"risk_on": 0.0, "risk_off": 0.0, "chop": 0.0}
    for _, row in btc_df.iterrows():
        regime.update(float(row["ret"]), float(row["rv24"]), float(breadth.loc[row.name]))
        probs = regime.current_state_probs()
        for key in counts:
            counts[key] += float(probs.get(key, 0.0))
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def main() -> int:
    fixture_csv = REPO_ROOT / "tests" / "fixtures" / "walk_forward_bars.csv"
    baseline_json = REPO_ROOT / "tests" / "fixtures" / "walk_forward_baseline.json"

    bars_by_symbol = _load_bars(fixture_csv)
    if not bars_by_symbol or len(next(iter(bars_by_symbol.values()))) < 4500:
        bars_by_symbol = _deterministic_fixture()

    symbols = tuple(sorted(bars_by_symbol.keys()))
    all_bars = sorted((b for bars in bars_by_symbol.values() for b in bars), key=lambda b: b.timestamp)
    split_index = int(len(all_bars) * 0.60)
    split_time = all_bars[split_index].timestamp

    harness = BarReplayHarness(fee_rate=0.001, slippage_rate=0.001, paper_trade_mode=True)

    oos_cfg = ExperimentConfig(
        name="wf_ci_oos",
        start=split_time,
        end=all_bars[-1].timestamp,
        symbols=symbols,
        initial_cash=10_000.0,
    )
    oos_wf = harness.walk_forward_run(oos_cfg, bars_by_symbol, n_folds=3)
    _ = harness.run(oos_cfg, bars_by_symbol)
    session = harness.last_paper_session

    trade_count = session.buy_count()
    total_orders = len(session.events)
    cancel_count = sum(1 for e in session.events if getattr(e, "action", "") == "CANCEL")
    cancel_rate = (cancel_count / total_orders) if total_orders > 0 else 0.0
    avg_win, avg_loss = _avg_win_loss_from_events(session.events)
    ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    regime_dist = _regime_distribution(bars_by_symbol)
    active_regimes = [k for k, v in regime_dist.items() if v >= 0.05]

    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    print(f"OOS Sharpe: {oos_wf.overall_sharpe:.4f}")
    print(f"OOS trade count: {trade_count}")
    print(f"OOS avg_win/avg_loss: {ratio:.4f}")
    print(f"OOS cancel rate: {cancel_rate:.4f}")
    print(f"Regime distribution: {regime_dist}")

    if oos_wf.overall_sharpe < 0.5:
        raise SystemExit(f"OOS Sharpe {oos_wf.overall_sharpe:.4f} is below 0.5 threshold")
    if ratio < 1.0:
        raise SystemExit(f"OOS avg_win/avg_loss {ratio:.4f} is below 1.0")
    if not (15 <= trade_count <= 200):
        raise SystemExit(f"OOS trade count {trade_count} outside [15, 200]")
    if cancel_rate >= 0.20:
        raise SystemExit(f"OOS cancel rate {cancel_rate:.4f} exceeds 0.20")

    if regime_dist.get("chop", 0.0) < 0.05:
        if len([k for k in ("risk_on", "risk_off") if regime_dist.get(k, 0.0) >= 0.05]) < 2:
            raise SystemExit("Regime distribution is degenerate for 2-state fallback")
    else:
        if len(active_regimes) < 3:
            raise SystemExit("Regime distribution is degenerate for 3-state HMM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
