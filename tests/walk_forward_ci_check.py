from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nextgen.core.models import Bar
from nextgen.research.harness import BarReplayHarness, ExperimentConfig


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


def main() -> int:
    repo_root = REPO_ROOT
    fixture_csv = repo_root / "tests" / "fixtures" / "walk_forward_bars.csv"
    baseline_json = repo_root / "tests" / "fixtures" / "walk_forward_baseline.json"

    bars_by_symbol = _load_bars(fixture_csv)
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
    oos_wf = harness.walk_forward_run(oos_cfg, bars_by_symbol, n_folds=2)
    oos_run = harness.run(oos_cfg, bars_by_symbol)
    session = harness.last_paper_session
    oos_trade_count = session.buy_count()
    total_orders = len(session.events)
    cancel_count = sum(1 for e in session.events if getattr(e, "action", "") == "CANCEL")
    cancel_rate = (cancel_count / total_orders) if total_orders > 0 else 0.0

    baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
    baseline_trades = float(baseline["oos_trade_count"])
    min_allowed_trades = baseline_trades * 0.5

    print(f"OOS Sharpe: {oos_wf.overall_sharpe:.4f}")
    print(f"OOS trade count: {oos_trade_count}")
    print(f"OOS total return: {oos_run.metrics['total_return']:.4f}")
    print(f"OOS cancel rate: {cancel_rate:.4f}")

    if oos_wf.overall_sharpe < 0.3:
        raise SystemExit(f"OOS Sharpe {oos_wf.overall_sharpe:.4f} is below 0.3 threshold")
    if oos_trade_count < min_allowed_trades:
        raise SystemExit(
            f"OOS trade count {oos_trade_count} dropped by >50% vs baseline {baseline_trades}"
        )
    if cancel_rate > 0.20:
        raise SystemExit(f"OOS cancel rate {cancel_rate:.4f} exceeds 0.20 regression guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
