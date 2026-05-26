#!/usr/bin/env python3
"""Update test/research imports after Kraken Max consolidation."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REPLACEMENTS = [
    (r"from features import", "from core import"),
    (r"from universe import", "from core import"),
    (r"from correlation import", "from core import"),
    (r"from scalper_sleeve import", "from core import"),
    (r"from sizing import", "from core import"),
    (r"from ensemble import", "from core import"),
    (r"from advanced_regime import", "from regime import"),
    (r"from regime_bridge import", "from regime import"),
    (r"from regime_ensemble import", "from regime import"),
    (r"from cluster_risk import", "from risk import"),
    (r"from portfolio_optimizer import", "from risk import"),
    (r"from data_feeds import", "from data import"),
    (r"from sentiment import", "from data import"),
    (r"from cross_venue import", "from data import"),
    (r"from ml_scorer import", "from ml import"),
    (r"from ml_trainer import", "from ml import"),
    (r"from fill_tracker import", "from ops import"),
    (r"from drift_monitor import", "from ops import"),
    (r"from cost_model import", "from ops import"),
    (r"from scorecard import", "from ops import"),
    (r"from telemetry import", "from ops import"),
    (r"from dashboard_digest import", "from ops import"),
    (r"from notifications import", "from ops import"),
    (r"from baseline_manager import", "from ops import"),
    (r"from backtest_validator import", "from workflow import"),
    (r"from walk_forward_engine import", "from workflow import"),
    (r"from bars_util import", "from workflow import"),
    (r"from auto_revalidation import", "from workflow import"),
    (r"from regime_walk_forward import", "from workflow import"),
    (r"from execution_bridge import", "from execution import"),
    (r"from brackets import", "from execution import"),
]

PATHS = list((REPO / "tests").glob("test_kraken_max*.py")) + list((REPO / "Kraken Max" / "research").glob("*.py"))


def update_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    for p in PATHS:
        if update_file(p):
            print(f"updated {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
