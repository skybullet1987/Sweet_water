#!/usr/bin/env python3
"""Merge Kraken Max modules into consolidated files."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "Kraken Max"
BACKUP = ROOT / "_merge_backup"
QC_LIMIT = 63_000

MERGES: dict[str, list[str]] = {
    "data.py": ["data_feeds.py", "sentiment.py", "cross_venue.py"],
    "ml.py": ["ml_scorer.py", "ml_trainer.py"],
    "regime.py": ["regime.py", "advanced_regime.py", "regime_bridge.py", "regime_ensemble.py"],
    "core.py": [
        "features.py",
        "universe.py",
        "correlation.py",
        "scalper_sleeve.py",
        "sizing.py",
        "ensemble.py",
    ],
    "risk.py": ["risk.py", "cluster_risk.py", "portfolio_optimizer.py"],
    "execution.py": ["execution.py", "execution_bridge.py", "brackets.py"],
    "ops.py": [
        "fill_tracker.py",
        "drift_monitor.py",
        "cost_model.py",
        "scorecard.py",
        "telemetry.py",
        "dashboard_digest.py",
        "notifications.py",
        "baseline_manager.py",
    ],
    "workflow.py": [
        "bars_util.py",
        "walk_forward_engine.py",
        "backtest_validator.py",
        "regime_walk_forward.py",
        "auto_revalidation.py",
    ],
}

REMOVE = {s for files in MERGES.values() for s in files} - {"main.py", "config.py"}


def strip_module_preamble(text: str) -> str:
    lines = text.splitlines()
    i = 0
    if lines and lines[0].startswith("from __future__"):
        i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip().startswith(('"""', "'''")):
        q = lines[i].strip()[:3]
        if lines[i].strip().count(q) >= 2:
            i += 1
        else:
            i += 1
            while i < len(lines) and q not in lines[i]:
                i += 1
            i += 1
    while i < len(lines) and (lines[i].startswith("import ") or lines[i].startswith("from ")):
        i += 1
    return "\n".join(lines[i:])


def patch_body(src: str, body: str) -> str:
    if src == "regime.py":
        body = re.sub(r"from sentiment import.*\n", "", body, count=1)
    if src == "sentiment.py":
        body = re.sub(r"try:[\s\S]*?ExternalSentiment.*?\n\n", "", body, count=1)
    if src == "ensemble.py":
        body = re.sub(r"from ml_scorer import.*\n", "", body)
        body = re.sub(r"from regime_ensemble import.*\n", "", body)
        body = re.sub(r"try:[\s\S]*?load_regime_weights_merged[\s\S]*?\n\n", "", body)
    if src == "advanced_regime.py":
        body = re.sub(r"from regime import.*\n", "", body)
        body = re.sub(r"from sentiment import.*\n", "", body)
    if src == "regime_bridge.py":
        body = re.sub(r"from advanced_regime import.*\n", "", body)
        body = re.sub(r"from regime import.*\n", "", body)
        body = re.sub(r"from sentiment import.*\n", "", body)
    if src == "portfolio_optimizer.py":
        body = body.replace("from correlation import hourly_returns", "# hourly_returns from core below")
    if src == "telemetry.py":
        body = body.replace(
            "from cluster_risk import max_cluster_exposure",
            "# max_cluster_exposure defined in this package risk.py",
        )
        body = body.replace(
            "from cluster_risk import max_cluster_exposure",
            "from risk import max_cluster_exposure",
        )
    if src == "sizing.py":
        body = body.replace("from cost_model import CalibratedCostModel", "from ops import CalibratedCostModel")
    if src == "auto_revalidation.py":
        for pat in (
            r"from backtest_validator import.*\n",
            r"from baseline_manager import.*\n",
            r"from regime_walk_forward import.*\n",
            r"from walk_forward_engine import.*\n",
        ):
            body = re.sub(pat, "", body)
    if src == "regime_walk_forward.py":
        body = body.replace("from ensemble import AlphaEnsemble", "from core import AlphaEnsemble")
        body = re.sub(r"from regime_ensemble import.*\n", "", body)
        body = re.sub(r"from walk_forward_engine import.*\n", "", body)
    if src == "walk_forward_engine.py":
        body = body.replace("from ensemble import AlphaEnsemble", "from core import AlphaEnsemble")
        body = body.replace("from ml_scorer import MLScorer", "from ml import MLScorer")
    if src == "backtest_validator.py":
        body = re.sub(r"from walk_forward_engine import.*\n", "", body)
    if src == "baseline_manager.py":
        body = re.sub(r"from drift_monitor import.*\n", "", body)
        body = re.sub(r"from walk_forward_engine import.*\n", "", body)
    if src == "execution.py":
        body = body.replace("from scalper_sleeve import", "from core import")
        body = body.replace("from sizing import", "from core import")
    return body


def header_for(name: str) -> list[str]:
    h = [
        f'"""Kraken Max — consolidated `{name}` (QC deploy file; keep under {QC_LIMIT:,} chars)."""',
        "from __future__ import annotations",
        "",
    ]
    if name == "data.py":
        h += ["from config import CONFIG, KrakenMaxConfig", ""]
    elif name == "ml.py":
        h += ["from config import CONFIG, KrakenMaxConfig", ""]
    elif name == "regime.py":
        h += [
            "from config import CONFIG, KrakenMaxConfig",
            "from data import SentimentSnapshot, adjust_deployment_cap",
            "",
        ]
    elif name == "core.py":
        h += [
            "from config import CONFIG, KrakenMaxConfig",
            "from ml import MLScorer, load_ml_weights",
            "from regime import (",
            "    config_for_regime,",
            "    load_regime_weights,",
            "    load_regime_weights_from_object_store,",
            "    load_regime_weights_merged,",
            ")",
            "",
        ]
    elif name == "risk.py":
        h += ["from config import CONFIG, KrakenMaxConfig", ""]
    elif name == "execution.py":
        h += ["from config import CONFIG", ""]
    elif name == "ops.py":
        h += ["from config import CONFIG, KrakenMaxConfig", ""]
    elif name == "workflow.py":
        h += [
            "from config import CONFIG, KrakenMaxConfig",
            "from ops import BaselineManager, DriftMonitor",
            "from core import AlphaEnsemble",
            "from ml import MLScorer",
            "from regime import _DEFAULT_REGIME_MAP, _WEIGHT_KEYS, load_regime_weights, normalize_regime_key",
            "",
        ]
    return h


def merge_file(name: str, sources: list[str], cache: dict[str, str]) -> str:
    chunks = header_for(name)
    for src in sources:
        body = patch_body(src, strip_module_preamble(cache[src]))
        chunks.append(f"\n# --- from {src} ---\n")
        chunks.append(body)
    text = "\n".join(chunks)
    if name == "risk.py":
        text = text.replace(
            "# hourly_returns from core below",
            "from core import hourly_returns  # portfolio_optimizer",
            1,
        )
    if name == "ops.py" and "from risk import max_cluster_exposure" not in text:
        text = text.replace(
            "def build(",
            "from risk import max_cluster_exposure  # telemetry\n\n\ndef build(",
            1,
        )
    return text


def main() -> None:
    cache = {p.name: p.read_text(encoding="utf-8") for p in ROOT.glob("*.py")}
    if not BACKUP.exists():
        BACKUP.mkdir()
        for p in ROOT.glob("*.py"):
            if p.name in cache:
                shutil.copy2(p, BACKUP / p.name)
    for name, sources in MERGES.items():
        merged = merge_file(name, sources, cache)
        if len(merged) > QC_LIMIT:
            print(f"WARNING {name}: {len(merged)} chars > {QC_LIMIT}")
        (ROOT / name).write_text(merged, encoding="utf-8")
        print(f"wrote {name}: {len(merged)} chars")
    for s in sorted(REMOVE):
        p = ROOT / s
        if p.exists() and s not in MERGES:
            p.unlink()
            print(f"removed {s}")


if __name__ == "__main__":
    main()
