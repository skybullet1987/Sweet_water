from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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
    from pathlib import Path

    p = Path(__file__).resolve().parent / str(config.validation_report_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
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
