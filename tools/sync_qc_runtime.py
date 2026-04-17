#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


SYNC_WHITELIST = (
    "main.py",
    "execution.py",
    "reporting.py",
    "order_management.py",
    "realistic_slippage.py",
    "events.py",
    "scoring.py",
    "strategy_core.py",
    "trade_quality.py",
    "fee_model.py",
    "regime_router.py",
    "chop_engine.py",
    "entry_exec.py",
    "alt_data.py",
    "nextgen/__init__.py",
    "nextgen/core/__init__.py",
    "nextgen/core/models.py",
    "nextgen/risk/__init__.py",
    "nextgen/risk/engine.py",
)


def sync(check_only: bool) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = repo_root / "qc_runtime"
    out_of_sync: list[str] = []

    for rel in SYNC_WHITELIST:
        src = repo_root / rel
        dst = runtime_root / rel
        if not src.exists():
            out_of_sync.append(rel)
            continue
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            out_of_sync.append(rel)
            if not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    if check_only and out_of_sync:
        print("qc_runtime out of sync with root for:")
        for rel in out_of_sync:
            print(f" - {rel}")
        return 1

    if not check_only:
        print(f"Synced {len(out_of_sync)} file(s).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync whitelisted root files into qc_runtime/")
    parser.add_argument("--check", action="store_true", help="Fail if qc_runtime is out of sync")
    args = parser.parse_args()
    raise SystemExit(sync(check_only=args.check))
