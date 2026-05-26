#!/usr/bin/env python3
"""Fix imports and known merge bugs in consolidated Kraken Max modules."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "Kraken Max"

LOAD_MERGED = '''
def load_regime_weights_merged(path: Path | None = None) -> dict[str, dict[str, float]]:
    base = load_regime_weights(path)
    target = path or (Path(__file__).resolve().parent / str(CONFIG.regime_weights_path))
    if not target.exists():
        return base
    try:
        blob = json.loads(target.read_text(encoding="utf-8"))
        if blob.get("source", "").startswith("regime_walk_forward"):
            regimes = blob.get("regimes") or {}
            for name, w in regimes.items():
                key = normalize_regime_key(name)
                base[key] = {k: float(w[k]) for k in _WEIGHT_KEYS if k in w}
    except Exception:
        pass
    return base
'''

HEADERS = {
    "data.py": '''"""Kraken Max — market data & sentiment (`data.py`)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import CONFIG, KrakenMaxConfig

''',
    "kraken_ml.py": '''"""Kraken Max — ML scorer & trainer (`kraken_ml.py`)."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import CONFIG, KrakenMaxConfig

''',
    "regime.py": '''"""Kraken Max — regime detection & ensemble weights (`regime.py`)."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from config import CONFIG, KrakenMaxConfig
from data import SentimentSnapshot, adjust_deployment_cap

''',
    "core.py": '''"""Kraken Max — features, universe, ensemble, sizing (`core.py`)."""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig
from kraken_ml import MLScorer, load_ml_weights
from regime import (
    config_for_regime,
    load_regime_weights,
    load_regime_weights_from_object_store,
    load_regime_weights_merged,
)

''',
    "risk.py": '''"""Kraken Max — risk, clusters, ERC (`risk.py`)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig

''',
    "execution.py": '''"""Kraken Max — order execution (`execution.py`)."""
from __future__ import annotations

import importlib.util
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import CONFIG
from risk import PositionRisk, should_exit

''',
    "kraken_ops.py": '''"""Kraken Max — monitoring, alerts, scorecard (`kraken_ops.py`)."""
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

''',
    "workflow.py": '''"""Kraken Max — walk-forward, validation, revalidation (`workflow.py`)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig
from kraken_ops import BaselineManager, DriftMonitor
from regime import (
    _DEFAULT_REGIME_MAP,
    _WEIGHT_KEYS,
    load_regime_weights,
    normalize_regime_key,
)

''',
}


def reheader(name: str, text: str) -> str:
    # drop old header until first '# --- from'
    idx = text.find("\n# --- from")
    body = text[idx:] if idx >= 0 else text
    return HEADERS[name] + body.lstrip()


def fix_ops_scorecard(text: str) -> str:
    broken = """    from risk import max_cluster_exposure  # telemetry


def build(self, algo, *, fill_tracker=None) -> ScorecardSnapshot:"""
    fixed = """    def build(self, algo, *, fill_tracker=None) -> ScorecardSnapshot:"""
    return text.replace(broken, fixed)


def fix_telemetry_cluster(text: str) -> str:
    return text.replace(
        """            try:
                # max_cluster_exposure defined in this package risk.py

                clusters = max_cluster_exposure(erc)""",
        "            try:\n                clusters = max_cluster_exposure(erc)",
    )


def fix_workflow(text: str) -> str:
    text = text.replace("from core import AlphaEnsemble\n", "")
    text = text.replace("from kraken_ml import MLScorer\n", "")
    text = text.replace(
        "    ens = AlphaEnsemble(cfg, MLScorer())",
        "    from core import AlphaEnsemble\n    from kraken_ml import MLScorer\n\n        ens = AlphaEnsemble(cfg, MLScorer())",
    )
    # fix indentation on ens line - read actual context
    text = text.replace(
        "        ens = AlphaEnsemble(cfg, MLScorer())",
        "        from core import AlphaEnsemble\n        from kraken_ml import MLScorer\n        ens = AlphaEnsemble(cfg, MLScorer())",
    )
    # remove duplicate load_regime_weights_merged at end if we move to regime
    return text


def fix_execution_scalper(text: str) -> str:
    return text.replace(
        "from core import evaluate_scalper_exit\nfrom core import can_afford, free_cash_usd, round_qty",
        "from core import can_afford, evaluate_scalper_exit, free_cash_usd, round_qty",
    )


def main() -> None:
    for name in HEADERS:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        text = reheader(name, text)
        if name == "kraken_ops.py":
            text = fix_ops_scorecard(text)
            text = fix_telemetry_cluster(text)
        if name == "workflow.py":
            text = fix_workflow(text)
        if name == "execution.py":
            text = fix_execution_scalper(text)
        if name == "risk.py" and "from core import hourly_returns" not in text:
            text = text.replace(
                "# --- from portfolio_optimizer.py ---",
                "# --- from portfolio_optimizer.py ---\n\nfrom core import hourly_returns\n",
                1,
            )
        path.write_text(text, encoding="utf-8")
        print(f"fixed {name}: {len(text)} chars")

    regime = ROOT / "regime.py"
    rt = regime.read_text(encoding="utf-8")
    if "def load_regime_weights_merged" not in rt:
        regime.write_text(rt.rstrip() + "\n\n" + LOAD_MERGED.strip() + "\n", encoding="utf-8")
        print("added load_regime_weights_merged to regime.py")

    wf = ROOT / "workflow.py"
    wt = wf.read_text(encoding="utf-8")
    if "def load_regime_weights_merged" in wt:
        # remove duplicate function from workflow (keep in regime only)
        start = wt.find("\ndef load_regime_weights_merged")
        if start > 0:
            wf.write_text(wt[:start].rstrip() + "\n", encoding="utf-8")
            print("removed duplicate load_regime_weights_merged from workflow.py")


if __name__ == "__main__":
    main()
