from __future__ import annotations

import sys
from pathlib import Path

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from config import CONFIG  # noqa: E402
from core import select_entry_candidates  # noqa: E402


def test_select_entry_candidates_uses_rank_when_all_below_threshold():
    scores = [
        ("BTCUSD", -0.18, {}),
        ("ETHUSD", -0.26, {}),
        ("SOLUSD", -0.65, {}),
    ]
    out = select_entry_candidates(scores, config=CONFIG)
    assert len(out) >= 1
    assert out[0][0] == "BTCUSD"
    assert out[0][1] == -0.18


def test_select_entry_candidates_absolute_when_above_threshold():
    scores = [
        ("BTCUSD", 0.5, {}),
        ("ETHUSD", 0.1, {}),
    ]
    out = select_entry_candidates(scores, config=CONFIG)
    assert out == [("BTCUSD", 0.5)]
