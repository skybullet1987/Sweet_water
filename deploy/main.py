"""
QuantConnect project entry point.

Copy this file to your QC project root as ``main.py`` (alongside config.py, core.py, …).
If you keep code in a subfolder, copy the whole ``Kraken Max/`` tree and leave this file at the root.

QuantConnect must see a ``QCAlgorithm`` subclass in ``main.py`` — we re-export ``KrakenMaxAlgorithm``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_KM = _ROOT / "Kraken Max"
if _KM.is_dir() and str(_KM) not in sys.path:
    sys.path.insert(0, str(_KM))
elif str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main import KrakenMaxAlgorithm  # noqa: E402

__all__ = ["KrakenMaxAlgorithm"]
