from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Any

from nextgen.core.types import StrategyConfig


def load_strategy_config(raw: Mapping[str, Any]) -> StrategyConfig:
    return StrategyConfig(
        name=str(raw["name"]),
        symbols=tuple(raw["symbols"]),
        target_volatility=float(raw["target_volatility"]),
        max_gross_exposure=float(raw["max_gross_exposure"]),
        max_net_exposure=float(raw["max_net_exposure"]),
        max_position_weight=float(raw["max_position_weight"]),
        drawdown_throttle=float(raw["drawdown_throttle"]),
        kill_switch_drawdown=float(raw["kill_switch_drawdown"]),
    )


def load_strategy_config_file(path: str | Path) -> StrategyConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return load_strategy_config(data)
