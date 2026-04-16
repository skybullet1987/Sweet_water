from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StressScenario:
    name: str
    spread_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    liquidity_haircut: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    start: datetime
    end: datetime
    symbols: tuple[str, ...]
    initial_cash: float
    stress_scenarios: tuple[StressScenario, ...] = ()


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    created_at: datetime
    config_hash: str
    git_sha: str | None = None


@dataclass(frozen=True)
class RunResult:
    metadata: RunMetadata
    metrics: Mapping[str, float]
    scenario_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


class BacktestHarness:
    """Foundational research runner with immutable metadata capture."""

    def _make_metadata(self, config: ExperimentConfig, git_sha: str | None = None) -> RunMetadata:
        payload = {
            "name": config.name,
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "symbols": list(config.symbols),
            "initial_cash": config.initial_cash,
            "stress_scenarios": [s.__dict__ for s in config.stress_scenarios],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return RunMetadata(run_id=digest[:12], created_at=datetime.now(UTC), config_hash=digest, git_sha=git_sha)

    def run(self, config: ExperimentConfig, returns_by_symbol: Mapping[str, Sequence[float]], git_sha: str | None = None) -> RunResult:
        metadata = self._make_metadata(config, git_sha=git_sha)

        combined = []
        for symbol in config.symbols:
            combined.extend(returns_by_symbol.get(symbol, ()))

        avg_return = sum(combined) / len(combined) if combined else 0.0
        metrics = {
            "average_return": float(avg_return),
            "observations": float(len(combined)),
        }

        scenario_metrics: dict[str, Mapping[str, float]] = {}
        for scenario in config.stress_scenarios:
            stressed_avg = avg_return * (1.0 - (scenario.spread_multiplier - 1.0) * 0.05) * (1.0 - (scenario.slippage_multiplier - 1.0) * 0.05)
            scenario_metrics[scenario.name] = {"average_return": float(stressed_avg)}

        return RunResult(metadata=metadata, metrics=metrics, scenario_metrics=scenario_metrics)
