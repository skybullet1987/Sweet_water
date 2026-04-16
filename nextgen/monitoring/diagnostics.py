from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class DiagnosticEvent:
    timestamp: datetime
    level: str
    code: str
    message: str


class DriftMonitor:
    """Scaffold for live drift/anomaly diagnostics."""

    def __init__(self, max_signal_drift: float = 0.5, min_fill_rate: float = 0.5) -> None:
        self.max_signal_drift = max_signal_drift
        self.min_fill_rate = min_fill_rate

    def evaluate(self, signal_drift: float, fill_rate: float) -> tuple[DiagnosticEvent, ...]:
        events = []
        now = datetime.now(UTC)
        if signal_drift > self.max_signal_drift:
            events.append(DiagnosticEvent(now, "warning", "signal_drift", f"Signal drift {signal_drift:.3f} exceeded {self.max_signal_drift:.3f}"))
        if fill_rate < self.min_fill_rate:
            events.append(DiagnosticEvent(now, "warning", "fill_rate", f"Fill rate {fill_rate:.3f} below {self.min_fill_rate:.3f}"))
        return tuple(events)
