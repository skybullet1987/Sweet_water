"""Pure signal and state helpers for scalper gating and dust classification."""

from __future__ import annotations

from datetime import date


def dust_is_effectively_flat(*, qty: float, price: float, min_qty: float, min_notional: float) -> bool:
    qty_f = max(float(qty or 0.0), 0.0)
    px_f = max(float(price or 0.0), 0.0)
    return qty_f < max(float(min_qty or 0.0), 0.0) and (qty_f * px_f) < max(float(min_notional or 0.0), 0.0)


def should_emit_once_per_day(last_day: date | None, current_day: date) -> bool:
    return last_day != current_day

