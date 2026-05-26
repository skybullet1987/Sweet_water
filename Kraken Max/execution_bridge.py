from __future__ import annotations

"""
Execution bridge: load qc_runtime/execution in isolation, else Kraken Max/execution.py.
"""

import importlib.util
import sys
from pathlib import Path

_KRAKEN_MAX = Path(__file__).resolve().parent
_REPO = _KRAKEN_MAX.parent
_QC_RUNTIME = _REPO / "qc_runtime"

_qc_execution = None
_local_execution = None
_USE_PRO = False


def _load_module(name: str, path: Path, prepend: Path):
    saved = sys.path[:]
    try:
        sys.path = [str(prepend)] + [p for p in sys.path if p not in {str(prepend), str(_KRAKEN_MAX)}]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path = saved


try:
    _qc_execution = _load_module("qc_runtime_execution", _QC_RUNTIME / "execution.py", _QC_RUNTIME)
    _USE_PRO = True
except Exception:
    _qc_execution = None
    _USE_PRO = False

try:
    _local_execution = _load_module("km_execution", _KRAKEN_MAX / "execution.py", _KRAKEN_MAX)
except Exception:
    _local_execution = None


def init_execution_state(algo) -> None:
    algo._submitted_orders = getattr(algo, "_submitted_orders", {})
    algo._pending_limits = getattr(algo, "_pending_limits", {})
    algo._abandoned_dust = getattr(algo, "_abandoned_dust", set())
    algo._failed_escalations = getattr(algo, "_failed_escalations", {})
    algo.stale_order_bars = int(getattr(getattr(algo, "config", None), "stale_order_bars", 3) or 3)
    algo.min_notional = float(getattr(getattr(algo, "config", None), "min_position_floor_usd", 25.0) or 25.0)


def track_order_submit(algo, ticket, *, symbol, qty: float, expected_price: float, force_market: bool = False) -> None:
    """Notify FillTracker when a limit (or market) order is submitted (v6)."""
    ft = getattr(algo, "fill_tracker", None)
    if ft is None or ticket is None:
        return
    oid = int(getattr(ticket, "OrderId", 0) or 0)
    if oid <= 0:
        return
    is_limit = not force_market and bool(getattr(getattr(algo, "config", None), "use_limit_orders", True))
    ft.on_submit(oid, is_limit=is_limit, expected_price=float(expected_price), qty=abs(float(qty)))


def place_buy_notional(algo, symbol, usd_notional: float, *, tag: str = "Entry", force_market: bool = False) -> bool:
    if _USE_PRO and _qc_execution is not None:
        from config import CONFIG as KM_CONFIG

        price = float(algo.Securities[symbol].Price)
        if price <= 0:
            return False
        hourly = float(getattr(algo.Securities[symbol], "Volume", 0.0) or 0.0) * price
        cap = hourly * float(KM_CONFIG.max_participation_rate)
        notional = min(float(usd_notional), cap) if cap > 0 else float(usd_notional)
        qty = _qc_execution.round_quantity(algo, symbol, notional / price)
        if qty <= 0:
            return False
        ticket = _qc_execution.place_entry(algo, symbol, qty, tag=tag, force_market=force_market)
        if ticket is not None:
            track_order_submit(algo, ticket, symbol=symbol, qty=qty, expected_price=price, force_market=force_market)
        return ticket is not None
    if _local_execution is not None:
        ok = bool(_local_execution.place_buy_notional(algo, symbol, usd_notional, tag=tag, force_market=force_market))
        if ok:
            pending = (getattr(algo, "_pending_limits", {}) or {}).get(symbol)
            if pending:
                track_order_submit(
                    algo,
                    type("T", (), {"OrderId": pending.get("order_id")})(),
                    symbol=symbol,
                    qty=float(pending.get("qty", qty)),
                    expected_price=price,
                    force_market=force_market,
                )
        return ok
    return False


def liquidate_symbol(algo, symbol, *, force_market: bool = True) -> bool:
    if _USE_PRO and _qc_execution is not None:
        return bool(_qc_execution.smart_liquidate(algo, symbol, tag="KM:Exit"))
    if _local_execution is not None:
        _local_execution.liquidate_symbol(algo, symbol, force_market=force_market)
        return True
    return False


def escalate_orders(algo) -> list:
    if _USE_PRO and _qc_execution is not None:
        return list(_qc_execution.escalate_stale_orders(algo) or [])
    if _local_execution is not None:
        _local_execution.escalate_stale_limits(algo)
    return []


def manage_position_exit(algo, symbol, state, close: float, now, feats: dict | None = None) -> bool:
    if _local_execution is not None:
        return _local_execution.manage_exits(algo, symbol, state, close, now, feats)
    return False


def position_qty(algo, symbol) -> float:
    if _USE_PRO and _qc_execution is not None:
        try:
            return float(getattr(algo.Portfolio[symbol], "Quantity", 0.0) or 0.0)
        except Exception:
            return 0.0
    if _local_execution is not None:
        return _local_execution.position_qty(algo, symbol)
    return 0.0


def estimate_fee_pct(algo, notional: float, is_limit: bool = True) -> float:
    if _USE_PRO and _qc_execution is not None:
        model = _qc_execution.KrakenTieredFeeModel(comparison_mode=True)
        return float(model.estimate_round_trip_cost("BTCUSD", notional, is_limit=is_limit) / max(notional, 1e-9))
    from config import CONFIG

    return float(CONFIG.expected_round_trip_fees)
