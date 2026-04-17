from __future__ import annotations

from execution import place_limit_or_market
from exits.triple_barrier import TripleBarrier, check_barrier_hit


def execute_regime_entries(algo, candidates, regime: str):
    tickets = []
    for candidate in candidates:
        symbol = candidate["symbol"]
        quantity = candidate["quantity"]
        tag = f"{regime} Entry"
        ticket = place_limit_or_market(
            algo,
            symbol,
            quantity,
            tag=tag,
            signal_engine="rule_scorer",
            signal_regime=regime,
            signal_score=candidate.get("score"),
            signal_threshold=candidate.get("threshold"),
        )
        if ticket is not None:
            tickets.append(ticket)
    return tickets


def manage_open_positions(algo, positions: dict, barriers: dict, bar_index: int):
    closed = []
    for symbol, position in list(positions.items()):
        barrier: TripleBarrier | None = barriers.get(symbol)
        if barrier is None:
            continue

        high = float(position["high"])
        low = float(position["low"])
        close = float(position["close"])
        hit = check_barrier_hit(barrier, bar_index=bar_index, high=high, low=low, close=close)
        if hit is None:
            continue

        qty = float(position["quantity"])
        if qty != 0:
            algo.MarketOrder(symbol, -qty, tag=f"{hit} exit")
            closed.append((symbol, hit))
            barriers.pop(symbol, None)
    return closed
