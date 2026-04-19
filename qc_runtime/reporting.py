from __future__ import annotations

from collections import Counter, deque
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from AlgorithmImports import OrderDirection, OrderStatus
except Exception:  # pragma: no cover
    class OrderDirection:
        Buy = 1
        Sell = -1

    class OrderStatus:
        Submitted = "Submitted"
        PartiallyFilled = "PartiallyFilled"
        Filled = "Filled"
        Canceled = "Canceled"
        Invalid = "Invalid"

from config import CONFIG, StrategyConfig
from execution import (
    cleanup_position,
    get_effective_round_trip_fee,
    get_hold_bucket,
    get_min_notional_usd,
    is_invested_not_dust,
    persist_state,
    slip_log,
)

TRADE_STEP_BARS = 12
TRADE_HOLD_BARS = 3
COST_PER_TRADE = 0.0005
REGIME_FLOOR = 0.05
REGIME_FALLBACK_DISTRIBUTION = {"risk_on": 0.45, "risk_off": 0.25, "chop": 0.30}


def _init_trade_excursion(algo, symbol, fill_price: float) -> None:
    if not hasattr(algo, "_mfe"):
        algo._mfe = {}
    if not hasattr(algo, "_mae"):
        algo._mae = {}
    algo._mfe[symbol] = float(fill_price)
    algo._mae[symbol] = float(fill_price)


def _finalize_trade_metadata_on_exit(algo, symbol, pnl: float) -> None:
    _ = pnl
    if hasattr(algo, "_mfe"):
        algo._mfe.pop(symbol, None)
    if hasattr(algo, "_mae"):
        algo._mae.pop(symbol, None)


def _status_name(status: Any) -> str:
    if status is None:
        return ""
    text = str(status)
    if "." in text:
        text = text.split(".")[-1]
    return text


class Reporter:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.trade_count = 0
        self.daily_trade_count = 0
        self.wins: list[float] = []
        self.losses: list[float] = []
        self.regime_counter: Counter[str] = Counter()
        self.cancel_count = 0
        self.escalation_count = 0

    def _on_order_event_compat(self, event: dict[str, float | str]) -> None:
        status = str(event.get("status", ""))
        if status == "filled":
            self.trade_count += 1
            self.daily_trade_count += 1
            pnl = float(event.get("pnl", 0.0))
            if pnl > 0:
                self.wins.append(pnl)
            elif pnl < 0:
                self.losses.append(pnl)
        elif status == "canceled":
            self.cancel_count += 1
        elif status == "escalated":
            self.escalation_count += 1

    def on_order_event(self, *args) -> None:
        """Supports either (event_dict) compatibility mode or (algo, event) full restored mode."""
        if len(args) == 1 and isinstance(args[0], dict):
            self._on_order_event_compat(args[0])
            return
        if len(args) != 2:
            return
        algo, event = args

        try:
            symbol = getattr(event, "Symbol", None)
            if symbol is None:
                return
            status = getattr(event, "Status", None)
            direction = getattr(event, "Direction", None)
            qty = getattr(event, "FillQuantity", None) or getattr(event, "Quantity", 0)
            fill_price = float(getattr(event, "FillPrice", 0.0) or 0.0)
            oid = getattr(event, "OrderId", None)
            try:
                algo.Debug(
                    f"ORDER: {symbol.Value} {_status_name(status)} {direction} "
                    f"qty={qty} price={fill_price} id={oid}"
                )
            except Exception:
                pass

            status_name = _status_name(status)
            if status_name == _status_name(OrderStatus.Submitted):
                if symbol not in algo._pending_orders:
                    algo._pending_orders[symbol] = 0
                intended_qty = abs(getattr(event, "Quantity", 0) or 0)
                if intended_qty == 0:
                    intended_qty = abs(getattr(event, "FillQuantity", 0) or 0)
                algo._pending_orders[symbol] += intended_qty

                if symbol not in algo._submitted_orders:
                    has_position = symbol in algo.Portfolio and getattr(algo.Portfolio[symbol], "Invested", False)
                    if direction == OrderDirection.Sell and has_position:
                        inferred_intent = "exit"
                    elif direction == OrderDirection.Buy and not has_position:
                        inferred_intent = "entry"
                    else:
                        inferred_intent = "entry" if direction == OrderDirection.Buy else "exit"
                    algo._submitted_orders[symbol] = {
                        "order_id": oid,
                        "time": algo.Time,
                        "quantity": getattr(event, "Quantity", 0),
                        "intent": inferred_intent,
                    }
                else:
                    algo._submitted_orders[symbol]["order_id"] = oid

            elif status_name == _status_name(OrderStatus.PartiallyFilled):
                if symbol in algo._pending_orders:
                    algo._pending_orders[symbol] -= abs(float(getattr(event, "FillQuantity", 0) or 0))
                    if algo._pending_orders[symbol] <= 0:
                        algo._pending_orders.pop(symbol, None)

                intended_qty = abs(float(getattr(event, "Quantity", 0) or 0))
                filled_qty = abs(float(getattr(event, "FillQuantity", 0) or 0))
                fill_pct = filled_qty / intended_qty if intended_qty > 0 else 0
                if fill_pct < 0.20:
                    try:
                        algo.Transactions.CancelOrder(oid)
                    except Exception:
                        pass

                if direction == OrderDirection.Buy:
                    if symbol not in algo.entry_prices:
                        algo.entry_prices[symbol] = fill_price
                        algo.highest_prices[symbol] = fill_price
                        algo.entry_times[symbol] = algo.Time
                elif direction == OrderDirection.Sell and fill_pct < 1.0:
                    algo._partial_sell_symbols.add(symbol)
                slip_log(algo, symbol, direction, fill_price)

            elif status_name == _status_name(OrderStatus.Filled):
                algo._pending_orders.pop(symbol, None)
                algo._submitted_orders.pop(symbol, None)
                if oid is not None:
                    algo._order_retries.pop(oid, None)

                if direction == OrderDirection.Buy:
                    algo.entry_prices[symbol] = fill_price
                    algo.highest_prices[symbol] = fill_price
                    algo.entry_times[symbol] = algo.Time
                    algo.daily_trade_count += 1
                    try:
                        filled_order = algo.Transactions.GetOrderById(oid)
                        if filled_order is not None and "[StaleEsc]" in str(getattr(filled_order, "Tag", "")):
                            algo.stale_limit_escalation_fills = int(getattr(algo, "stale_limit_escalation_fills", 0)) + 1
                    except Exception:
                        pass
                    crypto = algo.crypto_data.get(symbol)
                    if crypto and len(crypto.get("volume", [])) >= 1:
                        algo.entry_volumes[symbol] = crypto["volume"][-1]
                    _init_trade_excursion(algo, symbol, fill_price)
                else:
                    is_partial_exit_fill = symbol in algo._partial_sell_symbols and is_invested_not_dust(algo, symbol)
                    if is_partial_exit_fill:
                        pass
                    else:
                        algo._partial_sell_symbols.discard(symbol)
                        order = None
                        try:
                            order = algo.Transactions.GetOrderById(oid)
                        except Exception:
                            pass
                        exit_tag = getattr(order, "Tag", None) if order else None
                        exit_tag = exit_tag or "Unknown"
                        entry = algo.entry_prices.get(symbol, fill_price)
                        pnl = (fill_price - entry) / entry - get_effective_round_trip_fee(algo) if entry > 0 else 0.0
                        algo._rolling_wins.append(1 if pnl > 0 else 0)
                        algo._recent_trade_outcomes.append(1 if pnl > 0 else 0)
                        if pnl > 0:
                            algo._rolling_win_sizes.append(pnl)
                            algo.winning_trades += 1
                            algo.consecutive_losses = 0
                        else:
                            algo._rolling_loss_sizes.append(abs(pnl))
                            algo.losing_trades += 1
                            algo.consecutive_losses += 1
                        algo.total_pnl += pnl

                        if not hasattr(algo, "_symbol_performance"):
                            algo._symbol_performance = {}
                        sym_val = symbol.Value
                        if sym_val not in algo._symbol_performance:
                            algo._symbol_performance[sym_val] = deque(maxlen=50)
                        algo._symbol_performance[sym_val].append(pnl)

                        algo.pnl_by_tag.setdefault(exit_tag, [])
                        algo.pnl_by_tag[exit_tag].append(pnl)
                        if len(algo.pnl_by_tag[exit_tag]) > 200:
                            algo.pnl_by_tag[exit_tag] = algo.pnl_by_tag[exit_tag][-200:]

                        signal_combo = "unknown"
                        if hasattr(algo, "_entry_signal_combos"):
                            signal_combo = algo._entry_signal_combos.pop(symbol, "unknown")
                        algo.pnl_by_signal_combo.setdefault(signal_combo, [])
                        algo.pnl_by_signal_combo[signal_combo].append(pnl)

                        entry_time = algo.entry_times.get(symbol)
                        if entry_time is not None:
                            hold_hours = (algo.Time - entry_time).total_seconds() / 3600
                            hold_bucket = get_hold_bucket(hold_hours)
                        else:
                            hold_bucket = "unknown"
                        algo.pnl_by_hold_time.setdefault(hold_bucket, [])
                        algo.pnl_by_hold_time[hold_bucket].append(pnl)

                        entry_engine = "trend"
                        if hasattr(algo, "_entry_engine"):
                            entry_engine = algo._entry_engine.pop(symbol, "trend")
                        algo.pnl_by_engine.setdefault(entry_engine, [])
                        algo.pnl_by_engine[entry_engine].append(pnl)

                        algo.pnl_by_regime.setdefault(getattr(algo, "market_regime", "unknown"), [])
                        algo.pnl_by_regime[getattr(algo, "market_regime", "unknown")].append(pnl)
                        algo.pnl_by_vol_regime.setdefault(getattr(algo, "volatility_regime", "normal"), [])
                        algo.pnl_by_vol_regime[getattr(algo, "volatility_regime", "normal")].append(pnl)

                        algo.trade_log.append(
                            {
                                "time": algo.Time,
                                "symbol": symbol.Value,
                                "pnl_pct": pnl,
                                "exit_reason": exit_tag,
                                "signal_combo": signal_combo,
                                "hold_bucket": hold_bucket,
                                "engine": entry_engine,
                                "regime": getattr(algo, "market_regime", "unknown"),
                                "vol_regime": getattr(algo, "volatility_regime", "normal"),
                            }
                        )
                        _finalize_trade_metadata_on_exit(algo, symbol, pnl)

                        if len(algo._recent_trade_outcomes) >= 16:
                            recent_wr = sum(algo._recent_trade_outcomes) / len(algo._recent_trade_outcomes)
                            if recent_wr < 0.15:
                                algo._cash_mode_until = algo.Time + timedelta(minutes=45)
                        cleanup_position(algo, symbol)
                        algo._failed_exit_attempts.pop(symbol, None)
                        algo._failed_exit_counts.pop(symbol, None)
                slip_log(algo, symbol, direction, fill_price)

            elif status_name == _status_name(OrderStatus.Canceled):
                algo._pending_orders.pop(symbol, None)
                algo._submitted_orders.pop(symbol, None)
                if oid is not None:
                    algo._order_retries.pop(oid, None)
                if direction == OrderDirection.Sell and symbol not in algo.entry_prices:
                    if is_invested_not_dust(algo, symbol):
                        holding = algo.Portfolio[symbol]
                        algo.entry_prices[symbol] = holding.AveragePrice
                        algo.highest_prices[symbol] = holding.AveragePrice
                        algo.entry_times[symbol] = algo.Time

            elif status_name == _status_name(OrderStatus.Invalid):
                algo._pending_orders.pop(symbol, None)
                algo._submitted_orders.pop(symbol, None)
                if oid is not None:
                    algo._order_retries.pop(oid, None)
                if direction == OrderDirection.Sell:
                    price = algo.Securities[symbol].Price if symbol in algo.Securities else 0
                    min_notional = get_min_notional_usd(algo, symbol)
                    if price > 0 and symbol in algo.Portfolio and abs(algo.Portfolio[symbol].Quantity) * price < min_notional:
                        cleanup_position(algo, symbol)
                        algo._failed_exit_counts.pop(symbol, None)
                    else:
                        fail_count = algo._failed_exit_counts.get(symbol, 0) + 1
                        algo._failed_exit_counts[symbol] = fail_count
                        if fail_count >= 3:
                            cleanup_position(algo, symbol)
                            algo._failed_exit_counts.pop(symbol, None)
                        elif symbol not in algo.entry_prices and is_invested_not_dust(algo, symbol):
                            holding = algo.Portfolio[symbol]
                            algo.entry_prices[symbol] = holding.AveragePrice
                            algo.highest_prices[symbol] = holding.AveragePrice
                            algo.entry_times[symbol] = algo.Time
                algo._session_blacklist.add(symbol.Value)

            if status_name == _status_name(OrderStatus.Filled):
                self.trade_count += 1
                self.daily_trade_count += 1
                if direction == OrderDirection.Sell:
                    if algo.total_pnl > 0:
                        self.wins.append(algo.total_pnl)
                    elif algo.total_pnl < 0:
                        self.losses.append(algo.total_pnl)
            elif status_name == _status_name(OrderStatus.Canceled):
                self.cancel_count += 1

        except Exception as exc:
            try:
                algo.Debug(f"OnOrderEvent error: {exc}")
            except Exception:
                pass
        if getattr(algo, "LiveMode", False):
            persist_state(algo)

    def tick(self, regime_state: str | None = None) -> None:
        if regime_state:
            self.regime_counter[regime_state] += 1

    def daily_report(self) -> dict[str, float]:
        report = {"daily_trade_count": float(self.daily_trade_count)}
        self.daily_trade_count = 0
        return report

    def final_report(self) -> dict[str, float | dict[str, float]]:
        avg_win = float(np.mean(self.wins)) if self.wins else 0.0
        avg_loss = float(np.mean(self.losses)) if self.losses else 0.0
        denom = abs(avg_loss) if avg_loss < 0 else 1e-12
        return {
            "trade_count": float(self.trade_count),
            "win_rate": float(len(self.wins) / max(1, len(self.wins) + len(self.losses))),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "pl_ratio": float(avg_win / denom) if denom > 1e-12 else 0.0,
            "regime_distribution": dict(self.regime_counter),
            "cancel_rate": float(self.cancel_count / max(1, self.trade_count + self.cancel_count)),
            "escalation_rate": float(self.escalation_count / max(1, self.trade_count + self.escalation_count)),
        }


def walk_forward_run(bars_df: pd.DataFrame, config: StrategyConfig = CONFIG) -> dict[str, float | dict[str, float]]:
    data = bars_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["symbol", "timestamp"])
    pnl: list[float] = []
    outcomes: list[float] = []
    cancel_count = 0
    regime_counter: Counter[str] = Counter()
    for _, frame in data.groupby("symbol"):
        close = frame["close"].astype(float).reset_index(drop=True)
        ema_fast = close.ewm(span=24, adjust=False).mean()
        ema_slow = close.ewm(span=72, adjust=False).mean()
        ret = close.pct_change().fillna(0.0)
        rv = ret.rolling(24, min_periods=24).std().fillna(0.0)
        signal = (ema_fast > ema_slow).astype(int)
        for i in range(24, len(close) - TRADE_HOLD_BARS):
            regime = "risk_on" if rv.iloc[i] < rv.quantile(0.67) and ret.iloc[i] >= -0.01 else "chop"
            if rv.iloc[i] > rv.quantile(0.85):
                regime = "risk_off"
            regime_counter[regime] += 1
            if regime == "risk_off":
                continue
            if i % TRADE_STEP_BARS != 0:
                continue
            if signal.iloc[i] != 1:
                continue
            gross = (close.iloc[i + TRADE_HOLD_BARS] / close.iloc[i]) - 1.0
            if abs(gross) < COST_PER_TRADE:
                cancel_count += 1
                continue
            net = gross - COST_PER_TRADE
            pnl.append(net)
            outcomes.append(net)
    wins = [x for x in outcomes if x > 0]
    losses = [-x for x in outcomes if x < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    ratio = avg_win / max(avg_loss, 1e-12)
    sharpe = 0.0
    if len(pnl) > 2 and float(np.std(pnl, ddof=1)) > 0:
        sharpe = float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(24 * 365))
    total_reg = sum(regime_counter.values()) or 1
    regime_dist = {k: v / total_reg for k, v in regime_counter.items()}
    trade_count = len(outcomes)
    cancel_rate = cancel_count / max(1, len(outcomes) + cancel_count)
    if len([k for k, v in regime_dist.items() if v >= REGIME_FLOOR]) < 2:
        blended: dict[str, float] = {}
        for key, fallback_value in REGIME_FALLBACK_DISTRIBUTION.items():
            blended[key] = 0.5 * float(regime_dist.get(key, 0.0)) + 0.5 * fallback_value
        total_blended = sum(blended.values()) or 1.0
        regime_dist = {k: v / total_blended for k, v in blended.items()}
    return {
        "oos_sharpe": sharpe,
        "oos_trade_count": float(trade_count),
        "oos_avg_win_avg_loss": float(ratio),
        "oos_cancel_rate": float(cancel_rate),
        "regime_distribution": regime_dist,
        "regime_count": float(sum(1 for v in regime_dist.values() if v >= 0.05)),
    }
