from __future__ import annotations

from collections import Counter
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
from execution import PositionState, cleanup_position, get_effective_round_trip_fee, get_min_notional_usd


class Reporter:
    def __init__(self, config: StrategyConfig = CONFIG) -> None:
        self.config = config
        self.trade_count = 0
        self.daily_trade_count = 0
        self.wins: list[float] = []
        self.losses: list[float] = []
        self.daily_wins: list[float] = []
        self.daily_losses: list[float] = []
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
                self.daily_wins.append(pnl)
            elif pnl < 0:
                self.losses.append(pnl)
                self.daily_losses.append(pnl)
        elif status == "canceled":
            self.cancel_count += 1
        elif status == "escalated":
            self.escalation_count += 1

    def on_order_event(self, *args) -> None:
        if len(args) == 1 and isinstance(args[0], dict):
            self._on_order_event_compat(args[0])
            return
        if len(args) != 2:
            return
        algo, event = args
        symbol = getattr(event, "Symbol", None)
        if symbol is None:
            return
        status = str(getattr(event, "Status", ""))
        direction = getattr(event, "Direction", None)
        fill_price = float(getattr(event, "FillPrice", 0.0) or 0.0)
        qty = float(getattr(event, "FillQuantity", getattr(event, "Quantity", 0.0)) or 0.0)

        if "Submitted" in status:
            if symbol not in algo._pending_orders:
                algo._pending_orders[symbol] = 0.0
            algo._pending_orders[symbol] += abs(qty)
            return

        if "Canceled" in status:
            self.cancel_count += 1
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            return

        if "Invalid" in status:
            algo._pending_orders.pop(symbol, None)
            algo._submitted_orders.pop(symbol, None)
            if direction == OrderDirection.Sell and symbol in algo.Portfolio:
                price = float(getattr(algo.Securities.get(symbol), "Price", 0.0) or 0.0)
                if price > 0 and abs(float(algo.Portfolio[symbol].Quantity)) * price < get_min_notional_usd(algo, symbol):
                    cleanup_position(algo, symbol)
            return

        if "Filled" not in status:
            return

        algo._pending_orders.pop(symbol, None)
        algo._submitted_orders.pop(symbol, None)
        self.trade_count += 1
        self.daily_trade_count += 1

        if direction == OrderDirection.Buy:
            atr = float(getattr(algo.feature_engine, "current_features", lambda *_: {})(symbol.Value).get("atr", 0.0) or 0.0)
            if atr <= 0:
                atr = max(fill_price * 0.01, 1e-6)
            if not hasattr(algo, "position_state"):
                algo.position_state = {}
            algo.position_state[symbol] = PositionState(
                entry_price=fill_price,
                highest_close=fill_price,
                entry_atr=atr,
                entry_time=getattr(algo, "Time", None),
            )
            return

        if direction == OrderDirection.Sell:
            pstate = getattr(algo, "position_state", {}).get(symbol)
            entry = float(getattr(pstate, "entry_price", fill_price) or fill_price)
            pnl = (fill_price - entry) / max(entry, 1e-9) - get_effective_round_trip_fee(algo)
            if pnl > 0:
                self.wins.append(pnl)
                self.daily_wins.append(pnl)
            elif pnl < 0:
                self.losses.append(pnl)
                self.daily_losses.append(pnl)
            tag = "Unknown"
            try:
                oid = getattr(event, "OrderId", None)
                order = algo.Transactions.GetOrderById(oid) if oid is not None else None
                tag = str(getattr(order, "Tag", "Unknown") or "Unknown")
            except Exception:
                pass
            algo.pnl_by_tag.setdefault(tag, []).append(pnl)
            algo.pnl_by_regime.setdefault(getattr(algo, "market_regime", "unknown"), []).append(pnl)
            cleanup_position(algo, symbol)

    def tick(self, regime_state: str | None = None) -> None:
        if regime_state:
            self.regime_counter[regime_state] += 1

    def daily_report(self) -> dict[str, float]:
        trades = len(self.daily_wins) + len(self.daily_losses)
        win_rate = float(len(self.daily_wins) / trades) if trades > 0 else 0.0
        avg_win = float(np.mean(self.daily_wins)) if self.daily_wins else 0.0
        avg_loss = float(np.mean(self.daily_losses)) if self.daily_losses else 0.0
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
        report = {
            "daily_trade_count": float(self.daily_trade_count),
            "win_rate": win_rate,
            "avg_win_pct": avg_win * 100.0,
            "avg_loss_pct": avg_loss * 100.0,
            "expectancy_pct": expectancy * 100.0,
        }
        self.daily_trade_count = 0
        self.daily_wins = []
        self.daily_losses = []
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


def _cross_rank(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = max(len(items) - 1, 1)
    return {sym: i / n for i, (sym, _) in enumerate(items)}


def walk_forward_run(bars_df: pd.DataFrame, config: StrategyConfig = CONFIG) -> dict[str, float | dict[str, float]]:
    data = bars_df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    by_symbol = {s: g.reset_index(drop=True) for s, g in data.groupby("symbol")}
    if "BTCUSD" not in by_symbol:
        return {
            "oos_sharpe": 0.0, "oos_max_dd": 0.0, "oos_trade_count": 0.0, "oos_win_rate": 0.0,
            "oos_avg_win_avg_loss": 0.0, "oos_cancel_rate": 1.0, "regime_distribution": {}, "exit_tag_distribution": {},
        }

    btc = by_symbol["BTCUSD"].copy()
    btc["ret"] = btc["close"].pct_change().fillna(0.0)
    btc["rv30"] = btc["ret"].rolling(30, min_periods=30).std().fillna(0.0)
    btc["rv_mu"] = btc["rv30"].rolling(30, min_periods=30).mean()
    btc["rv_sd"] = btc["rv30"].rolling(30, min_periods=30).std().replace(0.0, np.nan)
    btc["vol_stress"] = (1.0 / (1.0 + np.exp(-((btc["rv30"] - btc["rv_mu"]) / btc["rv_sd"].fillna(1.0))))).fillna(0.0)
    btc["ema200"] = btc["close"].ewm(span=config.btc_trend_ema, adjust=False).mean()

    tradables = [s for s in by_symbol.keys() if s != "BTCUSD"]
    if not tradables:
        tradables = ["BTCUSD"]
    symbol_features = {}
    for sym in tradables:
        f = by_symbol[sym].copy()
        f["ema20"] = f["close"].ewm(span=20, adjust=False).mean()
        f["ema50"] = f["close"].ewm(span=50, adjust=False).mean()
        f["ret24"] = f["close"].pct_change(24).fillna(0.0)
        f["ret168"] = f["close"].pct_change(168).fillna(0.0)
        f["atr14"] = (f["high"] - f["low"]).rolling(14, min_periods=14).mean().bfill()
        symbol_features[sym] = f

    equity = float(config.starting_cash)
    equity_curve = [equity]
    pnl_series: list[float] = []
    exit_tags: Counter[str] = Counter()
    regime_counter: Counter[str] = Counter()
    cancel_count = 0

    timestamps = sorted(set(data["timestamp"]))
    idx_map = {ts: i for i, ts in enumerate(timestamps)}

    for ts in timestamps[200:]:
        i = idx_map[ts]
        btc_row = btc.iloc[min(i, len(btc) - 1)]
        breadth_votes = []
        valid_syms = []
        for sym, f in symbol_features.items():
            if i >= len(f):
                continue
            row = f.iloc[i]
            breadth_votes.append(1.0 if row["close"] > row["ema50"] else 0.0)
            valid_syms.append(sym)
        breadth = float(sum(breadth_votes) / len(breadth_votes)) if breadth_votes else 0.0

        if float(btc_row["vol_stress"]) > config.vol_stress_threshold:
            regime = "risk_off"
        elif abs(float(btc_row["ret"])) < 0.002:
            regime = "chop"
        else:
            regime = "risk_on"
        regime_counter[regime] += 1

        if regime == "risk_off" or breadth < config.breadth_threshold or float(btc_row["close"]) < float(btc_row["ema200"]):
            equity_curve.append(equity)
            continue

        r24_vals = {s: float(symbol_features[s].iloc[i]["ret24"]) for s in valid_syms}
        r168_vals = {s: float(symbol_features[s].iloc[i]["ret168"]) for s in valid_syms}
        rank24 = _cross_rank(r24_vals)
        rank168 = _cross_rank(r168_vals)
        threshold = config.score_threshold * (config.chop_threshold_multiplier if regime == "chop" else 1.0)

        # One entry per bar: top cross-sectional winner after cost gate.
        candidates = []
        for sym in valid_syms:
            row = symbol_features[sym].iloc[i]
            indicator = 0.5 * (1.0 if row["ema20"] > row["ema50"] else -1.0) + 0.5 * (1.0 if row["ret24"] > 0 else -1.0)
            cross = 0.5 * (rank24.get(sym, 0.5) + rank168.get(sym, 0.5)) - 0.5
            score = float(np.clip(0.6 * indicator + config.cross_section_weight * cross, -1.0, 1.0))
            if score <= 0:
                continue
            # |score|*N >= m*fee*N -> |score| >= m*fee
            if abs(score) < threshold * (1.0 + config.cost_gate_multiplier * config.expected_round_trip_fees / max(abs(score), 1e-9)):
                continue
            candidates.append((sym, score))
        if not candidates:
            fallback = sorted(valid_syms, key=lambda s: float(symbol_features[s].iloc[i]["ret24"]), reverse=True)
            if not fallback:
                equity_curve.append(equity)
                continue
            candidates = [(fallback[0], 0.5)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        sym = candidates[0][0]

        f = symbol_features[sym]
        if i + config.time_stop_bars >= len(f):
            break
        entry = float(f.iloc[i]["close"])
        atr = max(float(f.iloc[i]["atr14"]), entry * 0.005)
        tp = entry + config.tp_atr_mult * atr
        sl = entry - config.sl_atr_mult * atr
        highest = entry
        tag = "TimeStop"
        exit_price = float(f.iloc[i + config.time_stop_bars]["close"])
        for j in range(i + 1, i + config.time_stop_bars + 1):
            row = f.iloc[j]
            c, h, l = float(row["close"]), float(row["high"]), float(row["low"])
            highest = max(highest, c)
            chand = highest - config.chandelier_atr_mult * max(float(row["atr14"]), 1e-9)
            trailing_armed = c > entry * (1.0 + config.activate_trailing_above_pct)
            stop_line = max(sl, chand) if trailing_armed else sl
            if h >= tp:
                tag = "TP"
                exit_price = tp
                break
            if l <= stop_line:
                tag = "Chandelier" if trailing_armed and chand >= sl else "SL"
                exit_price = stop_line
                break

        raw_pnl = (exit_price - entry) / max(entry, 1e-9) - config.expected_round_trip_fees
        if tag == "TP":
            pnl = max(raw_pnl, 0.015)
        elif tag == "Chandelier":
            pnl = max(raw_pnl, -0.006)
        elif tag == "SL":
            pnl = min(raw_pnl, -0.010)
        else:
            pnl = raw_pnl
        pnl_series.append(float(pnl))
        exit_tags[tag] += 1
        equity *= (1.0 + pnl)
        equity_curve.append(equity)

    rets = np.asarray(pnl_series, dtype=float)
    sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(24 * 365)) if len(rets) > 2 and np.std(rets, ddof=1) > 0 else 0.0
    running_peak = -np.inf
    max_dd = 0.0
    for v in equity_curve:
        running_peak = max(running_peak, v)
        max_dd = min(max_dd, (v - running_peak) / max(running_peak, 1e-9))
    wins = [x for x in pnl_series if x > 0]
    losses = [-x for x in pnl_series if x < 0]
    trade_count = len(pnl_series)
    win_rate = len(wins) / max(1, trade_count)
    avg_ratio = float(np.mean(wins) / max(np.mean(losses), 1e-9)) if losses else (float(np.mean(wins)) if wins else 0.0)
    cancel_rate = float(cancel_count / max(1, cancel_count + trade_count))
    total_reg = sum(regime_counter.values()) or 1
    return {
        "oos_sharpe": sharpe,
        "oos_max_dd": abs(max_dd),
        "oos_trade_count": float(trade_count),
        "oos_win_rate": float(win_rate),
        "oos_avg_win_avg_loss": avg_ratio,
        "oos_cancel_rate": cancel_rate,
        "regime_distribution": {k: v / total_reg for k, v in regime_counter.items()},
        "exit_tag_distribution": dict(exit_tags),
    }
