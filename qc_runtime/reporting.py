from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from config import CONFIG, StrategyConfig

TRADE_STEP_BARS = 12
TRADE_HOLD_BARS = 3
COST_PER_TRADE = 0.0005
REGIME_FLOOR = 0.05
REGIME_FALLBACK_DISTRIBUTION = {"risk_on": 0.45, "risk_off": 0.25, "chop": 0.30}


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

    def on_order_event(self, event: dict[str, float | str]) -> None:
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
