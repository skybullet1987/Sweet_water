"""Kraken Max — features, universe, ensemble, sizing (`core.py`)."""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig
from ml import MLScorer, load_ml_weights
from regime import (
    config_for_regime,
    load_regime_weights,
    load_regime_weights_from_object_store,
    load_regime_weights_merged,
)

# --- from features.py ---


import numpy as np
import pandas as pd

from config import CONFIG, KrakenMaxConfig

DEFAULT_LOOKBACK = 24 * 30 * 4


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_dn = dn.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_bar_features(frame: pd.DataFrame, config: KrakenMaxConfig = CONFIG) -> dict[str, float]:
    min_bars = max(60, int(config.feature_min_bars) // max(config.bph(), 1))
    if frame is None or len(frame) < min_bars:
        return {}
    df = frame.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return {}
        df[col] = df[col].astype(float)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    ret = close.pct_change().fillna(0.0)

    def _mom(hours: int) -> float:
        lookback = min(len(close) - 1, max(config.lookback_bars(hours), 1))
        base = float(close.iloc[-1 - lookback])
        if base <= 0:
            return 0.0
        return float(close.iloc[-1] / base - 1.0)

    mom_7d = _mom(7)
    mom_21d = _mom(21)
    mom_63d = _mom(63)
    mom_accel = mom_7d - mom_21d / 3.0

    rv_window = min(len(ret), config.lookback_bars(21))
    rv_21d = float(ret.tail(rv_window).std() * math.sqrt(24 * 365 * config.bph()))
    rv_21d_inv = 1.0 / max(rv_21d, 1e-6)

    ema_fast = _ema(close, config.lookback_bars(24))
    ema_slow = _ema(close, config.lookback_bars(24 * 5))
    trend_quality = float((ema_fast.iloc[-1] / ema_slow.iloc[-1]) - 1.0)

    donchian = float(high.tail(min(len(high), config.lookback_bars(20 * 24))).max())
    breakout_strength = float((close.iloc[-1] / donchian) - 1.0) if donchian > 0 else 0.0

    vol_med = float((close * volume).tail(min(len(close), config.lookback_bars(7 * 24))).median())
    vol_recent = float((close * volume).tail(min(len(close), config.lookback_bars(24))).median())
    volume_24h = vol_recent
    volume_surge = float(vol_recent / max(vol_med, 1e-9) - 1.0)

    rsi = float(_rsi(close).iloc[-1])
    rsi_pullback = max(0.0, (45.0 - rsi) / 45.0) if trend_quality > 0 else 0.0

    dd_63d = float((close.iloc[-1] / close.tail(min(len(close), config.lookback_bars(63 * 24))).max()) - 1.0)
    atr = float(_atr(high, low, close).iloc[-1])
    ema50 = float(_ema(close, 50).iloc[-1])
    ema200 = float(_ema(close, 200).iloc[-1])
    ret_1h = float(close.iloc[-1] / close.iloc[-max(2, 2)] - 1.0) if len(close) > 2 else 0.0
    look6 = min(len(close) - 1, config.lookback_bars(6))
    ret_6h = float(close.iloc[-1] / close.iloc[-1 - look6] - 1.0) if look6 > 0 else 0.0

    return {
        "mom_7d": mom_7d,
        "mom_21d": mom_21d,
        "mom_63d": mom_63d,
        "mom_accel": mom_accel,
        "rv_21d": rv_21d,
        "rv_21d_inv": rv_21d_inv,
        "trend_quality": trend_quality,
        "breakout_strength": breakout_strength,
        "volume_surge": volume_surge,
        "volume_24h": volume_24h,
        "rsi": rsi,
        "rsi_pullback": rsi_pullback,
        "dd_63d": dd_63d,
        "atr": atr,
        "close": float(close.iloc[-1]),
        "ema50": ema50,
        "ema200": ema200,
        "ret_1h": ret_1h,
        "ret_6h": ret_6h,
        "adx": 18.0,
    }


def btc_beta_vs(features: dict[str, float], btc_features: dict[str, float]) -> float:
    """Simple beta proxy: coin 7d momentum minus BTC 7d momentum."""
    return float(features.get("mom_7d", 0.0)) - float(btc_features.get("mom_7d", 0.0))


class FeatureCache:
    """Per-symbol rolling OHLCV cache for hourly Kraken bars."""

    def __init__(self, max_bars: int = DEFAULT_LOOKBACK) -> None:
        self.max_bars = max_bars
        self._bars: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_bars))

    def update(self, symbol_key: str, bar: Any) -> None:
        self._bars[symbol_key].append(
            {
                "time": bar.EndTime,
                "open": float(bar.Open),
                "high": float(bar.High),
                "low": float(bar.Low),
                "close": float(bar.Close),
                "volume": float(bar.Volume),
            }
        )

    def frame(self, symbol_key: str) -> pd.DataFrame:
        rows = list(self._bars.get(symbol_key, []))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def features(self, symbol_key: str) -> dict[str, float]:
        return compute_bar_features(self.frame(symbol_key))


def cross_section_ranks(feature_map: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    vals = [(s, float(f.get(key, 0.0))) for s, f in feature_map.items() if f]
    if not vals:
        return {}
    symbols, numbers = zip(*vals)
    series = pd.Series(numbers, index=symbols)
    ranks = series.rank(pct=True, method="average")
    return {str(s): float(ranks[s]) for s in ranks.index}

# --- from universe.py ---


import pandas as pd

from config import CONFIG

# Kraken Canada: BTC, ETH, LTC, BCH have no CAD net-purchase limits.
CANADA_UNLIMITED = frozenset({"BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"})

REFERENCE_SYMBOLS = ("BTCUSD", "ETHUSD")

BLACKLIST: frozenset[str] = frozenset({
    "PEPEUSD",
    "SHIBUSD",
    "BONKUSD",
    "FLOKIUSD",
    "WIFUSD",
    "BANANAS31USD",
    "CHILLHOUSEUSD",
    "SKLUSD",
})

# Liquid Kraken pairs — unlimited coins first, then high-liquidity alts.
KRAKEN_MAX_UNIVERSE = (
    "BTCUSD",
    "ETHUSD",
    "LTCUSD",
    "BCHUSD",
    "SOLUSD",
    "XRPUSD",
    "LINKUSD",
    "ADAUSD",
    "DOTUSD",
    "AVAXUSD",
    "ATOMUSD",
    "NEARUSD",
    "ARBUSD",
    "OPUSD",
    "INJUSD",
    "SUIUSD",
    "APTUSD",
    "SEIUSD",
    "TIAUSD",
    "RNDRUSD",
    "FETUSD",
    "UNIUSD",
    "AAVEUSD",
    "MKRUSD",
    "CRVUSD",
    "SNXUSD",
    "COMPUSD",
    "LDOUSD",
    "IMXUSD",
    "FILUSD",
    "ETCUSD",
    "ALGOUSD",
    "XLMUSD",
    "TRXUSD",
    "MANAUSD",
    "SANDUSD",
    "AXSUSD",
    "APEUSD",
    "DYDXUSD",
    "ENSUSD",
    "GRTUSD",
    "FLOWUSD",
    "RUNEUSD",
    "KSMUSD",
    "MINAUSD",
    "ZECUSD",
    "DASHUSD",
)

MIN_HOURLY_DOLLAR_VOLUME = 150_000.0
DEFAULT_UNIVERSE_SIZE = 24


def _median_dollar_volume(frame: pd.DataFrame) -> float:
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    return float((close * volume).median())


def _priority_boost(symbol: str) -> float:
    return 1e12 if symbol in CANADA_UNLIMITED else 0.0


def select_universe(
    history_provider: Callable[[str, object, object], pd.DataFrame],
    asof_date,
) -> list[str]:
    start = asof_date - timedelta(days=21)
    rows: list[tuple[str, float]] = []
    for symbol in KRAKEN_MAX_UNIVERSE:
        if symbol in BLACKLIST:
            continue
        frame = history_provider(symbol, start, asof_date)
        if frame is None or frame.empty:
            rows.append((symbol, 0.0))
            continue
        rows.append((symbol, _median_dollar_volume(frame)))

    liquid = [(s, v) for s, v in rows if v >= MIN_HOURLY_DOLLAR_VOLUME]
    pool = liquid if liquid else rows
    ranked = sorted(pool, key=lambda x: (_priority_boost(x[0]), x[1]), reverse=True)
    limit = max(1, min(int(CONFIG.universe_size), len(ranked)))
    return [s for s, _ in ranked[:limit]]

# --- from correlation.py ---


from config import CONFIG, KrakenMaxConfig


def hourly_returns(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or "close" not in frame.columns:
        return pd.Series(dtype=float)
    close = frame["close"].astype(float)
    return close.pct_change().dropna()


def return_correlation(
    cache,
    tickers: list[str],
    *,
    lookback_hours: int | None = None,
    min_samples: int | None = None,
) -> pd.DataFrame:
    lookback = int(lookback_hours or CONFIG.corr_lookback_hours)
    min_n = int(min_samples or CONFIG.min_corr_samples)
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        frame = cache.frame(ticker)
        if frame is None or frame.empty:
            continue
        rets = hourly_returns(frame.tail(lookback))
        if len(rets) >= min_n:
            series_map[ticker] = rets
    if len(series_map) < 2:
        return pd.DataFrame()
    aligned = pd.DataFrame(series_map).dropna(how="any")
    if aligned.shape[0] < min_n:
        return pd.DataFrame()
    return aligned.corr()


def max_corr_to_selected(ticker: str, selected: list[str], corr: pd.DataFrame) -> float:
    if not selected or corr.empty or ticker not in corr.columns:
        return 0.0
    vals = []
    for other in selected:
        if other == ticker or other not in corr.columns:
            continue
        try:
            vals.append(abs(float(corr.loc[ticker, other])))
        except Exception:
            continue
    return max(vals) if vals else 0.0


def filter_uncorrelated_picks(
    ranked: list[tuple[str, float]],
    cache,
    *,
    top_k: int | None = None,
    max_corr: float | None = None,
    config: KrakenMaxConfig = CONFIG,
) -> list[str]:
    """Greedy decorrelation: highest score first, skip high-beta clones."""
    k = int(top_k or config.top_k)
    cap = float(max_corr if max_corr is not None else config.max_pairwise_corr)
    tickers = [t for t, _ in ranked]
    corr = return_correlation(cache, tickers)
    chosen: list[str] = []
    for ticker, _score in ranked:
        if len(chosen) >= k:
            break
        if corr.empty:
            chosen.append(ticker)
            continue
        if max_corr_to_selected(ticker, chosen, corr) <= cap:
            chosen.append(ticker)
    return chosen

# --- from scalper_sleeve.py ---


from config import CONFIG, KrakenMaxConfig


def _z_score(closes: list[float], current: float) -> float:
    if len(closes) < 20:
        return 0.0
    window = closes[-20:]
    mu = sum(window) / len(window)
    var = sum((x - mu) ** 2 for x in window) / max(len(window) - 1, 1)
    sd = math.sqrt(max(var, 0.0))
    if sd <= 0:
        return 0.0
    return (current - mu) / sd


def scalper_regime(feats: dict[str, float], config: KrakenMaxConfig = CONFIG) -> str:
    close = float(feats.get("close", 0.0))
    ema50 = float(feats.get("ema50", 0.0))
    ema200 = float(feats.get("ema200", 0.0))
    adx = float(feats.get("adx", 15.0))
    if close <= 0 or ema50 <= 0 or ema200 <= 0:
        return "neutral"
    if adx < float(config.scalper_adx_range_max):
        return "ranging"
    if close > ema50 > ema200:
        return "uptrend_pullback"
    if close < ema50 < ema200:
        return "downtrend"
    return "neutral"


def build_scalper_features(frame) -> dict[str, float]:
    base = {}
    if frame is None or len(frame) < 30:
        return base
    close_s = frame["close"].astype(float)
    high_s = frame["high"].astype(float)
    low_s = frame["low"].astype(float)
    closes = close_s.tolist()
    c = float(close_s.iloc[-1])
    z = _z_score(closes, c)

    def _ret(h: int) -> float:
        idx = min(len(close_s) - 1, h)
        base_px = float(close_s.iloc[-1 - idx])
        return (c / base_px - 1.0) if base_px > 0 else 0.0

    ema50 = close_s.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close_s.ewm(span=200, adjust=False).mean().iloc[-1]
    tr = (high_s - low_s).abs()
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    delta = close_s.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up.iloc[-1] / max(dn.iloc[-1], 1e-9)
    rsi = float(100 - (100 / (1 + rs)))

    vol_recent = float((close_s * frame["volume"].astype(float)).tail(24).median())
    vol_base = float((close_s * frame["volume"].astype(float)).tail(20 * 24).median())
    vol_rel = vol_recent / max(vol_base, 1e-9)

    out = {
        "close": c,
        "z_20h": z,
        "rsi_14": rsi,
        "ret_1h": _ret(1),
        "ret_6h": _ret(6),
        "atr": atr,
        "ema50": float(ema50),
        "ema200": float(ema200),
        "adx": 15.0,
        "volume_rel_20h": vol_rel,
        "rv_21d": float(close_s.pct_change().tail(24 * 21).std() * math.sqrt(24 * 365)),
    }
    out["scalper_regime"] = scalper_regime(out)
    return out


def evaluate_scalper_entry(
    feats: dict[str, float],
    *,
    btc_ret_1h: float,
    btc_ret_6h: float,
    last_trade_hours: float,
    config: KrakenMaxConfig = CONFIG,
) -> tuple[bool, str]:
    regime = str(feats.get("scalper_regime", "neutral"))
    if regime not in {"ranging", "uptrend_pullback", "neutral"}:
        return False, f"regime_block:{regime}"
    z = float(feats.get("z_20h", 0.0))
    entry_z = float(config.scalper_relaxed_z_entry if regime == "ranging" else config.scalper_z_entry)
    if z > entry_z:
        return False, f"z_above:{z:.2f}"
    rsi = float(feats.get("rsi_14", 50.0))
    if not (config.scalper_rsi_min <= rsi <= config.scalper_rsi_max):
        return False, f"rsi_band:{rsi:.1f}"
    if rsi > float(config.scalper_rsi_long_max):
        return False, f"rsi_confirm:{rsi:.1f}"
    if btc_ret_1h < float(config.scalper_btc_1h_floor):
        return False, "btc_1h"
    if btc_ret_6h < float(config.scalper_btc_6h_floor):
        return False, "btc_6h"
    if last_trade_hours < float(config.scalper_anti_churn_hours):
        return False, "anti_churn"
    rv = float(feats.get("rv_21d", 0.0))
    if rv > 1.8:
        return False, f"rv_high:{rv:.2f}"
    return True, "OK"


def evaluate_scalper_exit(
    *,
    entry_price: float,
    entry_time: datetime,
    current_price: float,
    now: datetime,
    feats: dict[str, float],
    highest_close: float,
    entry_atr: float,
    config: KrakenMaxConfig = CONFIG,
) -> tuple[bool, str]:
    if entry_price <= 0 or current_price <= 0:
        return False, ""
    pnl = current_price / entry_price - 1.0
    if pnl <= float(config.scalper_hard_stop_pct):
        return True, "hard_stop"
    z = float(feats.get("z_20h", 0.0))
    if z >= float(config.scalper_overshoot_z):
        return True, "overshoot"
    if z >= float(config.scalper_meanrev_z):
        return True, "mean_revert"
    atr = max(entry_atr, entry_price * 0.008)
    risk = atr * 1.2
    r_mult = (current_price - entry_price) / max(risk, 1e-9)
    if r_mult >= float(config.scalper_tp_r):
        return True, "tp_r"
    high = max(highest_close, current_price)
    trail = high - 1.5 * atr
    if pnl > 0.01 and current_price <= trail:
        return True, "trail"
    hours = (now - entry_time).total_seconds() / 3600.0
    if hours >= float(config.scalper_time_stop_hours):
        return True, "time_stop"
    return False, ""

# --- from sizing.py ---


from config import CONFIG, KrakenMaxConfig

BPS = 10_000.0


class AggressiveSizer:
    def __init__(self, config: KrakenMaxConfig = CONFIG) -> None:
        self.config = config
        self.outcomes: deque[float] = deque(maxlen=40)

    def record_trade(self, pnl_fraction: float) -> None:
        self.outcomes.append(float(pnl_fraction))

    def _kelly(self) -> float:
        if len(self.outcomes) < 8:
            return 0.18
        wins = [x for x in self.outcomes if x > 0]
        losses = [-x for x in self.outcomes if x < 0]
        if not losses:
            return min(float(self.config.kelly_cap), 0.25)
        p = len(wins) / len(self.outcomes)
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = sum(losses) / len(losses)
        b = avg_win / max(avg_loss, 1e-9)
        raw = p - (1.0 - p) / max(b, 1e-9)
        return min(float(self.config.kelly_cap), max(0.05, raw))

    def weight_for_score(self, score: float, rv_annual: float, rank_pct: float) -> float:
        if score < float(self.config.entry_score_threshold):
            return 0.0
        vol = max(float(rv_annual), 1e-6)
        vol_w = min(0.55, float(self.config.target_annual_vol) / vol)
        conviction = min(1.0, (score - self.config.entry_score_threshold) / 0.8)
        rank_boost = 0.5 + 0.5 * max(0.0, rank_pct - 0.5)
        raw = vol_w * self._kelly() * conviction * rank_boost
        return min(float(self.config.max_position_pct), max(0.0, raw))

    def passes_cost_gate(self, score: float, notional: float, algo=None) -> bool:
        if score <= 0 or notional <= 0:
            return False
        if algo is not None and bool(self.config.use_calibrated_costs):
            from ops import CalibratedCostModel

            return CalibratedCostModel(self.config).passes_edge_gate(score, notional, algo)
        fee = notional * float(self.config.expected_round_trip_fees)
        spread = notional * (float(self.config.assumed_spread_bps) / BPS)
        slip = notional * (float(self.config.assumed_slippage_bps) / BPS)
        cost_pct = (fee + spread + slip) / notional
        edge = score * float(self.config.edge_scale)
        return edge > cost_pct * float(self.config.edge_cost_multiplier)


def can_afford(algo, qty: float, price: float) -> tuple[bool, float, float]:
    if qty <= 0:
        return True, 0.0, free_cash_usd(algo)
    safety = float(CONFIG.cash_safety_factor)
    required = float(qty) * max(float(price), 0.0)
    available = free_cash_usd(algo) * safety
    return required <= available + 1e-9, required, available


def free_cash_usd(algo) -> float:
    try:
        cash = float(algo.Portfolio.CashBook["USD"].Amount)
    except Exception:
        cash = float(getattr(algo.Portfolio, "Cash", 0.0) or 0.0)
    reserved = 0.0
    try:
        orders = algo.Transactions.GetOpenOrders()
    except Exception:
        orders = []
    for order in orders:
        qty = float(getattr(order, "Quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        px = float(getattr(order, "LimitPrice", 0.0) or 0.0) or float(getattr(order, "Price", 0.0) or 0.0)
        reserved += qty * max(px, 0.0)
    return max(0.0, cash - reserved)


def round_qty(qty: float, min_qty: float) -> float:
    if qty <= 0:
        return 0.0
    if min_qty <= 0:
        return qty
    steps = int(qty / min_qty)
    return steps * min_qty

# --- from ensemble.py ---


from config import KrakenMaxConfig, CONFIG

ENSEMBLE_WEIGHTS_PATH = Path(__file__).resolve().parent / "ensemble_weights.json"


def load_optimized_ensemble_weights(path: Path | None = None) -> dict[str, float]:
    target = path or ENSEMBLE_WEIGHTS_PATH
    if not target.exists():
        return {}
    blob = json.loads(target.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in (blob.get("ensemble") or {}).items()}


class AlphaEnsemble:
    def __init__(
        self,
        config: KrakenMaxConfig = CONFIG,
        ml: MLScorer | None = None,
        *,
        regime_weights: dict[str, dict[str, float]] | None = None,
        algo=None,
    ) -> None:
        opt = load_optimized_ensemble_weights()
        if opt:
            allowed = {"w_momentum", "w_breakout", "w_dip", "w_ml"}
            kwargs = {k: v for k, v in opt.items() if k in allowed}
            config = replace(config, **kwargs) if kwargs else config
        self.config = config
        self.ml = ml or MLScorer()
        self._regime_weights: dict[str, dict[str, float]] = {}
        if bool(config.use_regime_ensembles):
            if regime_weights is not None:
                self._regime_weights = regime_weights
            elif algo is not None:
                stored = load_regime_weights_from_object_store(algo)
                self._regime_weights = stored or (
                    load_regime_weights_merged() if bool(config.use_regime_wf_weights) else load_regime_weights()
                )
            else:
                self._regime_weights = (
                    load_regime_weights_merged() if bool(config.use_regime_wf_weights) else load_regime_weights()
                )

    def score_symbol(
        self,
        features: dict[str, float],
        *,
        rank_mom_21: float = 0.5,
        rank_breakout: float = 0.5,
        breadth: float = 0.5,
        btc_beta: float = 0.0,
        regime_name: str = "neutral",
    ) -> dict[str, float]:
        if not features:
            return {"final": -1e9, "momentum": 0.0, "breakout": 0.0, "dip": 0.0, "ml": 0.0}

        rv = max(float(features.get("rv_21d", 0.05)), 1e-6)
        momentum = (
            0.45 * float(features.get("mom_21d", 0.0)) / rv
            + 0.35 * float(features.get("mom_accel", 0.0)) / rv
            + 0.20 * (rank_mom_21 - 0.5)
        )
        breakout = (
            0.60 * float(features.get("breakout_strength", 0.0)) * 8.0
            + 0.25 * max(0.0, float(features.get("volume_surge", 0.0)))
            + 0.15 * (rank_breakout - 0.5)
        )
        dip = float(features.get("rsi_pullback", 0.0)) * max(0.0, float(features.get("trend_quality", 0.0)) * 5.0)

        ml_ctx = {
            "breadth": breadth,
            "btc_beta": btc_beta,
        }
        ml_score = self.ml.score(features, ml_ctx)

        cfg = config_for_regime(self.config, regime_name, self._regime_weights)
        w_m = float(cfg.w_momentum)
        w_b = float(cfg.w_breakout)
        w_d = float(cfg.w_dip)
        w_ml = float(cfg.w_ml)
        final = w_m * momentum + w_b * breakout + w_d * dip + w_ml * ml_score
        clip = float(self.config.score_clip)
        final = max(-clip, min(clip, final))
        return {
            "final": final,
            "momentum": momentum,
            "breakout": breakout,
            "dip": dip,
            "ml": ml_score,
        }