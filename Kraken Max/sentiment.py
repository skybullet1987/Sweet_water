from __future__ import annotations

from dataclasses import dataclass

from config import KrakenMaxConfig, CONFIG

try:
    from data_feeds import ExternalSentiment
except ImportError:
    from .data_feeds import ExternalSentiment  # type: ignore


@dataclass(frozen=True)
class SentimentSnapshot:
    """Combined proxy + external fear/greed, dominance, funding."""

    fear_greed: float  # 0 = extreme fear, 1 = extreme greed
    btc_dominance: float  # 0 = alt season, 1 = BTC leading
    funding_proxy: float  # positive = long-bias stress in majors
    fear_greed_index: float = 50.0  # raw 0-100
    data_source: str = "proxy"


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_sentiment(
    *,
    btc_features: dict[str, float],
    eth_features: dict[str, float] | None,
    breadth: float,
    median_rv: float,
    alt_median_mom_7d: float = 0.0,
    config: KrakenMaxConfig = CONFIG,
) -> SentimentSnapshot:
    btc_mom7 = float(btc_features.get("mom_7d", 0.0))
    btc_mom21 = float(btc_features.get("mom_21d", 0.0))
    eth_mom7 = float((eth_features or {}).get("mom_7d", btc_mom7))

    # BTC dominance proxy: BTC short-term strength vs alt basket.
    rel = btc_mom7 - alt_median_mom_7d
    dom = _clamp01(0.5 + rel * 2.5)

    # Fear/greed: breadth + trend + inverse vol stress.
    vol_penalty = _clamp01((median_rv - 0.35) / 0.9)
    trend_boost = _clamp01(0.5 + btc_mom21 * 2.0)
    fg = _clamp01(0.45 * breadth + 0.35 * trend_boost + 0.20 * (1.0 - vol_penalty))

    # Funding proxy: when BTC runs faster than ETH, long crowding risk rises.
    funding = _clamp01(0.5 + (btc_mom7 - eth_mom7) * 4.0)

    return SentimentSnapshot(
        fear_greed=fg,
        btc_dominance=dom,
        funding_proxy=funding,
        fear_greed_index=fg * 100.0,
        data_source="proxy",
    )


def merge_external_sentiment(
    proxy: SentimentSnapshot,
    external: ExternalSentiment | None,
    *,
    config: KrakenMaxConfig = CONFIG,
) -> SentimentSnapshot:
    if external is None:
        return proxy
    fg_norm = _clamp01(float(external.fear_greed_normalized))
    dom = _clamp01(0.6 * external.btc_dominance + 0.4 * proxy.btc_dominance)
    funding = _clamp01(0.5 * external.funding_stress + 0.5 * proxy.funding_proxy)
    return SentimentSnapshot(
        fear_greed=fg_norm,
        btc_dominance=dom,
        funding_proxy=funding,
        fear_greed_index=float(external.fear_greed_index),
        data_source=str(external.source_fg),
    )


def adjust_deployment_cap(
    base_cap: float,
    sentiment: SentimentSnapshot,
    regime_name: str,
    config: KrakenMaxConfig = CONFIG,
) -> float:
    cap = float(base_cap)
    if regime_name in {"chaos", "bear"}:
        return cap
    if sentiment.fear_greed <= float(config.fg_extreme_fear):
        cap *= 1.0 - float(config.sentiment_fear_cut)
    elif sentiment.fear_greed >= float(config.fg_extreme_greed) and regime_name == "bull":
        cap = min(0.99, cap + float(config.sentiment_greed_boost))
    if sentiment.btc_dominance >= float(config.btc_dom_high) and regime_name == "bull":
        cap = min(0.99, cap * 1.02)
    if sentiment.btc_dominance <= float(config.btc_dom_low) and regime_name == "neutral":
        cap = min(0.99, cap * 1.03)
    if sentiment.funding_proxy > 0.85:
        cap *= 0.95
    return max(0.0, min(0.99, cap))
