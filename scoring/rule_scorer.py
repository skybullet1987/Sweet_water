from __future__ import annotations

from typing import Mapping


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def score_symbol(features: Mapping[str, float], regime: str, btc_context: Mapping[str, float] | None = None) -> float:
    btc_context = btc_context or {}
    if regime == "risk_off":
        return 0.0

    if regime == "risk_on":
        votes = [
            1.0 if features.get("adx", 0.0) > 25 else 0.0,
            1.0 if features.get("ema20", 0.0) > features.get("ema50", 0.0) else 0.0,
            1.0 if features.get("macd_hist", 0.0) > 0 else 0.0,
            1.0 if features.get("aroon_osc", 0.0) > 0 else 0.0,
            1.0 if features.get("mfi", 50.0) > 50 else 0.0,
        ]
        score = sum(votes) / len(votes)
        score *= 1.0 if btc_context.get("btc_trend", 0.0) >= -0.01 else 0.8
        score = _clip(score)
        return score if score >= 0.40 else 0.0

    if regime == "chop":
        cci = float(features.get("cci", 0.0))
        bb_pos = float(features.get("bb_pos", 0.5))
        rsi = float(features.get("rsi", 50.0))

        long_votes = [
            1.0 if cci < -100 else 0.0,
            1.0 if bb_pos < 0.2 else 0.0,
            1.0 if rsi < 30 else 0.0,
        ]
        short_votes = [
            1.0 if cci > 100 else 0.0,
            1.0 if bb_pos > 0.8 else 0.0,
            1.0 if rsi > 70 else 0.0,
        ]

        long_score = sum(long_votes) / len(long_votes)
        short_score = sum(short_votes) / len(short_votes)
        signed = long_score - short_score
        signed = _clip(signed)
        return signed if abs(signed) >= 0.40 else 0.0

    return 0.0
