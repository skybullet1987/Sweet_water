"""
candidate_ranking.py — Explicit candidate-quality ranking helpers.

Ranks already-valid entry candidates using transparent component scores so
capital is allocated to the best opportunities first when constrained.
"""

import itertools


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _get_recent_symbol_quality(algo, symbol_value):
    perf = getattr(algo, '_symbol_performance', {})
    if symbol_value not in perf:
        return 0.0
    recent = perf[symbol_value]
    if len(recent) < 3:
        return 0.0
    n = min(6, len(recent))
    avg = sum(itertools.islice(reversed(recent), n)) / n
    # ±3% recent average maps to ±1 quality score.
    return _clamp(avg / 0.03, -1.0, 1.0)


def _get_recent_failure_penalty(algo, symbol):
    penalty = 0.0
    cooldowns = getattr(algo, '_symbol_entry_cooldowns', {})
    if symbol.Value in cooldowns and getattr(algo, 'Time', None) < cooldowns[symbol.Value]:
        penalty -= 1.0
    perf = getattr(algo, '_symbol_performance', {}).get(symbol.Value)
    threshold = int(getattr(algo, 'symbol_penalty_threshold', 3))
    if perf and len(perf) >= threshold and threshold > 0:
        recent_losses = sum(
            1 for p in itertools.islice(reversed(perf), threshold) if p < 0
        )
        penalty -= recent_losses / float(threshold)
    return _clamp(penalty, -1.0, 0.0)


def _get_correlation_penalty(algo, symbol):
    corr_fn = getattr(algo, '_max_open_position_correlation', None)
    if corr_fn is None:
        return 0.0
    corr = corr_fn(symbol)
    if corr is None:
        return 0.0
    # No penalty below 0.60; full penalty at >=0.95.
    raw = (corr - 0.60) / 0.35
    return -_clamp(raw, 0.0, 1.0)


def _compute_ranking_components(algo, cand, base_score, score_scale):
    symbol = cand['symbol']
    spread_pct = cand.get('spread_pct')
    expected_move = cand.get('expected_move_pct')
    expected_cost = cand.get('expected_cost_pct')
    router = getattr(algo, '_regime_router', None)
    regime_conf = float(router.get_confidence() if router is not None else 0.5)
    setup_conf = _clamp(base_score / max(score_scale, 1e-9), 0.0, 1.0)

    edge_quality = 0.0
    if expected_move is not None and expected_cost is not None and expected_cost > 0:
        edge_ratio = (expected_move - expected_cost) / expected_cost
        edge_quality = _clamp(edge_ratio / 2.0, -1.0, 1.0)

    spread_quality = 0.0
    if spread_pct is not None:
        spread_cap = max(float(getattr(algo, 'max_spread_pct', 0.005)), 1e-6)
        spread_quality = _clamp(1.0 - (spread_pct / spread_cap), -1.0, 1.0)

    symbol_quality = _get_recent_symbol_quality(algo, symbol.Value)
    corr_penalty = _get_correlation_penalty(algo, symbol)
    failure_penalty = _get_recent_failure_penalty(algo, symbol)

    # Keep setup score dominant; use small explicit additive nudges.
    components = {
        'setup_confidence': setup_conf * 0.25,
        'regime_confidence': (regime_conf - 0.5) * 0.18,
        'edge_vs_cost': edge_quality * 0.14,
        'spread_quality': spread_quality * 0.08,
        'symbol_quality': symbol_quality * 0.08,
        'correlation_penalty': corr_penalty * 0.07,
        'recent_failure_penalty': failure_penalty * 0.10,
        'existing_rank_overlay': float(cand.get('rank_adj', 0.0)),
    }
    return components


def rank_candidates(algo, candidates, base_score_key, score_scale):
    """
    Return candidates sorted by explicit rank_score (descending), including a
    ``rank_components`` breakdown for reporting/debugging.
    """
    ranked = []
    for cand in candidates:
        base_score = float(cand.get(base_score_key, 0.0))
        components = _compute_ranking_components(algo, cand, base_score, score_scale)
        rank_score = base_score + sum(components.values())
        enriched = dict(cand)
        enriched['rank_components'] = components
        enriched['rank_score'] = rank_score
        ranked.append(enriched)

    ranked.sort(
        key=lambda c: (
            c.get('rank_score', 0.0),
            c.get(base_score_key, 0.0),
            c['symbol'].Value,
        ),
        reverse=True
    )
    return ranked
