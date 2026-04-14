"""
candidate_ranking.py — Explicit candidate-quality ranking helpers.

Ranks already-valid entry candidates using transparent component scores so
capital is allocated to the best opportunities first when constrained.
"""

RECENT_SYMBOL_QUALITY_PNL_SCALE = 0.03
CORRELATION_PENALTY_FLOOR = 0.60
CORRELATION_PENALTY_RANGE = 0.35
EXPECTED_COST_EPS = 1e-9
SCORE_SCALE_MIN = 1e-9

RANK_COMPONENT_WEIGHTS = {
    'setup_confidence': 0.25,
    'regime_confidence': 0.18,
    'edge_vs_cost': 0.14,
    'spread_quality': 0.08,
    'symbol_quality': 0.08,
    'correlation_penalty': 0.07,
    'recent_failure_penalty': 0.10,
}


def _clamp(v, lo, hi):
    """Clamp a numeric value into the closed interval [lo, hi]."""
    return max(lo, min(hi, v))


def _get_recent_symbol_quality(algo, symbol_value):
    """Return bounded [-1, 1] quality score from recent same-symbol trade outcomes."""
    perf = getattr(algo, '_symbol_performance', {})
    if symbol_value not in perf:
        return 0.0
    recent = perf[symbol_value]
    if len(recent) < 3:
        return 0.0
    n = min(6, len(recent))
    recent_slice = list(recent)[-n:]
    avg = sum(recent_slice) / n
    # ±RECENT_SYMBOL_QUALITY_PNL_SCALE recent average maps to ±1 quality score.
    return _clamp(avg / RECENT_SYMBOL_QUALITY_PNL_SCALE, -1.0, 1.0)


def _get_recent_failure_penalty(algo, symbol):
    """Return bounded [-1, 0] penalty for cooldown/clustered recent losses."""
    penalty = 0.0
    cooldowns = getattr(algo, '_symbol_entry_cooldowns', {})
    if symbol.Value in cooldowns and getattr(algo, 'Time', None) < cooldowns[symbol.Value]:
        penalty -= 1.0
    perf = getattr(algo, '_symbol_performance', {}).get(symbol.Value)
    threshold = int(getattr(algo, 'symbol_penalty_threshold', 3))
    if perf and len(perf) >= threshold and threshold > 0:
        recent_slice = list(perf)[-threshold:]
        recent_losses = sum(
            1 for p in recent_slice if p < 0
        )
        penalty -= recent_losses / float(threshold)
    return _clamp(penalty, -1.0, 0.0)


def _get_correlation_penalty(algo, symbol):
    """Return bounded [-1, 0] penalty from correlation with currently open positions."""
    corr_fn = getattr(algo, '_max_open_position_correlation', None)
    if corr_fn is None:
        return 0.0
    corr = corr_fn(symbol)
    if corr is None:
        return 0.0
    # No penalty below CORRELATION_PENALTY_FLOOR; full penalty at floor+range.
    raw = (corr - CORRELATION_PENALTY_FLOOR) / CORRELATION_PENALTY_RANGE
    return -_clamp(raw, 0.0, 1.0)


def _compute_ranking_components(algo, cand, base_score, score_scale):
    """Build transparent weighted ranking components for one trade candidate."""
    symbol = cand['symbol']
    spread_pct = cand.get('spread_pct')
    expected_move = cand.get('expected_move_pct')
    expected_cost = cand.get('expected_cost_pct')
    router = getattr(algo, '_regime_router', None)
    regime_conf = float(router.get_confidence() if router is not None else 0.5)
    setup_conf = _clamp(base_score / max(score_scale, SCORE_SCALE_MIN), 0.0, 1.0)

    edge_quality = 0.0
    if expected_move is not None and expected_cost is not None and expected_cost > EXPECTED_COST_EPS:
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
        'setup_confidence': setup_conf * RANK_COMPONENT_WEIGHTS['setup_confidence'],
        'regime_confidence': (regime_conf - 0.5) * RANK_COMPONENT_WEIGHTS['regime_confidence'],
        'edge_vs_cost': edge_quality * RANK_COMPONENT_WEIGHTS['edge_vs_cost'],
        'spread_quality': spread_quality * RANK_COMPONENT_WEIGHTS['spread_quality'],
        'symbol_quality': symbol_quality * RANK_COMPONENT_WEIGHTS['symbol_quality'],
        'correlation_penalty': corr_penalty * RANK_COMPONENT_WEIGHTS['correlation_penalty'],
        'recent_failure_penalty': failure_penalty * RANK_COMPONENT_WEIGHTS['recent_failure_penalty'],
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
