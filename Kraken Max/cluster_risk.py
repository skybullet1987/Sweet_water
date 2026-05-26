from __future__ import annotations

from config import CONFIG, KrakenMaxConfig

# Beta / narrative clusters for portfolio-level caps (v7)
DEFAULT_CLUSTERS: dict[str, tuple[str, ...]] = {
    "major": ("BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"),
    "l1": ("SOLUSD", "AVAXUSD", "DOTUSD", "ADAUSD", "ATOMUSD"),
    "defi": ("LINKUSD", "UNIUSD", "AAVEUSD"),
    "meme": ("DOGEUSD", "SHIBUSD", "PEPEUSD"),
    "legacy": ("XRPUSD", "XLMUSD", "ETCUSD"),
}


def cluster_for_ticker(ticker: str, clusters: dict[str, tuple[str, ...]] | None = None) -> str:
    mapping = clusters or DEFAULT_CLUSTERS
    for name, members in mapping.items():
        if ticker in members:
            return name
    return "other"


def filter_cluster_caps(
    ranked: list[tuple[str, float]],
    *,
    current_holdings: list[str] | None = None,
    max_per_cluster: int | None = None,
    clusters: dict[str, tuple[str, ...]] | None = None,
    config: KrakenMaxConfig = CONFIG,
) -> list[str]:
    """Greedy pick from ranked list respecting per-cluster position caps."""
    cap = int(max_per_cluster if max_per_cluster is not None else config.max_positions_per_cluster)
    counts: dict[str, int] = {}
    for t in current_holdings or []:
        c = cluster_for_ticker(t, clusters)
        counts[c] = counts.get(c, 0) + 1
    selected: list[str] = []
    for ticker, _score in ranked:
        c = cluster_for_ticker(ticker, clusters)
        if counts.get(c, 0) >= cap:
            continue
        selected.append(ticker)
        counts[c] = counts.get(c, 0) + 1
    return selected


def max_cluster_exposure(weights: dict[str, float], clusters: dict[str, tuple[str, ...]] | None = None) -> dict[str, float]:
    """Sum ERC/target weights by cluster for telemetry."""
    mapping = clusters or DEFAULT_CLUSTERS
    out: dict[str, float] = {}
    for ticker, w in weights.items():
        c = cluster_for_ticker(ticker, mapping)
        out[c] = out.get(c, 0.0) + float(w)
    return out
