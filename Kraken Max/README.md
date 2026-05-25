# Kraken Max

Aggressive **long-only**, **cash account** (no margin) crypto strategy for **QuantConnect** live/paper on **Kraken**, tuned for Canadian clients.

## Objective

Designed for **high convexity** — concentrated momentum + breakout + dip-buy + logistic ML ensemble — with explicit acceptance of **large drawdowns** in exchange for upside tail exposure. The nominal story is growing **$1,000 → $20,000**; that requires exceptional market conditions and carries a **high probability of total loss**. Treat this as experimental capital only.

## Canada / Kraken compliance

- `AccountType.Cash` — no leverage, no shorts (`ENABLE_SHORTS = False`)
- Universe prioritizes **BTC, ETH, LTC, BCH** (no CAD net-purchase limits on Kraken Canada)
- Alts are liquidity-filtered; meme/low-liquidity names are blacklisted
- If you are subject to **$30k CAD alt limits**, prefer unlimited pairs or upgrade investor tier per [Kraken Canada limits](https://support.kraken.com/articles/15568473780628-cad-net-purchase-limits-for-certain-cryptocurrencies-in-canada)

## Strategy stack

| Sleeve | Weight | Idea |
|--------|--------|------|
| Momentum | 35% | Risk-adjusted 7d/21d trend + cross-section rank |
| Breakout | 25% | Donchian + volume surge |
| Dip-in-trend | 15% | RSI pullback only when trend_quality > 0 |
| ML logistic | 25% | `ml_weights.json` + online bias updates |

**Regimes:** bull (up to 98% deployed) → neutral (75%) → bear (35%, BTC/ETH bias) → chaos (flat).

**Risk:** -8% hard stop, -12% catastrophic, chandelier trail after +10%, portfolio halt at **-28%** drawdown.

## QuantConnect deployment

1. Create a new QC project and upload all files in this folder (keep `main.py` as the algorithm entry).
2. Set brokerage: **Kraken**, **Cash**, verification tier matching your account.
3. Dataset: **Kraken Crypto Price** (`Market.Kraken`, hourly resolution).
4. Recommended path: backtest **2022–2025**, paper trade **60+ days**, then live with capital you can lose entirely.

```python
# Entry class
from main import KrakenMaxAlgorithm
```

## Configuration

Edit `config.py`:

- `starting_cash = 1000.0`
- `top_k`, `max_position_pct`, `rebalance_hours` — aggression knobs
- `drawdown_halt_pct` — portfolio circuit breaker (default -28%)

## ML weights

- Default weights: `ml_weights.json`
- Refit offline: `python research/train_weights.py --csv your_features.csv`

## Local tests

From repo root:

```bash
pip install -r requirements-dev.txt
pytest tests/test_kraken_max.py -q
```

## Disclaimer

Past backtests do not guarantee future results. **20× is not a forecast** — it is a stretch goal that implies sustained bull markets, favorable fills, and survivable drawdowns. Fees, slippage, halts, and regulatory limits can materially reduce returns or prevent deployment.
