# Kraken Max

Aggressive **long-only**, **cash account** (no margin) crypto strategy for **QuantConnect** live/paper on **Kraken**, tuned for Canadian clients.

**Current version: v8** (refactored layout) — same features, **10 Python modules** for QuantConnect (each file &lt; 63k chars).

## Package layout (QC deploy)

Upload the entire **`Kraken Max/`** folder plus **`qc_runtime/`**. Core code lives in:

| File | Role |
|------|------|
| `main.py` | `KrakenMaxAlgorithm` entry point |
| `config.py` | `KrakenMaxConfig` / `CONFIG` |
| `core.py` | Features, universe, correlation, scalper, sizing, ensemble |
| `regime.py` | Regime engines, QC gates, per-regime weights |
| `risk.py` | Stops, portfolio risk, clusters, ERC optimizer |
| `execution.py` | Limits, bridge to `qc_runtime`, brackets |
| `data.py` | Sentiment, funding/OI feeds, cross-venue lead |
| `kraken_ml.py` | Logistic scorer + trainer |
| `kraken_ops.py` | Fill tracker, drift, scorecard, telemetry, alerts |
| `workflow.py` | Walk-forward, validation, auto-revalidation |

`research/` holds CLI scripts only (not required on QC cloud). JSON/CSV data files unchanged.

## Objective

Designed for **high convexity** — concentrated momentum + breakout + dip-buy + logistic ML ensemble — with explicit acceptance of **large drawdowns** in exchange for upside tail exposure. The nominal story is growing **$1,000 → $20,000**; that requires exceptional market conditions and carries a **high probability of total loss**. Treat this as experimental capital only.

## Canada / Kraken compliance

- `AccountType.Cash` — no leverage, no shorts (`ENABLE_SHORTS = False`)
- Universe prioritizes **BTC, ETH, LTC, BCH** (no CAD net-purchase limits on Kraken Canada)
- Alts are liquidity-filtered; meme/low-liquidity names are blacklisted
- If you are subject to **$30k CAD alt limits**, prefer unlimited pairs or upgrade investor tier per [Kraken Canada limits](https://support.kraken.com/articles/15568473780628-cad-net-purchase-limits-for-certain-cryptocurrencies-in-canada)

## Strategy stack (v8)

| Layer | Module | Description |
|-------|--------|-------------|
| **Regime walk-forward** | `workflow.py` | Grid-search `w_*` per bull/neutral/bear/chaos from BTC labels |
| **Auto revalidation** | `workflow.py` | Monthly: walk-forward + validation + baseline + regime weights → ObjectStore |
| **Native 15m export** | `research/export_qc_minute_bars.py` | QC `Resolution.Minute` → consolidate (not hourly upsample) |
| **Dashboard digest** | `kraken_ops.py` | Text + HTML snapshot; daily Telegram/Discord via alerts |

### Regime weight optimization (local)

```bash
python3 "Kraken Max/research/optimize_regime_weights.py" --csv kraken_hourly.csv
```

### Native 15m in QC Research

```python
from export_qc_minute_bars import export_minute_history_qc
bars = export_minute_history_qc(qb, ["BTCUSD","ETHUSD"], start, end, bar_minutes=15)
```

### ObjectStore (v8)

- `kraken_max_regime_weights.json` — optimized per-regime weights
- `kraken_max_revalidation.json` — last monthly revalidation summary
- `kraken_max_dashboard.txt` / `.html` — daily digest

## Strategy stack (v7)

| Layer | Module | Description |
|-------|--------|-------------|
| **Backtest validator** | `backtest_validator.py` | Walk-forward PASS/FAIL vs Sharpe, DD, trades, win-rate thresholds |
| **Paper scorecard** | `scorecard.py` | Live/paper metrics → ObjectStore; optional `PAPER_GATE` alert |
| **Calibrated costs** | `cost_model.py` | Edge gate uses live fill slippage + fill-rate spread adjustment |
| **Regime ensembles** | `regime_ensemble.py` + `regime_weights.json` | Different `w_*` in bull / neutral / bear / chaos |
| **Cluster risk** | `cluster_risk.py` | Max N positions per beta cluster (major, L1, defi, …) |

### Validation (local)

```bash
python3 "Kraken Max/research/run_backtest_validation.py" --csv kraken_hourly.csv --folds 3
```

Tune thresholds in `config.py`: `validation_min_sharpe`, `validation_max_drawdown`, etc.

### Paper gate (live)

After `paper_min_days` (default 30), alerts if live Sharpe, drawdown, or trade count fail `paper_*` thresholds.

## Strategy stack (v6)

| Layer | Module | Description |
|-------|--------|-------------|
| **Telemetry** | `telemetry.py` | Hourly ObjectStore snapshot: equity, regime, fill rate, drift, ERC weights |
| **Cross-venue lead** | `cross_venue.py` | Binance spot CSV lead nudges entry scores; execution stays on Kraken |
| **Fill submit wire** | `execution_bridge.py` | `track_order_submit()` → `FillTracker` on every limit entry |
| **15m walk-forward** | `walk_forward_engine.py` + `research/walk_forward_15m.py` | Bar-frequency-aware OOS optimization |
| **Auto baseline** | `baseline_manager.py` | Refresh drift baseline after ML retrain / walk-forward |
| **CI** | `.github/workflows/kraken_max_ci.yml` | v1–v6 tests + walk-forward smoke |

### v6 ObjectStore keys

- `kraken_max_telemetry.json` — latest dashboard snapshot
- `kraken_max_baseline_sharpe.json` — drift baseline Sharpe
- `kraken_max_baseline_meta.json` — last refresh source (walk_forward / ml_retrain)

### Cross-venue CSV

Place `data/binance_spot_lead.csv` with columns `symbol,timestamp,close` (Binance symbols e.g. `BTCUSDT`). Generate via `research/fetch_external_data.py` or your own pipeline.

### 15m walk-forward (local)

```bash
python3 "Kraken Max/research/qc_history_pipeline.py" --csv kraken_hourly.csv --to-15m --out kraken_15m.csv
python3 "Kraken Max/research/walk_forward_15m.py" --csv kraken_15m.csv --folds 3
```

## Strategy stack (v5)

| Layer | Module | Description |
|-------|--------|-------------|
| **15m bars** | `main.py` | Minute data consolidated to 15m (configurable) |
| **Shrinkage ERC** | `portfolio_optimizer.py` | Covariance shrinkage + turnover penalty vs prior weights |
| **Unified regime** | `regime_bridge.py` | Hurst/VR + `qc_runtime` vol/breadth/EMA30d gates |
| **Fill tracker** | `fill_tracker.py` | Limit fill rate + slippage bps alerts |
| **Drift monitor** | `drift_monitor.py` | Live Sharpe vs walk-forward baseline |
| **CI** | `.github/workflows/kraken_max_ci.yml` | Full test suite + walk-forward smoke on every PR |

Set walk-forward baseline Sharpe via `ensemble_weights.json` metrics or ObjectStore key `kraken_max_baseline_sharpe.json`.

## Strategy stack (v4)

| Layer | Module | Description |
|-------|--------|-------------|
| Funding + OI | `research/fetch_external_data.py` | Binance perp funding + open interest → CSV |
| ERC sizing | `portfolio_optimizer.py` | Equal-risk-contribution weights across picks |
| Advanced regime | `advanced_regime.py` | Hurst + variance-ratio on BTC (from `qc_runtime` logic) |
| Brackets | `brackets.py` | Stop-market SL + limit TP after each momentum entry |
| Alerts | `notifications.py` | Telegram/Discord via algorithm parameters |
| QC history | `research/qc_history_pipeline.py` | Multi-asset Kraken `History()` export for walk-forward |

### Live alert parameters (QuantConnect)

Set in algorithm **Parameters**:
- `telegram_webhook` — Telegram bot sendMessage URL
- `discord_webhook` — Discord channel webhook URL

## Strategy stack (v3)

| Layer | Module | Description |
|-------|--------|-------------|
| External data | `data_feeds.py` | QC `FearGreedIndex` + CSV fallback; funding CSV; cap-weight BTC dominance |
| Execution | `execution_bridge.py` | Uses `qc_runtime/execution.py` when available (limits, dust, stale escalate) |
| Walk-forward | `walk_forward_engine.py` | OOS Sharpe optimization → `ensemble_weights.json` |
| Research | `research/KrakenMax_V3_WalkForward.ipynb` | QC Research / local notebook |

## Strategy stack (v2)

| Sleeve | Weight | Idea |
|--------|--------|------|
| Momentum | 35% | Risk-adjusted 7d/21d trend + cross-section rank |
| Breakout | 25% | Donchian + volume surge |
| Dip-in-trend | 15% | RSI pullback only when trend_quality > 0 |
| ML logistic | 25% | `ml_weights.json` + **monthly walk-forward retrain** |
| **Scalper** (parallel) | 12% per slot × 2 | 6h mean-reversion in **ranging/neutral** regimes |

**v2 upgrades:**
- **Limit orders** at bid/ask with stale-limit escalation to market (Kraken maker bias)
- **Participation cap** — order size ≤ 12% of hourly dollar volume
- **Correlation filter** — greedy decorrelation so top 4 are not all SOL-beta clones
- **Sentiment regime** — fear/greed + BTC dominance proxies adjust deployment
- **ML auto-retrain** — monthly (+ when enough samples), persists to `ml_weights.json` / QC ObjectStore

**Regimes:** bull (up to 98% deployed) → neutral (75%) → bear (35%, BTC/ETH bias) → chaos (flat).

**Risk:** -8% hard stop, -12% catastrophic, chandelier trail after +10%, portfolio halt at **-28%** drawdown.

## QuantConnect deployment

1. Create a QC project and upload **all `.py` files from this folder** plus the **`qc_runtime/`** folder (as `qc_runtime/` inside the project, not one level up).
2. **Entry point:** `main.py` in the project root must expose `KrakenMaxAlgorithm`. If files live in a subfolder, copy `deploy/main.py` from the repo to the project root as `main.py`.
3. In the Algorithm Lab, open **Build** and confirm compile succeeds. Select class **`KrakenMaxAlgorithm`** if prompted.
4. Set brokerage: **Kraken**, **Cash**.
5. Dataset: **Kraken Crypto** — default config uses **hourly** bars (`use_sub_hour_bars = False`). Only set `use_sub_hour_bars = True` in `config.py` if you subscribe to **minute** data.
6. Backtest **2022–2025**, then paper, then live with risk capital only.

### Backtest does not start (checklist)

| Symptom | Fix |
|--------|-----|
| Compile error / red build | Open **Logs**; fix missing file or import. Ensure `kraken_ml.py` / `kraken_ops.py` are uploaded (names must be &gt; 3 characters). |
| Build OK but nothing runs | Confirm `KrakenMaxAlgorithm` is the active class in `main.py`. |
| Stuck on “Loading…” | Often **minute data** without a Minute subscription — keep `use_sub_hour_bars = False`. |
| No trades | Check **Debug** for `KRAKEN_MAX skip subscribe` — add Kraken crypto universe or fix tickers. |
| `qc_runtime` not found | Upload `qc_runtime/` **inside** the QC project directory (same level as `main.py`). |

```python
# Entry class (must be reachable from project main.py)
from main import KrakenMaxAlgorithm
```

## Configuration

Edit `config.py`:

- `starting_cash = 1000.0`
- `top_k`, `max_position_pct`, `rebalance_hours` — aggression knobs
- `drawdown_halt_pct` — portfolio circuit breaker (default -28%)

## ML weights

- Default weights: `ml_weights.json`
- **Live/backtest:** `MLTrainer` retrains every `ml_retrain_days` (30) when ≥ `ml_min_samples` (80) closed trades exist
- Refit offline: `python research/train_weights.py --csv your_features.csv`
- QC ObjectStore key: `kraken_max_ml_weights.json`

## v3 setup

```bash
# Refresh Fear & Greed CSV (alternative.me API)
python "Kraken Max/research/fetch_external_data.py"

# Walk-forward optimize ensemble weights
python "Kraken Max/research/walk_forward_optimize.py" \
  --csv your_hourly_bars.csv --folds 4

# Or open research/KrakenMax_V3_WalkForward.ipynb in QC Research
```

Upload **`Kraken Max/`** and **`qc_runtime/`** to QuantConnect for full execution bridge (recommended).

## Local tests

From repo root:

```bash
pip install -r requirements-dev.txt
pytest tests/test_kraken_max.py tests/test_kraken_max_v3.py tests/test_kraken_max_v4.py tests/test_kraken_max_v5.py -q
```

## Disclaimer

Past backtests do not guarantee future results. **20× is not a forecast** — it is a stretch goal that implies sustained bull markets, favorable fills, and survivable drawdowns. Fees, slippage, halts, and regulatory limits can materially reduce returns or prevent deployment.
