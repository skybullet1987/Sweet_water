# Next-Generation Strategy Architecture Scaffold

This repository now includes a **parallel modular architecture** under `nextgen/` to start the transition toward a regime-aware, cost-aware, multi-sleeve system.

## Legacy vs nextgen

- Legacy live/backtest path remains in root modules (`main.py`, `execution.py`, `scoring.py`, etc.) and is not removed.
- New scaffold is additive and isolated in `nextgen/*`.
- Future migration should route legacy strategy decisions through `nextgen` risk/accounting paths.

## Implemented now

- Modular package layout:
  - `nextgen/data`
  - `nextgen/features`
  - `nextgen/regime`
  - `nextgen/signals`
  - `nextgen/portfolio`
  - `nextgen/risk`
  - `nextgen/execution`
  - `nextgen/accounting`
  - `nextgen/research`
  - `nextgen/monitoring`
  - `nextgen/config`
- Typed core dataclasses and protocols in `nextgen/core`.
- Unified risk engine skeleton with volatility scaling, per-position caps, gross/net exposure caps, drawdown throttle, and kill-switch rejections.
- Probabilistic regime engine skeleton with smoothing + hysteresis/persistence behavior.
- Signal sleeve scaffolding:
  - cross-sectional momentum
  - trend breakout
  - pullback-in-trend
  - selective mean reversion
- Portfolio allocation layer that consumes sleeve outputs and regime constraints.
- Central accounting scaffold for fills, fees/funding, and realized PnL interfaces.
- Research harness foundation with experiment config, immutable run metadata, run result schema, and stress scenario placeholders.
- Monitoring scaffold for drift/fill-rate anomalies.

## Scaffolded for future work (TODO direction)

- Full migration of legacy live execution flow into these interfaces.
- Empirical calibration of costs/slippage/funding and risk limits.
- Full walk-forward and scenario library integration with real data adapters.
- Portfolio optimization and correlation-aware budgeting.

## Running tests

The project currently uses Python `unittest` tests for this scaffold:

```bash
cd <project_root>
python -m unittest discover -s tests -p "test_*.py" -v
```
