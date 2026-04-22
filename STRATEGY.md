# Strategy

## Edge hypothesis
The default runtime uses cross-sectional momentum in crypto majors over roughly 21–63 day horizons, rebalanced weekly. The hypothesis is that relative-strength persistence (Jegadeesh & Titman style momentum) can survive in liquid crypto cross-sections, consistent with published evidence for crypto return factors (e.g., Liu & Tsyvinski, 2021).

## Expected performance
Target expectations are modest: roughly 5–15% annualized net of fees, 15–25% max drawdown, and long-run Sharpe around 0.4–0.8 in favorable regimes. This is not a get-rich-quick system.

## Kill conditions
Halt and reassess if realized 60-day Sharpe drops below -1 or if drawdown exceeds 25%. Automated protection is already present via `DrawdownCircuitBreaker` (configured at -12% over 168 bars).

## What this strategy is NOT
- Not a high-frequency edge
- Not arbitrage
- Not suitable for accounts below $250 (fees dominate)
- Not suitable without at least 6 months of paper-trading first

## Pre-deployment checklist
- Run walk-forward backtests covering 2022–2025
- Paper-trade for at least 60 days
- Verify backtest vs paper P&L are within 50%
- Only then deploy capital you can fully afford to lose
