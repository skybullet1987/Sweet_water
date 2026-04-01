# region imports
from AlgorithmImports import *
# endregion


class RealisticCryptoSlippage(ISlippageModel):
    """
    Realistic crypto slippage model for backtesting.
    Volume-aware, calibrated against empirical Kraken fill data.

    NOTE: __init__ takes only self (no extra params) — required by PythonNet
    when inheriting from C# interfaces like ISlippageModel.

    Backtest realism additions:
    - max_slippage_pct raised to 2.0% (was 0.5%) for realistic altcoin costs.
    - Synthetic spread floor applied when bid/ask unavailable (typical in backtest).
    """

    def __init__(self):
        self.base_slippage_pct = 0.0010      # 0.10%
        self.volume_impact_factor = 0.15
        self.max_slippage_pct = 0.0200       # 2.00% cap (was 0.50%)

    def GetSlippageApproximation(self, asset, order):
        try:
            price = asset.Price
            if price <= 0:
                return 0

            slippage_pct = self.base_slippage_pct

            # Spread component: use real bid/ask when available, otherwise
            # apply a synthetic floor based on the price tier (backtest realism).
            bid = asset.BidPrice if hasattr(asset, 'BidPrice') else 0
            ask = asset.AskPrice if hasattr(asset, 'AskPrice') else 0
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    spread_cost = (ask - bid) / (2 * mid)
                    slippage_pct += spread_cost
            else:
                # No real quote data (typical in backtest) — add a synthetic
                # half-spread floor calibrated to Kraken empirical data.
                if price < 0.01:
                    slippage_pct += 0.0100    # 1.00% half-spread for micro-cap
                elif price < 0.10:
                    slippage_pct += 0.0050    # 0.50% half-spread
                elif price < 1.0:
                    slippage_pct += 0.0025    # 0.25% half-spread
                elif price < 10.0:
                    slippage_pct += 0.0015    # 0.15% half-spread
                elif price < 100.0:
                    slippage_pct += 0.0008    # 0.08% half-spread
                else:
                    slippage_pct += 0.0005    # 0.05% half-spread (BTC/ETH tier)

            volume = asset.Volume if hasattr(asset, 'Volume') else 0
            if volume > 0:
                order_value = abs(order.Quantity) * price
                volume_value = volume * price
                if volume_value > 0:
                    participation_rate = order_value / volume_value
                    volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                    slippage_pct += volume_impact

            # Price tier multipliers for overall slippage (thin books on cheap coins)
            if price < 0.01:
                slippage_pct *= 4.0
            elif price < 0.10:
                slippage_pct *= 2.5
            elif price < 1.0:
                slippage_pct *= 1.8
            elif price < 10.0:
                slippage_pct *= 1.2

            slippage_pct = min(slippage_pct, self.max_slippage_pct)
            return price * slippage_pct

        except Exception:
            return asset.Price * self.base_slippage_pct if asset.Price > 0 else 0

