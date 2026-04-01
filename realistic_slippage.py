# region imports
from AlgorithmImports import *
# endregion


class RealisticCryptoSlippage:
    """
    Realistic crypto slippage model for backtesting.
    Volume-aware, calibrated against empirical Kraken fill data.

    Backtest realism additions:
    - max_slippage_pct raised to 2.0% (was 0.5%) for realistic altcoin costs.
    - Synthetic spread floor applied when bid/ask unavailable (typical in backtest).
    """

    def __init__(self):
        self.base_slippage_pct = 0.0010      # 0.10%
        self.volume_impact_factor = 0.15
        self.max_slippage_pct = 0.0200       # 2.00% cap

    def GetSlippageApproximation(self, asset, order):
        try:
            price = asset.Price
            if price <= 0:
                return 0

            slippage_pct = self.base_slippage_pct

            # Safely access bid/ask
            bid = getattr(asset, 'BidPrice', 0)
            ask = getattr(asset, 'AskPrice', 0)

            # Spread component
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    spread_cost = (ask - bid) / (2 * mid)
                    slippage_pct += spread_cost
            else:
                # Synthetic spread floor (backtest realism)
                if price < 0.01:
                    slippage_pct += 0.0100
                elif price < 0.10:
                    slippage_pct += 0.0050
                elif price < 1.0:
                    slippage_pct += 0.0025
                elif price < 10.0:
                    slippage_pct += 0.0015
                elif price < 100.0:
                    slippage_pct += 0.0008
                else:
                    slippage_pct += 0.0005

            # Volume impact
            volume = getattr(asset, 'Volume', 0)
            if volume > 0:
                order_value = abs(order.Quantity) * price
                volume_value = volume * price
                if volume_value > 0:
                    participation_rate = order_value / volume_value
                    volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                    slippage_pct += volume_impact

            # Price tier multipliers
            if price < 0.01:
                slippage_pct *= 4.0
            elif price < 0.10:
                slippage_pct *= 2.5
            elif price < 1.0:
                slippage_pct *= 1.8
            elif price < 10.0:
                slippage_pct *= 1.2

            # Cap slippage
            slippage_pct = min(slippage_pct, self.max_slippage_pct)

            return price * slippage_pct

        except Exception:
            return price * self.base_slippage_pct if price > 0 else 0
