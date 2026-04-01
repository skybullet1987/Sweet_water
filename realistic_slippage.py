# region imports
from AlgorithmImports import *
# endregion


class RealisticCryptoSlippage(ISlippageModel):
    """
    Realistic crypto slippage model for backtesting.
    Volume-aware, calibrated against empirical Kraken fill data.

    NOTE: __init__ takes only self (no extra params) — required by PythonNet
    when inheriting from C# interfaces like ISlippageModel.
    """

    def __init__(self):
        self.base_slippage_pct = 0.0010      # 0.10%
        self.volume_impact_factor = 0.15
        self.max_slippage_pct = 0.0050       # 0.50% cap

    def GetSlippageApproximation(self, asset, order):
        try:
            price = asset.Price
            if price <= 0:
                return 0

            slippage_pct = self.base_slippage_pct

            bid = asset.BidPrice if hasattr(asset, 'BidPrice') else 0
            ask = asset.AskPrice if hasattr(asset, 'AskPrice') else 0
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    spread_cost = (ask - bid) / (2 * mid)
                    slippage_pct += spread_cost

            volume = asset.Volume if hasattr(asset, 'Volume') else 0
            if volume > 0:
                order_value = abs(order.Quantity) * price
                volume_value = volume * price
                if volume_value > 0:
                    participation_rate = order_value / volume_value
                    volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                    slippage_pct += volume_impact

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

