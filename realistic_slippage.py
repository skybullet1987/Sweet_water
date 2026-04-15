# region imports
from AlgorithmImports import *
# endregion


class RealisticCryptoSlippage:
    """
    Realistic crypto slippage model for backtesting.
    Volume-aware, calibrated against empirical Kraken fill data.

    LEAN Python uses duck typing for slippage models — implementing
    GetSlippageApproximation(asset, order) is sufficient. Do NOT inherit
    from ISlippageModel; pythonnet raises "interface takes exactly one
    argument" when constructing the instance.

    Backtest realism additions:
    - base_slippage_pct 0.50% calibrated for small-cap Kraken altcoin execution.
    - volume_impact_factor raised for harsher market-impact assumptions.
    - max_slippage_pct raised to 8.0% for realistic altcoin cost tails.
    - Synthetic spread floors doubled vs prior values to match real Kraken spreads.
    - Synthetic spread floor applied when bid/ask unavailable (typical in backtest).

    Stress mode:
    - stress_mult > 1.0 scales base_slippage_pct for pessimistic robustness testing.
      Set via algo.stress_slippage_mult (default 1.0 = no stress).
    """

    def __init__(self, stress_mult=1.0, spread_floor_mult=1.25, impact_mult=1.5):
        base = 0.0050 * max(0.1, float(stress_mult))
        self.base_slippage_pct = base
        self.spread_floor_mult = max(0.5, float(spread_floor_mult))
        self.volume_impact_factor = 0.60 * max(0.25, float(impact_mult))
        self.max_slippage_pct = 0.0800

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
                # Synthetic spread floor (backtest realism — doubled vs prior values to
                # match real Kraken spreads which are roughly 2× the old floors)
                if price < 0.01:
                    slippage_pct += 0.0200 * self.spread_floor_mult
                elif price < 0.10:
                    slippage_pct += 0.0100 * self.spread_floor_mult
                elif price < 1.0:
                    slippage_pct += 0.0050 * self.spread_floor_mult
                elif price < 10.0:
                    slippage_pct += 0.0030 * self.spread_floor_mult
                elif price < 100.0:
                    slippage_pct += 0.0018 * self.spread_floor_mult
                else:
                    slippage_pct += 0.0012 * self.spread_floor_mult

            # Volume impact
            volume = getattr(asset, 'Volume', 0)
            if volume > 0:
                order_value = abs(order.Quantity) * price
                volume_value = volume * price
                if volume_value > 0:
                    participation_rate = min(order_value / volume_value, 0.20)
                    volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                    slippage_pct += volume_impact

            # Price tier multipliers
            if price < 0.01:
                slippage_pct *= 5.0
            elif price < 0.10:
                slippage_pct *= 3.5
            elif price < 1.0:
                slippage_pct *= 2.5
            elif price < 10.0:
                slippage_pct *= 1.8

            # Cap slippage
            slippage_pct = min(slippage_pct, self.max_slippage_pct)

            return price * slippage_pct

        except Exception:
            return price * self.base_slippage_pct if price > 0 else 0
