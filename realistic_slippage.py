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
    - baseline slippage calibrated for believable default backtests.
    - optional stress multipliers for spread/slippage/impact for harsher scenarios.
    - synthetic spread floor applied when bid/ask unavailable (typical in backtest).

    Stress mode:
    - stress_mult > 1.0 scales base_slippage_pct for pessimistic robustness testing.
      Set via algo.stress_slippage_mult (default 1.0 = no stress).
    - spread_floor_mult / impact_mult default to 1.0 for baseline realism.
    - participation_cap defaults to 0.15 in baseline mode (raise for stress tests).
    """

    BASE_VOLUME_IMPACT_FACTOR = 0.35

    def __init__(self, stress_mult=1.0, spread_floor_mult=1.0, impact_mult=1.0, participation_cap=0.15):
        base = 0.0030 * max(0.1, float(stress_mult))
        self.base_slippage_pct = base
        self.spread_floor_mult = max(0.5, float(spread_floor_mult))
        self.volume_impact_factor = self.BASE_VOLUME_IMPACT_FACTOR * max(0.25, float(impact_mult))
        self.max_slippage_pct = 0.0600
        # Cap participation so single-bar outliers don't dominate stress runs.
        self.participation_cap = max(0.01, float(participation_cap))

    def GetSlippageApproximation(self, asset, order):
        try:
            price = asset.Price
            if price <= 0:
                return 0

            slippage_pct = self.base_slippage_pct
            # Initialize before branch-specific assignment from quoted or synthetic spread.
            spread_component = 0.0

            # Safely access bid/ask
            bid = getattr(asset, 'BidPrice', 0)
            ask = getattr(asset, 'AskPrice', 0)

            # Spread component
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    spread_cost = (ask - bid) / (2 * mid)
                    spread_component = spread_cost
            else:
                # Synthetic spread floor (backtest realism when bid/ask are unavailable).
                if price < 0.01:
                    spread_component = 0.0120 * self.spread_floor_mult
                elif price < 0.10:
                    spread_component = 0.0060 * self.spread_floor_mult
                elif price < 1.0:
                    spread_component = 0.0030 * self.spread_floor_mult
                elif price < 10.0:
                    spread_component = 0.0018 * self.spread_floor_mult
                elif price < 100.0:
                    spread_component = 0.0012 * self.spread_floor_mult
                else:
                    spread_component = 0.0008 * self.spread_floor_mult

            # Volume impact
            volume = getattr(asset, 'Volume', 0)
            if volume > 0:
                order_value = abs(order.Quantity) * price
                volume_value = volume * price
                if volume_value > 0:
                    participation_rate = min(order_value / volume_value, self.participation_cap)
                    volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                    slippage_pct += volume_impact

            # Price tier multipliers
            if price < 0.01:
                slippage_pct *= 3.2
            elif price < 0.10:
                slippage_pct *= 2.4
            elif price < 1.0:
                slippage_pct *= 1.8
            elif price < 10.0:
                slippage_pct *= 1.35

            # Add spread separately so spread isn't over-amplified by tier multipliers.
            slippage_pct += spread_component

            # Cap slippage
            slippage_pct = min(slippage_pct, self.max_slippage_pct)

            return price * slippage_pct

        except Exception:
            return price * self.base_slippage_pct if price > 0 else 0
