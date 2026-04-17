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
    - base_slippage_pct 0.40% calibrated to real Kraken altcoin fills at $500-800 notional.
    - volume_impact_factor 0.25 reflects real market impact at 1% participation.
    - max_slippage_pct raised to 5.0% for realistic altcoin costs.
    - Synthetic spread floors doubled vs prior values to match real Kraken spreads.
    - Synthetic spread floor applied when bid/ask unavailable (typical in backtest).

    Stress mode:
    - stress_mult > 1.0 scales base_slippage_pct for pessimistic robustness testing.
      Set via algo.stress_slippage_mult (default 1.0 = no stress).
    """

    def __init__(self, stress_mult=1.0):
        base = 0.0040 * max(0.1, float(stress_mult))   # 0.4% base (was 0.2%)
        self.base_slippage_pct = base
        self.volume_impact_factor = 0.25               # was 0.40 — note: PR review intended 0.25 for new model
        self.max_slippage_pct = 0.0500                 # 5% cap (was 2%)

    def _estimate_slippage_pct(self, price, notional, volume=0.0, bid=0.0, ask=0.0):
        """Estimate one-way slippage as a fraction of notional.

        Parameters are price/size plus optional quote and volume context, using
        the same spread + impact logic as GetSlippageApproximation.
        """
        if price <= 0:
            return 0.0
        slippage_pct = self.base_slippage_pct

        if bid > 0 and ask > 0 and ask >= bid:
            mid = 0.5 * (bid + ask)
            if mid > 0:
                slippage_pct += (ask - bid) / (2 * mid)
        else:
            if price < 0.01:
                slippage_pct += 0.0200
            elif price < 0.10:
                slippage_pct += 0.0100
            elif price < 1.0:
                slippage_pct += 0.0050
            elif price < 10.0:
                slippage_pct += 0.0030
            elif price < 100.0:
                slippage_pct += 0.0016
            else:
                slippage_pct += 0.0010

        if volume > 0:
            volume_value = volume * price
            if volume_value > 0:
                participation_rate = abs(notional) / volume_value
                slippage_pct += self.volume_impact_factor * (participation_rate ** 1.5)

        if price < 0.01:
            slippage_pct *= 5.0
        elif price < 0.10:
            slippage_pct *= 3.5
        elif price < 1.0:
            slippage_pct *= 2.5
        elif price < 10.0:
            slippage_pct *= 1.8

        return min(slippage_pct, self.max_slippage_pct)

    def estimate_slippage_bps(self, symbol, notional, price, volume=0.0, bid=0.0, ask=0.0):
        """Estimate one-way slippage in basis points for a hypothetical order."""
        return self._estimate_slippage_pct(price, notional, volume=volume, bid=bid, ask=ask) * 10_000.0

    def GetSlippageApproximation(self, asset, order):
        try:
            price = asset.Price
            if price <= 0:
                return 0

            order_value = abs(order.Quantity) * price
            slippage_pct = self._estimate_slippage_pct(
                price=price,
                notional=order_value,
                volume=getattr(asset, 'Volume', 0),
                bid=getattr(asset, 'BidPrice', 0),
                ask=getattr(asset, 'AskPrice', 0),
            )
            return price * slippage_pct

        except Exception:
            return price * self.base_slippage_pct if price > 0 else 0
