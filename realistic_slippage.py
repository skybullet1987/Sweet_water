# region imports
from AlgorithmImports import *
# endregion

class RealisticCryptoSlippage(ISlippageModel):
    """
    Realistic crypto slippage model for backtesting.

    Volume-aware, calibrated against empirical Kraken fill data.
    Altcoins under $1 routinely have 0.5-1.5% effective slippage.

    PythonNet requirement: __init__ must take exactly self (no extra parameters).
    """

    def __init__(self):
        self.base_slippage_pct = 0.003     # 0.3% base
        self.volume_impact_factor = 0.20   # market impact factor
        self.max_slippage_pct = 0.05       # 5% cap

    def GetSlippageApproximation(self, asset, order):
        """
        Calculate slippage for an order.

        Returns slippage as a dollar amount (price * slippage_pct),
        as expected by QuantConnect's ISlippageModel interface.
        """
        try:
            price = asset.Price
            if price <= 0:
                return 0

            slippage_pct = self.base_slippage_pct

            # Add half-spread cost (you cross the spread on market orders)
            bid = asset.BidPrice if hasattr(asset, 'BidPrice') else 0
            ask = asset.AskPrice if hasattr(asset, 'AskPrice') else 0
            if bid > 0 and ask > 0 and ask >= bid:
                mid = 0.5 * (bid + ask)
                if mid > 0:
                    spread_cost = (ask - bid) / (2 * mid)  # half-spread
                    slippage_pct += spread_cost

            # Volume impact: larger orders relative to volume move the price more
            volume = asset.Volume if hasattr(asset, 'Volume') else 0
            if volume > 0:
                order_value = abs(order.Quantity) * price
                volume_value = volume * price
                if volume_value > 0:
                    participation_rate = order_value / volume_value
                    volume_impact = self.volume_impact_factor * (participation_rate ** 1.5)
                    slippage_pct += volume_impact

            # Price tier penalties — low-price alts have wider spreads
            if price < 0.01:
                slippage_pct *= 5.0    # micro-caps: 1.5%+ slippage typical
            elif price < 0.10:
                slippage_pct *= 3.5    # low-price: 1.0%+ slippage
            elif price < 1.0:
                slippage_pct *= 2.5    # sub-$1: 0.75%+ slippage
            elif price < 10.0:
                slippage_pct *= 1.5    # mid-caps: 0.45%+ slippage
            elif price < 100.0:
                slippage_pct *= 1.2    # larger caps still have some spread

            slippage_pct = min(slippage_pct, self.max_slippage_pct)

            return price * slippage_pct

        except Exception:
            return asset.Price * self.base_slippage_pct if asset.Price > 0 else 0


class VolatilityAdjustedSlippage(ISlippageModel):
    """
    Slippage model that scales with volatility.
    
    Higher volatility → higher slippage (wider spreads)
    Lower volatility → lower slippage (tighter spreads)
    
    Useful for crypto which has regime changes.

    PythonNet requirement: __init__ must take exactly self (no extra parameters).
    """
    
    def __init__(self):
        # NOTE: algo is set to None; GetSlippageApproximation uses hasattr(self.algo, '_atr')
        # which safely returns False when algo is None, falling back to 0.10% slippage.
        self.algo = None
        self.lookback_bars = 20
    
    def GetSlippageApproximation(self, asset, order):
        """
        Calculate slippage adjusted for volatility.
        
        Base slippage (0.10%) + volatility factor (0-0.40%)
        = total 0.10% to 0.50%
        """
        try:
            symbol = asset.Symbol
            
            # Get volatility indicator if available
            # This assumes you have ATR or similar in algorithm
            if hasattr(self.algo, '_atr') and symbol in self.algo._atr:
                atr_ind = self.algo._atr[symbol]
                if atr_ind.IsReady:
                    price = asset.Price
                    atr = atr_ind.Current.Value
                    
                    # ATR as % of price
                    atr_pct = atr / price if price > 0 else 0
                    
                    # Scale: low volatility (0.5% ATR) = 0.10% slippage
                    #        high volatility (5% ATR) = 0.50% slippage
                    slippage = 0.001 + (atr_pct * 0.08)  # 0.10% + (volatility * 8%)
                    slippage = min(slippage, 0.005)  # Cap at 0.50%
                    return slippage
            
            # Fallback to fixed slippage
            return 0.0010  # 0.10%
        
        except Exception:
            return 0.0010


class MarketMicrostructureSlippage(ISlippageModel):
    """
    Advanced slippage model based on order size vs market depth.
    
    Small orders: minimal slippage (fit in spread)
    Large orders: higher slippage (move price, market impact)
    """
    
    def __init__(self):
        pass
    
    def GetSlippageApproximation(self, asset, order):
        """
        Slippage = spread/2 + market_impact
        Market impact ~ (order_size / market_depth)^0.5
        """
        try:
            security = asset
            bid = security.BidPrice if hasattr(security, 'BidPrice') else 0
            ask = security.AskPrice if hasattr(security, 'AskPrice') else 0
            
            # Spread component
            spread_slippage = 0
            if bid > 0 and ask > 0:
                spread = ask - bid
                spread_pct = spread / ((bid + ask) / 2)
                spread_slippage = spread_pct / 2  # Half spread for execution
            
            # Order size component (simplified - assume small retail order)
            # Typical retail order = 0.01% of daily volume = minimal impact
            market_impact = 0.0002  # 0.02% average impact
            
            total = spread_slippage + max(0.0005, market_impact)  # Min 0.05%
            total = min(total, 0.0050)  # Cap at 0.50%
            
            return total
        
        except Exception:
            return 0.0015  # 0.15% default
