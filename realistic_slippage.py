# region imports
from AlgorithmImports import *
# endregion

class RealisticCryptoSlippage(ISlippageModel):
    """
    Realistic crypto slippage model for backtesting.
    
    Models real-world execution costs:
    - Spread cost (half the bid-ask spread)
    - Market impact for larger orders
    - Execution latency slippage
    
    Typical values:
    - Tight markets (BTC/ETH): 0.05-0.15% total
    - Normal markets: 0.15-0.30% total
    - Wide spreads: 0.50%+ total
    """
    
    def __init__(self, base_slippage_pct=0.0010, max_slippage_pct=0.0050):
        """
        Args:
            base_slippage_pct: float (default 0.10%)
                Minimum slippage even in perfect conditions
            max_slippage_pct: float (default 0.50%)
                Cap on maximum slippage per order
        """
        self.base_slippage_pct = base_slippage_pct  # 0.10%
        self.max_slippage_pct = max_slippage_pct    # 0.50%
    
    def GetSlippageApproximation(self, asset, order):
        """
        Calculate slippage for an order.
        
        Called by QuantConnect during backtest for each order.
        Returns slippage as decimal (e.g., 0.001 = 0.1%)
        """
        try:
            # Get current bid-ask spread if available
            security = asset
            bid = security.BidPrice if hasattr(security, 'BidPrice') else 0
            ask = security.AskPrice if hasattr(security, 'AskPrice') else 0
            
            # Calculate spread-based slippage
            spread_slippage = 0
            if bid > 0 and ask > 0:
                spread = ask - bid
                spread_pct = spread / ((bid + ask) / 2)
                # Slippage is roughly half the spread for limit orders
                # Full spread for market orders
                is_market_order = order.OrderType == OrderType.Market
                spread_slippage = (spread_pct / 2) if not is_market_order else spread_pct
            
            # Total slippage = base + spread-based
            total_slippage = self.base_slippage_pct + spread_slippage
            
            # Cap at maximum
            total_slippage = min(total_slippage, self.max_slippage_pct)
            
            # Ensure minimum base slippage always applies
            total_slippage = max(total_slippage, self.base_slippage_pct)
            
            return total_slippage
        
        except Exception:
            # Fallback to base slippage if calculation fails
            return self.base_slippage_pct


class VolatilityAdjustedSlippage(ISlippageModel):
    """
    Slippage model that scales with volatility.
    
    Higher volatility → higher slippage (wider spreads)
    Lower volatility → lower slippage (tighter spreads)
    
    Useful for crypto which has regime changes.
    """
    
    def __init__(self, algorithm, lookback_bars=20):
        """
        Args:
            algorithm: QCAlgorithm instance (for accessing indicator data)
            lookback_bars: int, bars for volatility calculation
        """
        self.algo = algorithm
        self.lookback_bars = lookback_bars
    
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
