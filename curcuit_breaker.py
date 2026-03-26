# region imports
# TODO: This file is misnamed — should be circuit_breaker.py (typo: 'curcuit').
# Rename to circuit_breaker.py and update all imports once import breakage can be coordinated.
from AlgorithmImports import *
from datetime import datetime, timedelta
# endregion

class DrawdownCircuitBreaker:
    """
    Maximum drawdown circuit breaker.
    
    Stops trading when portfolio loses more than a threshold (e.g., -10%).
    Prevents cascade losses from compounding during bad regimes.
    
    Usage:
        breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.10)
        breaker.update(algorithm)  # Call in OnData or scheduled event
        
        if breaker.is_triggered():
            algorithm.Liquidate()  # Close all positions
    """
    
    def __init__(self, max_drawdown_pct=-0.10, recovery_pct=0.05):
        """
        Args:
            max_drawdown_pct: float, negative (e.g., -0.10 for -10%)
                When portfolio equity drops this much from peak, breaker triggers
            recovery_pct: float, positive (e.g., 0.05 for +5%)
                After breach, breaker resets only if equity recovers this much
        """
        self.max_drawdown_pct = max_drawdown_pct  # -0.10
        self.recovery_pct = recovery_pct  # 0.05
        
        self.peak_equity = 0
        self.breaker_triggered = False
        self.trigger_time = None
        self.trigger_equity = None
        self.equity_at_trigger = 0
    
    def update(self, algorithm):
        """
        Update circuit breaker state. Call every bar or scheduled event.
        
        Args:
            algorithm: QCAlgorithm instance with Portfolio.TotalPortfolioValue
        """
        current_equity = algorithm.Portfolio.TotalPortfolioValue
        
        # Track peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            # Reset trigger if we recover beyond recovery threshold
            if self.breaker_triggered:
                recovery_target = self.equity_at_trigger * (1 + self.recovery_pct)
                if current_equity >= recovery_target:
                    algorithm.Debug(
                        f"BREAKER RESET: Equity recovered from ${self.equity_at_trigger:.2f} "
                        f"to ${current_equity:.2f} (target: ${recovery_target:.2f})"
                    )
                    self.breaker_triggered = False
                    self.trigger_time = None
        
        # Check drawdown from peak
        if self.peak_equity > 0:
            drawdown = (current_equity - self.peak_equity) / self.peak_equity
            
            if drawdown <= self.max_drawdown_pct and not self.breaker_triggered:
                # Breach threshold
                self.breaker_triggered = True
                self.trigger_time = algorithm.Time
                self.trigger_equity = current_equity
                self.equity_at_trigger = current_equity
                
                algorithm.Debug(
                    f"⚠️ CIRCUIT BREAKER TRIGGERED: Drawdown {drawdown:.2%} "
                    f"from peak ${self.peak_equity:.2f} to ${current_equity:.2f}. "
                    f"LIQUIDATING ALL POSITIONS."
                )
    
    def is_triggered(self):
        """Return True if circuit breaker is active."""
        return self.breaker_triggered
    
    def get_status(self):
        """Return formatted status string."""
        if self.breaker_triggered:
            return f"BREAKER ACTIVE (triggered {self.trigger_time})"
        return "Normal"


class RollingMaxDrawdown:
    """
    Track rolling maximum drawdown over N bars.
    Useful for monitoring strategy health without hard stops.
    """
    
    def __init__(self, lookback_bars=252):  # ~1 year of daily bars
        """
        Args:
            lookback_bars: int, window for max drawdown calculation
        """
        self.lookback_bars = lookback_bars
        self.equity_history = []
    
    def update(self, equity):
        """Add new equity value and calculate rolling max drawdown."""
        self.equity_history.append(equity)
        
        # Keep only last N bars
        if len(self.equity_history) > self.lookback_bars:
            self.equity_history.pop(0)
    
    def get_max_drawdown(self):
        """Return max drawdown as decimal (e.g., -0.05 for -5%)."""
        if len(self.equity_history) < 2:
            return 0
        
        peak = max(self.equity_history)
        trough = min(self.equity_history)
        
        if peak == 0:
            return 0
        
        return (trough - peak) / peak
    
    def get_max_drawdown_pct(self):
        """Return max drawdown as percentage string."""
        return f"{self.get_max_drawdown():.2%}"
