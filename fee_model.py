# region imports
from AlgorithmImports import *
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
from collections import deque
from datetime import timedelta
# endregion


class KrakenTieredFeeModel(FeeModel):
    """Volume-tiered Kraken Pro (Canada) fee model; 80% maker / 20% taker blend."""

    LIMIT_TAKER_RATIO = 0.20
    FEE_TIERS = [
        (500_000, 0.0008, 0.0018),
        (250_000, 0.0010, 0.0020),
        (100_000, 0.0012, 0.0022),
        (50_000,  0.0014, 0.0024),
        (25_000,  0.0020, 0.0035),
        (10_000,  0.0022, 0.0038),
        (2_500,   0.0030, 0.0060),
        (0,       0.0040, 0.0080),
    ]

    def __init__(self, fee_mult=1.0):
        # Rolling 30-day notional traded, used for Kraken tier selection.
        self._rolling_30d_volume = 0.0
        # Queue of (timestamp, trade_value) events for sliding-window volume updates.
        self._volume_events = deque()
        self._fee_mult = max(0.1, float(fee_mult))

    def _update_rolling_volume(self, when, trade_value):
        self._volume_events.append((when, trade_value))
        self._rolling_30d_volume += trade_value
        cutoff = when - timedelta(days=30)
        while self._volume_events and self._volume_events[0][0] < cutoff:
            _, old_value = self._volume_events.popleft()
            self._rolling_30d_volume -= old_value
        self._rolling_30d_volume = max(0.0, self._rolling_30d_volume)

    def GetOrderFee(self, parameters):
        order = parameters.Order
        price = parameters.Security.Price
        trade_value = order.AbsoluteQuantity * price
        if trade_value <= 0:
            return OrderFee(CashAmount(0, "USD"))

        self._update_rolling_volume(order.Time, trade_value)
        monthly_volume = max(0.0, self._rolling_30d_volume)
        maker_rate, taker_rate = self.FEE_TIERS[-1][1], self.FEE_TIERS[-1][2]
        for min_vol, maker, taker in self.FEE_TIERS:
            if monthly_volume >= min_vol:
                maker_rate, taker_rate = maker, taker
                break
        if order.Type == OrderType.Limit:
            fee_pct = (1 - self.LIMIT_TAKER_RATIO) * maker_rate + self.LIMIT_TAKER_RATIO * taker_rate
        else:
            fee_pct = taker_rate
        return OrderFee(CashAmount(trade_value * fee_pct * self._fee_mult, "USD"))
