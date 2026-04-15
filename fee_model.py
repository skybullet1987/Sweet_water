# region imports
from AlgorithmImports import *
from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
# endregion


class KrakenTieredFeeModel(FeeModel):
    """Volume-tiered Kraken Pro (Canada) fee model; 75% maker / 25% taker blend."""

    LIMIT_TAKER_RATIO = 0.25
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

    def __init__(self):
        self._cumulative_volume = 0.0
        self._start_time = None

    def GetOrderFee(self, parameters):
        order = parameters.Order
        price = parameters.Security.Price
        trade_value = order.AbsoluteQuantity * price
        self._cumulative_volume += trade_value
        if self._start_time is None:
            self._start_time = order.Time
        elapsed_days = max((order.Time - self._start_time).days, 1)
        monthly_volume = self._cumulative_volume * 30.0 / elapsed_days
        maker_rate, taker_rate = self.FEE_TIERS[-1][1], self.FEE_TIERS[-1][2]
        for min_vol, maker, taker in self.FEE_TIERS:
            if monthly_volume >= min_vol:
                maker_rate, taker_rate = maker, taker
                break
        if order.Type == OrderType.Limit:
            fee_pct = (1 - self.LIMIT_TAKER_RATIO) * maker_rate + self.LIMIT_TAKER_RATIO * taker_rate
        else:
            fee_pct = taker_rate
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))
