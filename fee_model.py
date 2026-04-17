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

    def __init__(self, comparison_mode=False, fixed_maker_rate=None, fixed_taker_rate=None):
        self._comparison_mode = bool(comparison_mode)
        default_maker = self.FEE_TIERS[-1][1]
        default_taker = self.FEE_TIERS[-1][2]
        self._fixed_maker_rate = default_maker if fixed_maker_rate is None else float(fixed_maker_rate)
        self._fixed_taker_rate = default_taker if fixed_taker_rate is None else float(fixed_taker_rate)
        self._cumulative_volume = 0.0
        self._start_time = None

    def _current_rates(self):
        """Return current (maker_rate, taker_rate) from mode/tier state."""
        if self._comparison_mode:
            return self._fixed_maker_rate, self._fixed_taker_rate
        maker_rate, taker_rate = self.FEE_TIERS[-1][1], self.FEE_TIERS[-1][2]
        for min_vol, maker, taker in self.FEE_TIERS:
            if self._cumulative_volume >= min_vol:
                maker_rate, taker_rate = maker, taker
                break
        return maker_rate, taker_rate

    def estimate_round_trip_cost(self, symbol, notional, is_limit=True):
        """Estimate round-trip fee in USD for a hypothetical order notional."""
        notional = abs(float(notional or 0.0))
        if notional <= 0:
            return 0.0
        maker_rate, taker_rate = self._current_rates()
        if is_limit:
            side_fee_pct = (1 - self.LIMIT_TAKER_RATIO) * maker_rate + self.LIMIT_TAKER_RATIO * taker_rate
        else:
            side_fee_pct = taker_rate
        return notional * side_fee_pct * 2.0

    def GetOrderFee(self, parameters):
        order = parameters.Order
        price = parameters.Security.Price
        trade_value = order.AbsoluteQuantity * price
        if not self._comparison_mode:
            self._cumulative_volume += trade_value
            if self._start_time is None:
                self._start_time = order.Time
        maker_rate, taker_rate = self._current_rates()
        if order.Type == OrderType.Limit:
            fee_pct = (1 - self.LIMIT_TAKER_RATIO) * maker_rate + self.LIMIT_TAKER_RATIO * taker_rate
        else:
            fee_pct = taker_rate
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))
