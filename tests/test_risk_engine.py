import unittest

from nextgen.core.models import PortfolioTarget
from nextgen.risk.engine import PortfolioState, PositionState, RiskConfig, UnifiedRiskEngine


class RiskEngineTests(unittest.TestCase):
    def test_kill_switch_rejects_trade(self) -> None:
        engine = UnifiedRiskEngine(RiskConfig(kill_switch_drawdown=0.2))
        state = PortfolioState(estimated_portfolio_volatility=0.1, current_drawdown=0.25, gross_exposure=0.1, net_exposure=0.1)
        decision = engine.evaluate_target(PortfolioTarget("BTCUSD", 0.1, 0.5), state)
        self.assertFalse(decision.approved)
        self.assertIn("kill_switch_drawdown", decision.reason_codes)

    def test_target_volatility_scales_position(self) -> None:
        engine = UnifiedRiskEngine(RiskConfig(target_portfolio_volatility=0.2, max_position_weight=1.0, max_net_exposure=2.0, max_gross_exposure=2.0))
        state = PortfolioState(estimated_portfolio_volatility=0.4, current_drawdown=0.0, gross_exposure=0.2, net_exposure=0.0)
        decision = engine.evaluate_target(PortfolioTarget("ETHUSD", 0.2, 0.9), state)
        self.assertTrue(decision.approved)
        self.assertAlmostEqual(decision.adjusted_target_weight, 0.1, places=6)
        self.assertIn("target_volatility_scaling", decision.reason_codes)

    def test_exposure_cap_can_reject(self) -> None:
        engine = UnifiedRiskEngine(RiskConfig(max_gross_exposure=0.25, max_net_exposure=0.5, max_position_weight=1.0))
        state = PortfolioState(
            estimated_portfolio_volatility=0.1,
            current_drawdown=0.0,
            gross_exposure=0.24,
            net_exposure=0.0,
            positions={"BTCUSD": PositionState(weight=0.0, annualized_volatility=0.5)},
        )
        decision = engine.evaluate_target(PortfolioTarget("ETHUSD", 0.1, 0.9), state)
        self.assertFalse(decision.approved)
        self.assertIn("gross_exposure_cap", decision.reason_codes)


if __name__ == "__main__":
    unittest.main()
