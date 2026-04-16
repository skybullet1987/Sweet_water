import unittest
from datetime import datetime

from nextgen.config.loader import load_strategy_config
from nextgen.core.types import Bar
from nextgen.features.basic import BasicFeatureEngine


class ConfigAndTypesTests(unittest.TestCase):
    def test_load_strategy_config_typed(self) -> None:
        config = load_strategy_config(
            {
                "name": "nextgen-scaffold",
                "symbols": ["BTCUSD", "ETHUSD"],
                "target_volatility": 0.2,
                "max_gross_exposure": 1.0,
                "max_net_exposure": 0.5,
                "max_position_weight": 0.2,
                "drawdown_throttle": 0.1,
                "kill_switch_drawdown": 0.2,
            }
        )
        self.assertEqual(config.name, "nextgen-scaffold")
        self.assertEqual(config.symbols, ("BTCUSD", "ETHUSD"))

    def test_feature_engine_outputs_typed_feature(self) -> None:
        engine = BasicFeatureEngine(lookback=3)
        output = engine.update(
            Bar(
                symbol="BTCUSD",
                timestamp=datetime(2026, 1, 1),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=10,
            )
        )
        self.assertEqual(output.symbol, "BTCUSD")
        self.assertIn("momentum", output.values)


if __name__ == "__main__":
    unittest.main()
