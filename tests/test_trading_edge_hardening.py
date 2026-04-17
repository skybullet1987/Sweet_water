import unittest
from pathlib import Path


class TradingEdgeHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _read(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text(encoding="utf-8")

    def test_main_and_runtime_use_strategy_config_defaults(self) -> None:
        config = self._read("config/strategy_config.py")
        self.assertIn("bar_resolution: str = \"Hour\"", config)
        self.assertIn("max_positions: int = 3", config)
        self.assertIn("score_threshold: float = 0.40", config)

        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn("self.cfg = DEFAULT_STRATEGY_CONFIG", content)
            self.assertIn("self.max_positions = self.cfg.max_positions", content)

    def test_entry_execution_single_engine_glue(self) -> None:
        for path in ("entry_exec.py", "qc_runtime/entry_exec.py"):
            content = self._read(path)
            self.assertIn("def execute_regime_entries", content)
            self.assertIn("signal_engine=\"rule_scorer\"", content)
            self.assertNotIn("run_chop_rebalance", content)

    def test_blacklist_excludes_high_noise_symbols(self) -> None:
        for path in ("execution.py", "qc_runtime/execution.py"):
            content = self._read(path)
            for ticker in ("TRUMPUSD", "FARTCOINUSD", "MOODENGUSD", "TITCOINUSD", "FWOGUSD", "XCNUSD"):
                self.assertIn(f"\"{ticker}\"", content)


if __name__ == "__main__":
    unittest.main()
