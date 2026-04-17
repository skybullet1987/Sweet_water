import unittest
from pathlib import Path


class TradingEdgeHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _read(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text(encoding="utf-8")

    def test_main_and_runtime_use_stricter_trade_limits(self) -> None:
        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn("self.max_positions      = 3", content)
            self.assertIn("self.max_daily_trades       = 8", content)
            self.assertIn("self.max_symbol_trades_per_day = 999", content)
            self.assertIn("self.entry_cooldown_minutes = 60", content)
            self.assertIn("self.reentry_cooldown_minutes = 360", content)
            self.assertIn("self.high_conviction_threshold = 0.66", content)
            self.assertIn("self.min_signal_count = int(self._get_param(\"min_signal_count\", 2))", content)
            self.assertIn("self.pyramid_enabled = bool(self._get_param(\"pyramid_enabled\", 0.0))", content)

    def test_entry_execution_contains_anti_cluster_logic(self) -> None:
        for path in ("entry_exec.py", "qc_runtime/entry_exec.py"):
            content = self._read(path)
            self.assertIn("def _is_symbol_trade_clustered(algo, sym):", content)
            self.assertIn("algo._symbol_entry_cooldowns[sym.Value] = algo.Time + timedelta(minutes=cooldown_minutes)", content)
            self.assertIn("history = algo._recent_entry_times.setdefault(sym, deque(maxlen=10))", content)

    def test_blacklist_excludes_high_noise_symbols(self) -> None:
        for path in ("execution.py", "qc_runtime/execution.py"):
            content = self._read(path)
            for ticker in ("TRUMPUSD", "FARTCOINUSD", "MOODENGUSD", "TITCOINUSD", "FWOGUSD", "XCNUSD"):
                self.assertIn(f"\"{ticker}\"", content)


if __name__ == "__main__":
    unittest.main()
