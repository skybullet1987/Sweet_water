import unittest
from pathlib import Path


class ZeroTradeRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _read(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text(encoding="utf-8")

    def test_regime_router_delegates_chop_engine(self) -> None:
        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn('if active_regime == "chop":', content)
            self.assertIn("self._run_chop_rebalance()", content)

    def test_pipeline_diagnostics_present(self) -> None:
        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn("PIPELINE trend:", content)
        for path in ("entry_exec.py", "qc_runtime/entry_exec.py"):
            content = self._read(path)
            self.assertIn("CHOP PIPELINE:", content)

    def test_zero_trade_bottlenecks_relaxed(self) -> None:
        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn("self.high_conviction_threshold = 0.66", content)
            self.assertIn("self.min_signal_count = int(self._get_param(\"min_signal_count\", 2))", content)
            self.assertIn("self.max_spread_pct         = 0.0065", content)
            self.assertIn("self.spread_widen_mult      = 2.4", content)
            self.assertIn("self.chop_score_buffer = 0.02", content)
        for path in ("regime_router.py", "qc_runtime/regime_router.py"):
            content = self._read(path)
            self.assertIn("REGIME_MIN_BARS  = 3", content)
            self.assertIn("TRANSITION_HOLD  = 1", content)


if __name__ == "__main__":
    unittest.main()
