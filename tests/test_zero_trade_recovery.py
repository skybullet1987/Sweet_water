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


if __name__ == "__main__":
    unittest.main()
