import unittest
from pathlib import Path


class QCRuntimeLayoutTests(unittest.TestCase):
    def test_qc_runtime_contains_expected_entrypoint_and_modules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        runtime_root = repo_root / "qc_runtime"

        self.assertTrue(runtime_root.is_dir())
        self.assertTrue((runtime_root / "main.py").is_file())
        self.assertTrue((runtime_root / "README.md").is_file())

        expected_files = {
            "execution.py",
            "reporting.py",
            "order_management.py",
            "realistic_slippage.py",
            "events.py",
            "scoring.py",
            "strategy_core.py",
            "trade_quality.py",
            "fee_model.py",
            "regime_router.py",
            "chop_engine.py",
            "entry_exec.py",
            "alt_data.py",
            "nextgen/core/types.py",
            "nextgen/risk/engine.py",
        }

        missing = [f for f in expected_files if not (runtime_root / f).is_file()]
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
