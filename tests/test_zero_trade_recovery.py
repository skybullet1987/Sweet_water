import unittest
from pathlib import Path


class ZeroTradeRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]

    def _read(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text(encoding="utf-8")

    def test_main_routes_risk_off_to_exit_only(self) -> None:
        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn('if regime == "risk_off":', content)
            self.assertIn("self._manage_existing_positions", content)

    def test_main_uses_single_rule_scorer_engine(self) -> None:
        for path in ("main.py", "qc_runtime/main.py"):
            content = self._read(path)
            self.assertIn("score_symbol", content)
            self.assertNotIn("MicroScalpEngine", content)

    def test_removed_legacy_modules_are_absent(self) -> None:
        for path in (
            "chop_engine.py",
            "trade_quality.py",
            "regime_router.py",
            "scoring.py",
            "qc_runtime/chop_engine.py",
            "qc_runtime/trade_quality.py",
            "qc_runtime/regime_router.py",
            "qc_runtime/scoring.py",
        ):
            self.assertFalse((self.repo_root / path).exists(), msg=f"Unexpected legacy module: {path}")


if __name__ == "__main__":
    unittest.main()
