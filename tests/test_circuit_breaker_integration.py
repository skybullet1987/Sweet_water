import unittest
from pathlib import Path


class CircuitBreakerIntegrationTests(unittest.TestCase):
    def test_main_wires_drawdown_entry_breaker(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "main.py").read_text(encoding="utf-8")
        self.assertIn("from circuit_breaker import DrawdownCircuitBreaker", content)
        self.assertIn("self._drawdown_entry_breaker = DrawdownCircuitBreaker(max_drawdown_pct=-0.10)", content)
        self.assertIn("self._drawdown_entry_breaker.update(self)", content)
        self.assertIn("self._drawdown_entry_breaker.is_triggered()", content)


if __name__ == "__main__":
    unittest.main()
