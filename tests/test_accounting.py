import unittest

from nextgen.accounting.ledger import AccountingConfig, UnifiedAccountingLedger


class AccountingTests(unittest.TestCase):
    def test_fee_defaults_and_realized_pnl(self) -> None:
        ledger = UnifiedAccountingLedger(AccountingConfig(default_fee_rate=0.001))
        fill = ledger.ingest_fill("BTCUSD", quantity=1.0, price=100.0)
        self.assertAlmostEqual(fill.fee, 0.1, places=6)

        pnl = ledger.realized_pnl_on_close("BTCUSD", close_quantity=1.0, close_price=110.0, fee=0.11)
        self.assertAlmostEqual(pnl.realized_pnl, 9.89, places=6)


if __name__ == "__main__":
    unittest.main()
