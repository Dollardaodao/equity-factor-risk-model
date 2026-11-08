import unittest

import numpy as np
import pandas as pd

from factor_project.descriptors import _dividend_yield, _split_adjusted_shares, session_returns


class DescriptorTests(unittest.TestCase):
    def test_missing_session_is_not_a_one_day_return(self):
        sessions = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame({"date": sessions[[0, 2, 3]], "symbol": "X", "adj_close": [100, 110, 121]})
        result = session_returns(prices, sessions)["X"]
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertAlmostEqual(result.iloc[3], .1)

    def test_share_basis_uses_filing_not_old_financial_end(self):
        dates = pd.date_range("2024-06-01", "2024-06-30")
        prices = pd.DataFrame({"split": 1.0}, index=dates)
        prices.loc["2024-06-10", "split"] = 10
        self.assertEqual(_split_adjusted_shares(100, "2024-06-05", dates[-1], prices), 1000)
        self.assertEqual(_split_adjusted_shares(1000, "2024-06-15", dates[-1], prices), 1000)
        with self.assertRaises(ValueError):
            _split_adjusted_shares(1000, "2024-07-01", dates[-1], prices)

    def test_dividend_cash_is_converted_to_current_share_basis(self):
        dates = pd.date_range("2023-06-01", "2024-06-30")
        prices = pd.DataFrame({"split": 1.0, "dividend": 0.0}, index=dates)
        prices.loc["2024-06-05", "dividend"] = 1
        prices.loc["2024-06-10", "split"] = 10
        self.assertAlmostEqual(_dividend_yield(prices, dates[-1], 10), .01)
        self.assertTrue(np.isnan(_dividend_yield(prices.loc["2024-01-01":], dates[-1], 10)))


if __name__ == "__main__":
    unittest.main()
