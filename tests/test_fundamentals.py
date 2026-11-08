"""PIT accounting tests cover filing availability, revisions, and annual flows."""
import json
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from factor_project.fundamentals import build_fundamentals


def fact(value, end, filed, *, start=None, form="10-K", accn="original"):
    result = {"val": value, "end": end, "filed": filed, "form": form, "accn": accn}
    if start:
        result["start"] = start
    return result


class FundamentalsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.cache = Path(self.temporary.name)
        self.universe = pd.DataFrame([{"symbol": "TEST", "cik": 1}])

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, usgaap, dei=None):
        payload = {"cik": 1, "facts": {"us-gaap": {
            tag: {"units": {"USD": records}} for tag, records in usgaap.items()}}}
        if dei:
            payload["facts"]["dei"] = {tag: {"units": {"shares": records}} for tag, records in dei.items()}
        (self.cache / "CIK0000000001.json").write_text(json.dumps(payload))

    def test_revision_available_only_after_filing_and_same_day_excluded(self):
        self.write({
            "Revenues": [
                fact(100, "2022-12-31", "2023-02-01", start="2022-01-01"),
                fact(75, "2022-12-31", "2024-02-01", start="2022-01-01", accn="restatement"),
            ],
            "Assets": [fact(200, "2022-12-31", "2023-02-01")],
        })
        rows = build_fundamentals(self.universe, ["2023-02-01", "2023-02-02", "2024-02-01", "2024-02-02"], self.cache)
        self.assertTrue(np.isnan(rows.iloc[0].revenue))
        self.assertEqual(list(rows.revenue.iloc[1:]), [100, 100, 75])
        self.assertIn("restatement", rows.iloc[-1].revenue_provenance)

    def test_annual_duration_and_form_filter_and_common_period_end(self):
        self.write({
            "Revenues": [
                fact(100, "2022-12-31", "2023-02-01", start="2022-01-01"),
                fact(31, "2023-03-31", "2023-05-01", start="2023-01-01", form="10-Q"),
                fact(999, "2023-03-31", "2023-05-01", start="2022-04-01", form="10-Q"),
                fact(888, "2022-12-31", "2023-03-01", start="2022-10-01"),
            ],
            "Assets": [fact(200, "2022-12-31", "2023-02-01"), fact(300, "2023-03-31", "2023-05-01", form="10-Q")],
        })
        row = build_fundamentals(self.universe, ["2023-06-01"], self.cache).iloc[0]
        self.assertEqual(row.revenue, 100)
        self.assertEqual(row.assets, 200)
        self.assertEqual(row.financial_period_end, pd.Timestamp("2022-12-31"))

    def test_actual_shares_staleness_and_no_weighted_average_fallback(self):
        self.write({"Revenues": [fact(100, "2022-12-31", "2023-02-01", start="2022-01-01")]},
                   {"EntityCommonStockSharesOutstanding": [fact(10, "2023-01-15", "2023-02-01")]})
        rows = build_fundamentals(self.universe, ["2023-02-01", "2023-02-02", "2023-08-01", "2024-08-01"], self.cache)
        self.assertTrue(np.isnan(rows.iloc[0].shares))
        self.assertEqual(rows.iloc[1].shares, 10)
        self.assertTrue(np.isnan(rows.iloc[2].shares))
        self.assertTrue(rows.iloc[3].financial_stale)
        self.assertTrue(np.isnan(rows.iloc[3].revenue))

    def test_negative_earnings_growth_uses_assets_and_missing_debt_not_zero(self):
        self.write({
            "Revenues": [fact(100, "2019-12-31", "2020-02-01", start="2019-01-01"),
                         fact(133.1, "2022-12-31", "2023-02-01", start="2022-01-01")],
            "NetIncomeLoss": [fact(-10, "2019-12-31", "2020-02-01", start="2019-01-01"),
                              fact(20, "2022-12-31", "2023-02-01", start="2022-01-01")],
            "Assets": [fact(100, "2019-12-31", "2020-02-01"), fact(200, "2022-12-31", "2023-02-01")],
        })
        row = build_fundamentals(self.universe, ["2023-03-01"], self.cache).iloc[0]
        self.assertAlmostEqual(row.revenue_3y_growth, np.log(1.1))
        self.assertAlmostEqual(row.earnings_3y_change, 0.3)
        self.assertTrue(np.isnan(row.debt))

    def test_identity_mismatch_excluded_but_missing_ticker_is_flagged(self):
        self.write({"Revenues": [fact(100, "2022-12-31", "2023-02-01", start="2022-01-01")]})
        target = self.cache / "CIK0000000001_submissions.json"
        target.write_text(json.dumps({"cik": 1, "name": "Different Co", "tickers": ["WRONG"]}))
        mismatch = build_fundamentals(self.universe, ["2023-03-01"], self.cache).iloc[0]
        self.assertFalse(mismatch.identity_match)
        self.assertTrue(np.isnan(mismatch.revenue))
        target.write_text(json.dumps({"cik": 2, "name": "Wrong identifier", "tickers": ["TEST"]}))
        mismatched_cik = build_fundamentals(self.universe, ["2023-03-01"], self.cache).iloc[0]
        self.assertFalse(mismatched_cik.identity_cik_verified)
        self.assertFalse(mismatched_cik.identity_verified)
        self.assertTrue(np.isnan(mismatched_cik.revenue))
        target.write_text(json.dumps({"cik": 1, "name": "Test Co", "tickers": []}))
        unknown = build_fundamentals(self.universe, ["2023-03-01"], self.cache).iloc[0]
        self.assertFalse(unknown.identity_verified)
        self.assertIsNone(unknown.identity_match)
        self.assertEqual(unknown.revenue, 100)

    def test_growth_revision_updates_only_after_new_filing_and_new_year_does_not_reuse_cache(self):
        self.write({
            "Revenues": [
                fact(100, "2019-12-31", "2020-02-01", start="2019-01-01"),
                fact(125, "2019-12-31", "2023-06-15", start="2019-01-01", accn="amended-base"),
                fact(200, "2022-12-31", "2023-02-01", start="2022-01-01"),
                fact(300, "2023-12-31", "2024-02-01", start="2023-01-01"),
            ],
            "NetIncomeLoss": [
                fact(10, "2019-12-31", "2020-02-01", start="2019-01-01"),
                fact(5, "2019-12-31", "2023-06-15", start="2019-01-01", accn="amended-base"),
                fact(20, "2022-12-31", "2023-02-01", start="2022-01-01"),
            ],
            "Assets": [fact(100, "2019-12-31", "2020-02-01"), fact(250, "2022-12-31", "2023-02-01")],
        })
        result = build_fundamentals(self.universe, ["2023-06-16", "2023-03-01", "2024-03-01", "2023-06-15"], self.cache)
        self.assertAlmostEqual(result.iloc[0].revenue_3y_growth, np.log(2)/3)
        self.assertAlmostEqual(result.iloc[1].revenue_3y_growth, np.log(2)/3)
        self.assertAlmostEqual(result.iloc[2].revenue_3y_growth, np.log(1.6)/3)
        self.assertAlmostEqual(result.iloc[2].earnings_3y_change, 0.15)
        self.assertTrue(np.isnan(result.iloc[3].revenue_3y_growth))

    def test_predecessor_cik_facts_stop_at_successor_effective_date_without_backfill(self):
        for symbol, cik, successor_cik, cutoff in [
            ("XOM", 34088, 2115436, date(2026, 7, 1)),
            ("VNOM", 1602065, 2074176, date(2025, 8, 19)),
        ]:
            with self.subTest(symbol=symbol):
                year = cutoff.year - 1
                observation = cutoff - timedelta(days=30)
                filed = cutoff - timedelta(days=20)
                self.write({"Revenues": [fact(100, f"{year}-12-31", f"{cutoff.year}-02-01", start=f"{year}-01-01")]},
                           {"EntityCommonStockSharesOutstanding": [fact(10, observation.isoformat(), filed.isoformat())]})
                payload = json.loads((self.cache / "CIK0000000001.json").read_text())
                payload["cik"] = cik
                (self.cache / f"CIK{cik:010d}.json").write_text(json.dumps(payload))
                # A successor cache exists, but its facts must never be grafted
                # backward or used after cutoff unless explicitly requested.
                payload["cik"] = successor_cik
                payload["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["val"] = 999
                (self.cache / f"CIK{successor_cik:010d}.json").write_text(json.dumps(payload))
                universe = pd.DataFrame([{"symbol": symbol, "cik": cik}])
                result = build_fundamentals(universe, [cutoff - timedelta(days=1), cutoff, cutoff + timedelta(days=1)], self.cache)
                self.assertEqual(result.iloc[0].revenue, 100)
                self.assertEqual(result.iloc[0].shares, 10)
                self.assertFalse(result.iloc[0].source_unavailable_due_to_successor)
                self.assertTrue(result.iloc[1:].source_unavailable_due_to_successor.all())
                self.assertTrue(result.iloc[1:].revenue.isna().all())
                self.assertTrue(result.iloc[1:].shares.isna().all())
                self.assertEqual(result.iloc[1].successor_cik, f"{successor_cik:010d}")
                self.assertIn("http", result.iloc[1].cik_validity_source)


if __name__ == "__main__":
    unittest.main()
