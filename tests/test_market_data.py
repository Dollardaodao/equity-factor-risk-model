import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from factor_project.market_data import _fetch_symbol, align_session_returns, chart_to_daily, share_split_classification
from factor_project.fundamentals import _sec_user_agent, download_companyfacts
from factor_project.prepare_data import prepare_data, stage_market_identity


def epoch(day):
    return int(pd.Timestamp(day + " 14:30", tz="UTC").timestamp())


def sample():
    days = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    return {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "America/New_York", "instrumentType": "EQUITY"},
        "timestamp": [epoch(day) for day in days],
        "indicators": {"quote": [{"close": [25, 25, 30, 30], "volume": [1000] * 4}],
                       "adjclose": [{"adjclose": [25 / 1.04, 25, 30, 30]}]},
        "events": {
            "splits": {str(epoch(day)): {"date": epoch(day), "numerator": 2, "denominator": 1}
                       for day in ["2024-01-02", "2024-01-04"]},
            "dividends": {str(epoch("2024-01-02")): {"date": epoch("2024-01-02"), "amount": 1}},
        },
    }]}}


class MarketDataTest(unittest.TestCase):
    def test_reconstructs_then_share_prices_and_dividends_with_post_end_split(self):
        frame, audit = chart_to_daily(sample(), "TEST", "2024-01-01", "2024-01-03")
        self.assertEqual(frame.raw_close.tolist(), [100, 50, 60])
        self.assertEqual(frame.dividend.tolist(), [0, 2, 0])
        self.assertEqual(frame.split.tolist(), [1, 2, 1])
        self.assertEqual(frame.volume.tolist(), [250, 500, 500])
        self.assertEqual(len(audit["split_events"]), 2)
        self.assertIsNone(frame.date.dt.tz)
        self.assertTrue((frame.date == frame.date.dt.normalize()).all())

    def test_total_return_uses_adjusted_close_without_double_counting_split_or_cash(self):
        frame, _ = chart_to_daily(sample(), "TEST", "2024-01-01", "2024-01-03")
        self.assertTrue(pd.isna(frame.total_return.iloc[0]))
        self.assertAlmostEqual(frame.total_return.iloc[1], 0.04)
        self.assertAlmostEqual(frame.total_return.iloc[2], 0.20)

    def test_rejects_duplicate_daily_rows(self):
        payload = sample()
        result = payload["chart"]["result"][0]
        result["timestamp"][1] = result["timestamp"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            chart_to_daily(payload, "TEST", "2024-01-01", "2024-01-03")

    def test_does_not_fill_a_missing_price(self):
        payload = sample()
        payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = None
        frame, audit = chart_to_daily(payload, "TEST", "2024-01-01", "2024-01-03")
        self.assertEqual(len(frame), 2)
        self.assertEqual(audit["invalid_price_dates_dropped"], ["2024-01-02"])
        self.assertTrue(pd.isna(frame.total_return.iloc[-1]))

    def test_omitted_session_cannot_be_reported_as_a_one_day_return(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-05"]),
                              "adj_close": [100, 110, 121]})
        returns = align_session_returns(frame, pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
        self.assertTrue(pd.isna(returns.iloc[1]))
        self.assertAlmostEqual(returns.iloc[2], 0.10)

    def test_nonstandard_source_adjustment_does_not_multiply_actual_share_counts(self):
        payload = sample()
        event = payload["chart"]["result"][0]["events"]["splits"][str(epoch("2024-01-02"))]
        event["numerator"], event["denominator"] = 1324, 1000
        frame, _ = chart_to_daily(payload, "TEST", "2024-01-01", "2024-01-03")
        self.assertEqual(frame.split.iloc[1], 1)
        self.assertEqual(frame.source_adjustment_ratio.iloc[1], 1.324)
        self.assertAlmostEqual(frame.raw_close.iloc[0], 25 * 1.324 * 2)

    def test_verified_fractional_split_is_applied_and_pseudo_distribution_is_not(self):
        kind, source = share_split_classification("AAON", "2023-08-17", 1.5)
        self.assertEqual(kind, "verified fractional share split")
        self.assertIn("sec.gov", source)
        kind, _ = share_split_classification("WTRG", "2013-09-03", 1.25)
        self.assertEqual(kind, "verified fractional share split")
        kind, _ = share_split_classification("T", "2022-04-11", 1.324)
        self.assertEqual(kind, "ambiguous source adjustment; not applied to shares")
        kind, _ = share_split_classification("UNVERIFIED", "2023-08-17", 1.5)
        self.assertEqual(kind, "ambiguous source adjustment; not applied to shares")


class PublicPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_sec_user_agent_required_only_for_uncached_requests(self):
        universe = pd.DataFrame([{"symbol": "TEST", "cik": 1}])
        facts = self.root / "CIK0000000001.json"
        facts.write_text(json.dumps({"cik": 1, "facts": {}}))
        submissions = self.root / "CIK0000000001_submissions.json"
        submissions.write_text(json.dumps({"cik": 1, "tickers": ["TEST"], "name": "Test Co"}))
        with patch.dict("os.environ", {}, clear=True), patch("urllib.request.urlopen") as network:
            summary = download_companyfacts(universe, self.root)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["cached"], ["TEST"])
            network.assert_not_called()
            submissions.unlink()
            with self.assertRaisesRegex(ValueError, "SEC_USER_AGENT"):
                download_companyfacts(universe, self.root)
            network.assert_not_called()
        with patch.dict("os.environ", {"SEC_USER_AGENT": "Research Team analyst@research.example"}, clear=True):
            self.assertIn("analyst@", _sec_user_agent())
        with patch.dict("os.environ", {"SEC_USER_AGENT": "analyst@research.example"}, clear=True):
            with self.assertRaises(ValueError):
                _sec_user_agent()

    def test_verified_yahoo_cache_reused_on_a_later_day_without_mutating_raw(self):
        raw_dir = self.root / "raw"
        raw_dir.mkdir()
        raw = raw_dir / "TEST_chart.json"
        raw.write_text(json.dumps(sample()))
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        source = {"url": "https://query1.finance.yahoo.com/v8/finance/chart/TEST",
                  "parameters": {"period1": epoch("2023-12-31"), "period2": epoch("2024-01-05"),
                                 "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"},
                  "raw_sha256": digest, "retrieved_at_utc": "2024-01-05T00:00:00+00:00"}
        metadata = raw_dir / "TEST_source.json"
        metadata.write_text(json.dumps(source))
        before = (raw.read_bytes(), metadata.read_bytes())
        result = _fetch_symbol("TEST", self.root, "2024-01-01", "2024-01-03")
        self.assertEqual(result["rows"], 3)
        self.assertEqual((raw.read_bytes(), metadata.read_bytes()), before)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            _fetch_symbol("TEST", self.root, "2020-01-01", "2024-01-03")
        raw.write_text("{}")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            _fetch_symbol("TEST", self.root, "2024-01-01", "2024-01-03")

    def make_downloaded_market(self):
        cache = self.root / "data/market_downloaded"
        cache.mkdir(parents=True)
        details = []
        for symbol in ["SPY", "CHRD", "SUPN"]:
            frame, _ = chart_to_daily(sample(), symbol, "2024-01-01", "2024-01-04")
            path = cache / f"{symbol}.csv"
            frame.to_csv(path, index=False)
            details.append({"symbol": symbol, "processed_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        rules = {"policy": "Dated issuer identity review, no magnitude clipping.", "rules": {
            "CHRD": {"action": "start_at", "date": "2024-01-03", "reason": "New security segment"},
            "SUPN": {"action": "exclude", "reason": "Unverified source identity"}}}
        (self.root / "market_identity_rules.json").write_text(json.dumps(rules))
        audit = {"status": "complete", "symbols": details, "requested_symbols": ["CHRD", "SUPN", "SPY"],
                 "succeeded": 3, "failed": []}
        return cache, audit

    def test_identity_staging_keeps_raw_prices_and_clears_segment_boundary_return(self):
        cache, downloaded = self.make_downloaded_market()
        originals = {path.name: path.read_bytes() for path in cache.glob("*.csv")}
        universe = pd.DataFrame({"symbol": ["CHRD", "SUPN"]})
        audit = stage_market_identity(self.root, {"market_cache": "data/market"}, universe, downloaded)
        self.assertEqual(audit["status"], "complete")
        self.assertFalse((self.root / "data/market/SUPN.csv").exists())
        chrd = pd.read_csv(self.root / "data/market/CHRD.csv")
        self.assertEqual(chrd.date.iloc[0], "2024-01-03")
        self.assertTrue(pd.isna(chrd.total_return.iloc[0]))
        self.assertEqual(audit["model_identity_screen"]["rules"][1]["model_rows"], 0)
        self.assertEqual({path.name: path.read_bytes() for path in cache.glob("*.csv")}, originals)
        # A failed retry keeps the previous validated directory and audit intact.
        old_csv = (self.root / "data/market/CHRD.csv").read_bytes()
        old_audit = (self.root / "data/market_audit.json").read_bytes()
        downloaded["symbols"][1]["processed_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            stage_market_identity(self.root, {"market_cache": "data/market"}, universe, downloaded)
        self.assertEqual((self.root / "data/market/CHRD.csv").read_bytes(), old_csv)
        self.assertEqual((self.root / "data/market_audit.json").read_bytes(), old_audit)

    def test_preparation_failure_is_recorded_and_does_not_claim_success(self):
        (self.root / "config.json").write_text(json.dumps({"sec_cache": "../private", "benchmark_file": "outside.csv"}))
        with self.assertRaisesRegex(ValueError, "Portable setup"):
            prepare_data(self.root)
        result = json.loads((self.root / "data/preparation_manifest.json").read_text())
        self.assertEqual(result["status"], "failed")

    def test_prepare_data_connects_public_caches_and_publishes_only_validated_prices(self):
        _, downloaded = self.make_downloaded_market()
        config = {"sec_cache": "data/sec", "benchmark_file": "data/benchmark_returns_daily.csv",
                  "market_cache": "data/market", "universe_file": "universe.csv",
                  "price_start": "2024-01-01", "price_end": "2024-01-04"}
        (self.root / "config.json").write_text(json.dumps(config))
        pd.DataFrame({"symbol": ["CHRD", "SUPN"], "cik": [1, 2]}).to_csv(self.root / "universe.csv", index=False)
        benchmark = {"panel_sha256": "benchmark-snapshot", "rows": 4,
                     "first_date": "2024-01-01", "last_date": "2024-01-04"}
        sec = {"failed": [], "files": [], "identities": [
            {"symbol": "CHRD", "verified": True, "ticker_match": True},
            {"symbol": "SUPN", "verified": True, "ticker_match": False}]}
        with patch("factor_project.prepare_data._benchmark", return_value=benchmark), \
                patch("factor_project.prepare_data.download_companyfacts", return_value=sec), \
                patch("factor_project.prepare_data.fetch_market", return_value=downloaded) as acquire:
            result = prepare_data(self.root)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["excluded_symbols"], ["SUPN"])
        self.assertEqual(result["model_market_symbols"], 2)
        self.assertEqual(acquire.call_args.kwargs["audit_path"], (self.root / "data/market_download_audit.json").resolve())
        self.assertFalse((self.root / "data/market/SUPN.csv").exists())
        saved = json.loads((self.root / "data/preparation_manifest.json").read_text())
        self.assertEqual(saved["market_audit_sha256"], result["market_audit_sha256"])


if __name__ == "__main__":
    unittest.main()
