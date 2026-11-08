"""Temporal, PSD and missing-data guarantees for historical risk estimates."""

import unittest

import numpy as np
import pandas as pd

from factor_project.risk import estimate_risk


def risk_inputs(n=320):
    rng = np.random.default_rng(91)
    dates = pd.bdate_range("2020-01-02", periods=n)
    sectors = rng.normal(scale=0.004, size=n)
    factors = pd.DataFrame({"market": rng.normal(scale=0.01, size=n),
        "sector:Energy": sectors, "sector:Technology": -sectors,
        "size": rng.normal(scale=0.003, size=n), "value": rng.normal(scale=0.002, size=n)}, index=dates)
    symbols = ["A", "B", "C", "D", "MISSING"]
    snapshot = pd.DataFrame({"asof_date": dates[-1], "symbol": symbols,
        "sector": ["Energy", "Technology", "Energy", "Technology", "Technology"],
        "size": [-1, -0.5, 0.2, 1, 0], "value": [0.5, 1, -1, -0.5, 0]})
    rows = []
    for symbol in symbols:
        observations = dates if symbol != "MISSING" else dates[-30:]
        for day, value in zip(observations, rng.normal(scale=0.012, size=len(observations))):
            rows.append({"date": day, "symbol": symbol, "residual": value})
    return factors, pd.DataFrame(rows), snapshot


class RiskTests(unittest.TestCase):
    def test_psd_aggregation_horizon_scaling_and_missing_specific_risk(self):
        factors, residuals, snapshot = risk_inputs()
        result = estimate_risk(factors, residuals, snapshot, factors.index[-1])
        self.assertEqual(result["diagnostics"]["status"], "ok")
        self.assertEqual(result["diagnostics"]["stocks_estimated"], 4)
        self.assertEqual(result["diagnostics"]["dropped_missing_specific_risk"], 1)
        self.assertNotIn("MISSING", set(result["stock_risk"].symbol))
        self.assertGreaterEqual(np.linalg.eigvalsh(result["factor_covariance"]).min(), -1e-15)
        covariance = result["stock_covariance"]
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-15)
        stock_risk = result["stock_risk"].set_index("symbol").reindex(covariance.index)
        np.testing.assert_allclose(stock_risk.annual_vol**2, np.diag(covariance) * 252)
        np.testing.assert_allclose(stock_risk.vol_21d**2, np.diag(covariance) * 21)
        np.testing.assert_allclose(stock_risk.factor_daily_variance + stock_risk.specific_daily_variance,
                                   np.diag(covariance))

    def test_future_factor_returns_and_residuals_do_not_change_estimate(self):
        factors, residuals, snapshot = risk_inputs()
        cutoff = factors.index[250]
        snapshot["asof_date"] = cutoff
        original = estimate_risk(factors, residuals, snapshot, cutoff)
        changed_factors = factors.copy()
        changed_factors.loc[changed_factors.index > cutoff] = 999
        changed_residuals = residuals.copy()
        changed_residuals.loc[changed_residuals.date > cutoff, "residual"] = 999
        changed = estimate_risk(changed_factors, changed_residuals, snapshot, cutoff)
        for key in ["factor_covariance", "specific_variance", "stock_risk", "stock_covariance"]:
            pd.testing.assert_frame_equal(original[key], changed[key])

    def test_missing_industry_rows_are_dropped_not_zero_filled(self):
        factors, residuals, snapshot = risk_inputs()
        factors.loc[factors.index[-40:], "sector:Energy"] = np.nan
        result = estimate_risk(factors, residuals, snapshot, factors.index[-1], lookback=200)
        self.assertEqual(result["diagnostics"]["factor_complete_rows"], 160)
        self.assertEqual(result["diagnostics"]["factor_rows_dropped_missing"], 40)
        self.assertEqual(result["diagnostics"]["latest_factor_date"], str(factors.index[-41].date()))
        failing = estimate_risk(factors, residuals, snapshot, factors.index[-1], min_history=180, lookback=200)
        self.assertEqual(failing["diagnostics"]["status"], "insufficient_factor_history")
        self.assertTrue(failing["factor_covariance"].isna().all().all())
        self.assertTrue(failing["stock_risk"].empty)

    def test_future_exposure_and_duplicate_residual_keys_raise(self):
        factors, residuals, snapshot = risk_inputs()
        with self.assertRaisesRegex(ValueError, "after the risk cutoff"):
            estimate_risk(factors, residuals, snapshot, factors.index[-2])
        with self.assertRaisesRegex(ValueError, "duplicate date/symbol"):
            estimate_risk(factors, pd.concat([residuals, residuals.iloc[:1]]), snapshot, factors.index[-1])

    def test_diagonal_shrinkage_is_exact_and_units_remain_daily(self):
        factors, residuals, snapshot = risk_inputs()
        unshrunk = estimate_risk(factors, residuals, snapshot, factors.index[-1], shrinkage=0)
        shrunk = estimate_risk(factors, residuals, snapshot, factors.index[-1], shrinkage=0.2)
        covariance = unshrunk["factor_covariance"].to_numpy()
        expected = 0.8 * covariance + 0.2 * np.diag(np.diag(covariance))
        np.testing.assert_allclose(shrunk["factor_covariance"], expected, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
