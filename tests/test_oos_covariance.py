"""Walk-forward timing, missing-future and singular-matrix guarantees."""
import unittest

import numpy as np
import pandas as pd

from factor_project.oos_covariance import walk_forward_covariance


def inputs(nstocks=10, ndays=220):
    rng = np.random.default_rng(882)
    dates = pd.bdate_range("2019-01-02", periods=ndays)
    symbols = [f"S{i:03d}" for i in range(nstocks)]
    sectors = np.resize(["A", "B"], nstocks)
    size, value = rng.normal(size=(2, nstocks))
    f = pd.DataFrame(rng.normal(scale=.006, size=(ndays, 5)), index=dates,
                     columns=["market", "sector:A", "sector:B", "size", "value"])
    x = np.column_stack([np.ones(nstocks), sectors == "A", sectors == "B", size, value])
    residual = rng.normal(scale=.012, size=(ndays, nstocks))
    returns = pd.DataFrame(f.to_numpy() @ x.T + residual, index=dates, columns=symbols)
    residuals = pd.DataFrame(residual, index=dates, columns=symbols).rename_axis("date").reset_index().melt(
        id_vars="date", var_name="symbol", value_name="residual")
    monthends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last()
    snapshots = [pd.DataFrame({"asof_date": day, "symbol": symbols, "sector": sectors,
                  "market_cap": np.exp(rng.normal(10, .4, nstocks)), "size": size, "value": value})
                 for day in monthends]
    return returns, f, residuals, pd.concat(snapshots, ignore_index=True)


def config(**overrides):
    return {"origin_start": "2019-03-29", "origin_end": "2019-03-29", "holdout_start": "2020-01-01",
        "horizons": [21, 63], "lookback": 120, "min_history": 40, "return_lookback": 40,
        "specific_lookback": 40, "specific_min_history": 25, "min_stocks": 3,
        "random_portfolios": 2, **overrides}


class WalkForwardTests(unittest.TestCase):
    def test_streaming_callback_matches_retained_matrices_without_retention(self):
        arguments = inputs()
        retained = walk_forward_covariance(*arguments, config=config())
        delivered = {}
        streamed = walk_forward_covariance(*arguments, config=config(store_forecasts=False),
            forecast_callback=lambda origin, entry: delivered.__setitem__(origin, entry))
        self.assertEqual(streamed["forecasts"], {})
        self.assertEqual(set(delivered), set(retained["forecasts"]))
        for origin, actual in delivered.items():
            expected = retained["forecasts"][origin]
            self.assertEqual(actual["symbols"], expected["symbols"])
            for collection in ["covariances", "weights", "raw_variance_diagonals"]:
                for name, array in expected[collection].items():
                    np.testing.assert_array_equal(actual[collection][name], array)
            for horizon in expected["realized"]:
                for name in ["centered_covariance", "second_moment"]:
                    np.testing.assert_array_equal(actual["realized"][horizon][name],
                                                  expected["realized"][horizon][name])
        for name in ["origins", "matrix_scores", "portfolio_scores", "summary", "portfolio_summary"]:
            pd.testing.assert_frame_equal(streamed[name], retained[name])

    def test_future_mutations_change_realization_but_never_forecasts_or_weights(self):
        returns, factors, residuals, exposures = inputs()
        first = walk_forward_covariance(returns, factors, residuals, exposures, config())
        origin = pd.Timestamp("2019-03-29")
        returns.loc[returns.index > origin] = returns.loc[returns.index > origin] * 3 + .03
        factors.loc[factors.index > origin] = 99
        residuals.loc[residuals.date > origin, "residual"] = 99
        exposures.loc[exposures.asof_date > origin, ["size", "value", "market_cap"]] = 999
        changed = walk_forward_covariance(returns, factors, residuals, exposures, config())
        original_entry, changed_entry = first["forecasts"]["2019-03-29"], changed["forecasts"]["2019-03-29"]
        self.assertEqual(original_entry["symbols"], changed_entry["symbols"])
        for collection in ["covariances", "weights"]:
            for key, values in original_entry[collection].items():
                np.testing.assert_array_equal(values, changed_entry[collection][key])
        self.assertFalse(np.allclose(original_entry["realized"][21]["second_moment"],
                                    changed_entry["realized"][21]["second_moment"]))

    def test_singular_sample_and_short_realized_matrix_are_scored_without_inverse(self):
        result = walk_forward_covariance(*inputs(nstocks=50), config=config(return_lookback=30))
        entry = result["forecasts"]["2019-03-29"]
        sample = entry["metadata"]["methods"]["sample_252_jitter"]
        self.assertLessEqual(sample["raw_rank"], 29)
        self.assertGreater(sample["diagonal_jitter"], 0)
        self.assertTrue(np.isfinite(result["matrix_scores"].gaussian_qlike_per_asset).all())
        self.assertLess(np.linalg.matrix_rank(entry["realized"][21]["centered_covariance"]), 50)
        for method, covariance in entry["covariances"].items():
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
            w = entry["weights"][f"minimum_variance::{method}"]
            self.assertAlmostEqual(w.sum(), 1.0)
            self.assertLessEqual(np.abs(w).sum(), 2.0 + 1e-10)

    def test_future_missing_keeps_exante_universe_and_skips_whole_horizon(self):
        arguments = inputs()
        arguments[0].loc["2019-04-02", "S000"] = np.nan
        result = walk_forward_covariance(*arguments, config=config())
        self.assertIn("S000", result["forecasts"]["2019-03-29"]["symbols"])
        self.assertTrue((result["origins"].status == "future_missing_returns").all())
        self.assertTrue(result["matrix_scores"].empty)
        self.assertTrue(result["portfolio_scores"].empty)
        self.assertTrue((result["origins"].future_complete_stocks == 9).all())

    def test_horizon_truncation_never_scores_partial_window(self):
        result = walk_forward_covariance(*inputs(ndays=90), config=config(horizons=[21, 63]))
        status = result["origins"].set_index("horizon").status
        self.assertEqual(status.loc[21], "scored")
        self.assertEqual(status.loc[63], "future_horizon_truncated")
        self.assertEqual(set(result["matrix_scores"].horizon), {21})

    def test_boundary_windows_excluded_and_zero_mean_calibration_is_explicit(self):
        result = walk_forward_covariance(*inputs(), config=config(holdout_start="2019-04-01"))
        self.assertTrue((result["origins"].split == "split_boundary").all())
        self.assertTrue(result["summary"].empty)
        p = result["portfolio_scores"].iloc[0]
        self.assertAlmostEqual(p.calibration_ratio, p.realized_daily_second_moment / p.forecast_daily_variance)
        entry = result["forecasts"]["2019-03-29"]
        self.assertEqual(entry["metadata"]["exposure_asof"], pd.Timestamp("2019-03-29"))
        row = result["portfolio_scores"].loc[lambda d: (d.portfolio == "equal_weight") & (d.method == "factor") & (d.horizon == 21)].iloc[0]
        w = entry["weights"]["equal_weight"]
        self.assertAlmostEqual(row.realized_daily_second_moment, w @ entry["realized"][21]["second_moment"] @ w)
        self.assertEqual(set(result["matrix_scores"].n_stocks), {10})


if __name__ == "__main__":
    unittest.main()
