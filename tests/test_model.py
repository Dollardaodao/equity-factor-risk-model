"""Identification, chronology and accounting checks on the regression core."""

import unittest

import numpy as np
import pandas as pd

from factor_project.model import fit_factor_model, standardize_descriptors


def make_snapshot(n=120, asof="2020-01-31", seed=14):
    rng = np.random.default_rng(seed)
    snapshot = pd.DataFrame({
        "asof_date": pd.Timestamp(asof),
        "symbol": [f"S{i:03}" for i in range(n)],
        "sector": np.resize(["Energy", "Finance", "Technology"], n),
        "market_cap": np.exp(rng.normal(8, 2, n)),
        "size": rng.normal(size=n),
        "value": rng.normal(size=n),
    })
    return standardize_descriptors(snapshot, ["size", "value"])


def synthetic_return(snapshot, market=0.005, size=0.002, value=-0.004, noise=None):
    sector_order = sorted(snapshot.sector.unique())
    sector_share = np.array([snapshot.loc[snapshot.sector == s, "market_cap"].sum()
                             for s in sector_order])
    sector_share /= sector_share.sum()
    sector_return = np.array([0.001, -0.003, 0.002])
    sector_return -= sector_share @ sector_return
    sector_map = dict(zip(sector_order, sector_return))
    y = (market + snapshot.sector.map(sector_map).to_numpy() + size * snapshot["size"].to_numpy()
         + value * snapshot["value"].to_numpy())
    if noise is not None:
        y += noise
    coefficients = {"market": market, "size": size, "value": value,
                    **{f"sector:{s}": sector_map[s] for s in sector_order}}
    return y, coefficients


class FactorModelTests(unittest.TestCase):
    def test_large_run_storage_flags_preserve_estimates(self):
        snapshot = make_snapshot()
        y, _ = synthetic_return(snapshot, noise=np.random.default_rng(16).normal(0, .01, len(snapshot)))
        observed = pd.DataFrame([y, y * .7], index=["2020-02-03", "2020-02-04"], columns=snapshot.symbol)
        full = fit_factor_model(observed, snapshot, factor_columns=["size", "value"])
        compact = fit_factor_model(observed, snapshot, factor_columns=["size", "value"],
                                   store_mimicking_weights=False, store_portfolio_diagnostics=False)
        for key in ["factor_returns", "residuals", "diagnostics"]:
            pd.testing.assert_frame_equal(full[key], compact[key])
        self.assertTrue(compact["mimicking_weights"].empty)
        self.assertTrue(compact["portfolio_diagnostics"].empty)

    def test_recovers_known_constrained_coefficients_and_portfolio_returns(self):
        snapshot = make_snapshot()
        y, expected = synthetic_return(snapshot)
        returns = pd.DataFrame([y + 0.0001], index=["2020-02-03"], columns=snapshot.symbol)
        result = fit_factor_model(returns, snapshot, pd.Series([0.0001], index=["2020-02-03"]),
                                  factor_columns=["size", "value"])
        actual = result["factor_returns"].iloc[0]
        for name, value in expected.items():
            self.assertAlmostEqual(actual[name], value, places=12)
        diagnostic = result["diagnostics"].iloc[0]
        self.assertEqual(diagnostic.status, "ok")
        self.assertLess(diagnostic.constraint_error, 1e-12)
        self.assertLess(diagnostic.max_mimicking_exposure_error, 1e-12)
        self.assertLess(result["residuals"].residual.abs().max(), 1e-12)
        self.assertAlmostEqual(diagnostic.weighted_r2, 1.0, places=12)
        weights = result["mimicking_weights"].pivot(index="factor", columns="symbol", values="weight")
        portfolio_returns = weights @ pd.Series(y, index=snapshot.symbol)
        np.testing.assert_allclose(portfolio_returns, actual.reindex(weights.index), atol=1e-12)
        self.assertAlmostEqual(weights.loc["market"].sum(), 1.0, places=12)
        self.assertAlmostEqual(weights.loc["size"].sum(), 0.0, places=12)

    def test_strict_lag_and_future_snapshot_cannot_change_prior_estimates(self):
        snapshot = make_snapshot()
        y, expected = synthetic_return(snapshot)
        future = make_snapshot(asof="2020-02-28", seed=50)
        dates = ["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"]
        returns = pd.DataFrame([y] * 4, index=dates, columns=snapshot.symbol)
        result = fit_factor_model(returns, pd.concat([snapshot, future]), factor_columns=["size", "value"])
        self.assertTrue(result["factor_returns"].iloc[0].isna().all())
        self.assertEqual(result["diagnostics"].iloc[0].reason, "no_strictly_prior_exposure_snapshot")
        self.assertEqual(result["diagnostics"].loc["2020-02-28", "asof_date"], pd.Timestamp("2020-01-31"))
        self.assertEqual(result["diagnostics"].loc["2020-03-02", "asof_date"], pd.Timestamp("2020-02-28"))
        self.assertAlmostEqual(result["factor_returns"].loc["2020-02-03", "size"], expected["size"])
        without_future = fit_factor_model(returns.iloc[:3], snapshot, factor_columns=["size", "value"])
        pd.testing.assert_frame_equal(result["factor_returns"].iloc[:3], without_future["factor_returns"])

    def test_missing_data_is_dropped_and_counted_without_neutral_imputation(self):
        snapshot = make_snapshot()
        y, _ = synthetic_return(snapshot)
        y[0] = np.nan
        snapshot.loc[1, "value"] = np.nan
        snapshot.loc[2, "market_cap"] = -1
        returns = pd.DataFrame([y], index=["2020-02-03"], columns=snapshot.symbol)
        result = fit_factor_model(returns, snapshot, factor_columns=["size", "value"])
        d = result["diagnostics"].iloc[0]
        self.assertEqual(d.status, "ok")
        self.assertEqual(d.n_stocks, len(snapshot) - 3)
        self.assertEqual(d.dropped_missing_return, 1)
        self.assertEqual(d.dropped_missing_exposure, 2)
        self.assertEqual(set(snapshot.symbol[:3]) & set(result["residuals"].symbol), set())

    def test_wls_residuals_are_weighted_orthogonal_to_every_design_column(self):
        snapshot = make_snapshot()
        noise = np.random.default_rng(200).normal(scale=0.01, size=len(snapshot))
        y, _ = synthetic_return(snapshot, noise=noise)
        returns = pd.DataFrame([y], index=["2020-02-03"], columns=snapshot.symbol)
        result = fit_factor_model(returns, snapshot, factor_columns=["size", "value"])
        residuals = result["residuals"].set_index("symbol").reindex(snapshot.symbol)
        wr = residuals.wls_weight.to_numpy() * residuals.residual.to_numpy()
        x = np.column_stack([np.ones(len(snapshot)), pd.get_dummies(snapshot.sector).to_numpy(),
                             snapshot[["size", "value"]].to_numpy()])
        np.testing.assert_allclose(x.T @ wr, 0, atol=1e-12)
        expected_w = np.sqrt(snapshot.market_cap / snapshot.market_cap.max())
        np.testing.assert_allclose(residuals.wls_weight, expected_w)

    def test_rank_failure_and_missing_risk_free_are_visible(self):
        snapshot = make_snapshot()
        snapshot["value"] = snapshot["size"]
        returns = pd.DataFrame([np.arange(len(snapshot)) / 10000], index=["2020-02-03"],
                               columns=snapshot.symbol)
        result = fit_factor_model(returns, snapshot, factor_columns=["size", "value"])
        self.assertEqual(result["diagnostics"].iloc[0].reason, "rank_deficient")
        self.assertTrue(result["factor_returns"].isna().all().all())
        self.assertTrue(result["residuals"].empty)
        missing_rf = fit_factor_model(returns, make_snapshot(),
            risk_free=pd.Series(dtype=float, index=pd.DatetimeIndex([])), factor_columns=["size", "value"])
        self.assertEqual(missing_rf["diagnostics"].iloc[0].reason, "missing_risk_free")

    def test_duplicates_are_rejected(self):
        snapshot = make_snapshot()
        returns = pd.DataFrame([np.zeros(len(snapshot))], index=["2020-02-03"], columns=snapshot.symbol)
        with self.assertRaisesRegex(ValueError, "duplicate asof_date/symbol"):
            fit_factor_model(returns, pd.concat([snapshot, snapshot.iloc[:1]]), factor_columns=["size", "value"])

    def test_standardization_preserves_missingness_and_caps_center(self):
        frame = pd.DataFrame({"market_cap": [1, 2, 3, 4], "x": [1, 2, 4, np.nan],
                              "constant": [5, 5, 5, 5]})
        output = standardize_descriptors(frame, ["x", "constant"], lower_quantile=0, upper_quantile=1)
        valid = output.x.notna()
        self.assertAlmostEqual(np.average(output.loc[valid, "x"], weights=frame.loc[valid, "market_cap"]), 0)
        self.assertAlmostEqual(output.loc[valid, "x"].std(ddof=0), 1)
        self.assertTrue(np.isnan(output.x.iloc[-1]))
        self.assertTrue(output.constant.isna().all())
        self.assertEqual(output.attrs["standardization_report"]["constant"]["status"], "constant_descriptor")


if __name__ == "__main__":
    unittest.main()
