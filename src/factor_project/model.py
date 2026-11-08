"""Daily characteristic-factor attribution with strictly lagged exposures.

This is a research attribution model, not a forecast or trading backtest.  Factor
returns are coefficients per unit of exposure; their mimicking portfolios need
not have fixed gross leverage and are before trading/borrowing costs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_STYLE_FACTORS = (
    "size", "momentum", "quality", "growth", "value", "low_vol", "beta_control"
)


def _dates(values: object, label: str) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if result.hasnans:
        raise ValueError(f"{label} contains missing dates")
    if result.tz is not None:
        result = result.tz_localize(None)
    return result.normalize()


def standardize_descriptors(
    frame: pd.DataFrame,
    columns: Sequence[str],
    market_cap_column: str = "market_cap",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Winsorize one cross section; cap-center and equal-weight-standardize.

    Apply this separately to each as-of snapshot, after defining descriptor
    directions/composites. Missing or invalid descriptors/caps remain NaN;
    constant descriptors become NaN rather than an invented neutral exposure.
    The returned copy retains other columns. Its ``standardization_report``
    attribute records coverage and transformation parameters for each column.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    if len(set(columns)) != len(columns):
        raise ValueError("descriptor columns must be unique")
    missing = set(columns).union({market_cap_column}).difference(frame.columns)
    if missing:
        raise ValueError(f"missing descriptor inputs: {sorted(missing)}")
    result = frame.copy()
    caps = pd.to_numeric(frame[market_cap_column], errors="raise").to_numpy(float)
    cap_valid = np.isfinite(caps) & (caps > 0)
    report = {}
    for column in columns:
        raw = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
        valid = cap_valid & np.isfinite(raw)
        values = raw[valid]
        z = np.full(len(frame), np.nan)
        detail = {"n_valid": int(valid.sum()), "n_missing_or_invalid": int((~valid).sum())}
        if len(values) < 2:
            detail["status"] = "insufficient_observations"
        else:
            lower, upper = np.quantile(values, [lower_quantile, upper_quantile])
            clipped = np.clip(values, lower, upper)
            mean = float(np.average(clipped, weights=caps[valid]))
            std = float(np.std(clipped, ddof=0))
            detail.update(lower=float(lower), upper=float(upper), cap_weighted_mean=mean,
                          equal_weighted_std=std)
            if std <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(clipped)))):
                detail["status"] = "constant_descriptor"
            else:
                z[valid] = (clipped - mean) / std
                detail["status"] = "ok"
        result[column] = z
        report[column] = detail
    result.attrs["standardization_report"] = report
    return result


def fit_factor_model(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    risk_free: pd.Series | None = None,
    factor_columns: Sequence[str] | None = None,
    min_stocks: int = 30,
    *,
    min_residual_df: int = 20,
    max_condition_number: float = 1e8,
    store_mimicking_weights: bool = True,
    store_portfolio_diagnostics: bool = True,
) -> dict[str, pd.DataFrame]:
    """Estimate daily market, sector and characteristic factor returns.

    ``returns`` is wide, indexed by date, with symbol columns and DECIMAL daily
    total returns. If ``risk_free`` is supplied (decimal daily simple returns),
    it is subtracted by date. If omitted, ``returns`` MUST already be excess
    returns. Missing risk-free observations cause that date to be skipped.

    ``exposures`` contains FULL cross-sectional snapshots, not incremental
    updates, with ``asof_date,symbol,sector,market_cap`` and selected style
    columns. Each snapshot is first usable on a STRICTLY later calendar date;
    the latest eligible snapshot is held until a newer one becomes eligible.
    The caller must construct genuinely point-in-time descriptors upstream.
    Do not pass restated/current fundamentals as historical observations.

    A unit market intercept and all sector dummies are identified by the
    constraint sum(sector_cap_share * sector_return) == 0. The constraint uses
    the day's complete-case estimation universe and lagged snapshot caps.
    WLS minimizes sum(sqrt(market_cap) * residual**2), so the OLS transformation
    multiplies rows by market_cap**0.25. Rank/conditioning failures are reported
    and skipped; no ridge, interpolation or missing-exposure fill is applied.

    Outputs: ``factor_returns`` and ``diagnostics`` (date index), ``residuals``
    (long), ``mimicking_weights`` (long), and ``portfolio_diagnostics`` (long).
    Sector columns use ``sector:<name>``. Missing/unestimated returns remain
    NaN. ``half_l1_weight_change`` compares consecutive available coefficient
    portfolios, excludes market drift/costs, and is NOT execution turnover.
    Set the two storage flags false for large covariance experiments; this
    avoids storing millions of repeated portfolio-weight rows without changing
    the regression, residuals, or numerical diagnostics.
    """
    styles = list(DEFAULT_STYLE_FACTORS if factor_columns is None else factor_columns)
    if len(set(styles)) != len(styles):
        raise ValueError("factor_columns must be unique")
    if any(not isinstance(s, str) or not s or s == "market" or s.startswith("sector:")
           or s in {"asof_date", "symbol", "sector", "market_cap"} for s in styles):
        raise ValueError("style names must be nonempty strings without reserved names")
    if min_stocks < 1 or min_residual_df < 1:
        raise ValueError("min_stocks and min_residual_df must be positive")
    if not np.isfinite(max_condition_number) or max_condition_number <= 1:
        raise ValueError("max_condition_number must be finite and > 1")
    required = ["asof_date", "symbol", "sector", "market_cap", *styles]
    missing = set(required).difference(exposures.columns)
    if missing:
        raise ValueError(f"missing exposure columns: {sorted(missing)}")
    if returns.columns.duplicated().any():
        raise ValueError("returns has duplicate symbols")
    if any(not isinstance(s, str) or not s for s in returns.columns):
        raise ValueError("returns symbols must be nonempty strings")
    observed = returns.copy()
    observed.index = _dates(observed.index, "returns")
    if observed.index.duplicated().any():
        raise ValueError("returns has duplicate daily dates")
    observed = observed.sort_index().apply(pd.to_numeric, errors="raise")
    snapshots = exposures.loc[:, required].copy()
    snapshots["asof_date"] = _dates(snapshots["asof_date"], "exposures")
    if snapshots["symbol"].map(lambda s: not isinstance(s, str) or not s).any():
        raise ValueError("exposure symbols must be nonempty strings")
    if snapshots.duplicated(["asof_date", "symbol"]).any():
        raise ValueError("exposures has duplicate asof_date/symbol keys")
    for column in ["market_cap", *styles]:
        snapshots[column] = pd.to_numeric(snapshots[column], errors="raise")
    invalid_sector = snapshots["sector"].map(lambda s: not isinstance(s, str) or not s.strip())
    snapshots.loc[invalid_sector, "sector"] = np.nan
    grouped = {date: group.set_index("symbol") for date, group in snapshots.groupby("asof_date")}
    snapshot_dates = pd.DatetimeIndex(sorted(grouped))
    sectors_all = sorted(snapshots["sector"].dropna().unique())
    output_columns = ["market", *[f"sector:{s}" for s in sectors_all], *styles]
    factors = pd.DataFrame(np.nan, index=observed.index, columns=output_columns)
    factors.index.name = "date"
    if risk_free is None:
        rf = pd.Series(0.0, index=observed.index)
    else:
        if not isinstance(risk_free, pd.Series):
            raise TypeError("risk_free must be a Series indexed by date")
        rf = risk_free.copy()
        rf.index = _dates(rf.index, "risk_free")
        if rf.index.duplicated().any():
            raise ValueError("risk_free has duplicate daily dates")
        rf = pd.to_numeric(rf, errors="raise").reindex(observed.index)
    diagnostics, residual_records, weight_records, portfolio_records = [], [], [], []
    previous_weights: dict[str, pd.Series] = {}
    for date, daily in observed.iterrows():
        diagnostic = {"date": date, "status": "skipped", "reason": "", "asof_date": pd.NaT,
                      "n_snapshot": 0, "n_with_return": 0, "n_stocks": 0,
                      "dropped_missing_return": 0, "dropped_missing_exposure": 0,
                      "coverage": np.nan, "n_params": np.nan, "rank": np.nan,
                      "condition_number": np.nan, "residual_df": np.nan,
                      "weighted_r2": np.nan, "r2": np.nan, "residual_std": np.nan,
                      "risk_free": rf.loc[date], "constraint_error": np.nan,
                      "max_mimicking_exposure_error": np.nan,
                      "max_mimicking_gross": np.nan}
        idx = snapshot_dates.searchsorted(date, side="left") - 1
        if idx < 0:
            diagnostic["reason"] = "no_strictly_prior_exposure_snapshot"
            diagnostics.append(diagnostic)
            continue
        asof = snapshot_dates[idx]
        snapshot = grouped[asof]
        diagnostic.update(asof_date=asof, n_snapshot=len(snapshot))
        y_raw = daily.reindex(snapshot.index).to_numpy(float)
        return_valid = np.isfinite(y_raw)
        caps = snapshot["market_cap"].to_numpy(float)
        z_all = snapshot[styles].to_numpy(float)
        complete = ((np.isfinite(caps) & (caps > 0)) & snapshot["sector"].notna().to_numpy()
                    & np.isfinite(z_all).all(axis=1))
        valid = return_valid & complete
        n = int(valid.sum())
        diagnostic.update(n_with_return=int(return_valid.sum()), n_stocks=n,
                          dropped_missing_return=int((~return_valid).sum()),
                          dropped_missing_exposure=int((return_valid & ~complete).sum()),
                          coverage=n / len(snapshot) if len(snapshot) else np.nan)
        if not np.isfinite(rf.loc[date]):
            diagnostic["reason"] = "missing_risk_free"
            diagnostics.append(diagnostic)
            continue
        if n < min_stocks:
            diagnostic["reason"] = "insufficient_stocks"
            diagnostics.append(diagnostic)
            continue
        selected = snapshot.iloc[np.flatnonzero(valid)]
        symbols = selected.index
        y = y_raw[valid] - float(rf.loc[date])
        cap = caps[valid]
        sectors = sorted(selected["sector"].unique())
        dummy = np.column_stack([(selected["sector"] == s).to_numpy(float) for s in sectors])
        z = z_all[valid]
        x = np.column_stack([np.ones(n), dummy, z])
        names = ["market", *[f"sector:{s}" for s in sectors], *styles]
        cap_scaled = cap / cap.max()
        sector_share = dummy.T @ (cap_scaled / cap_scaled.sum())
        constraint = np.r_[0.0, sector_share, np.zeros(len(styles))]
        # An orthonormal nullspace eliminates the one identification constraint.
        _, _, constraint_vt = np.linalg.svd(constraint[None, :], full_matrices=True)
        null = constraint_vt[1:].T
        design = x @ null
        nparams = design.shape[1]
        diagnostic.update(n_params=nparams, residual_df=n - nparams)
        if n - nparams < min_residual_df:
            diagnostic["reason"] = "insufficient_residual_df"
            diagnostics.append(diagnostic)
            continue
        # Normalize weights to avoid overflow and to make their scale explicit.
        w = np.sqrt(cap_scaled)
        sqrt_w = np.sqrt(w)
        a = design * sqrt_w[:, None]
        u, singular, vt = np.linalg.svd(a, full_matrices=False)
        tolerance = np.finfo(float).eps * max(a.shape) * singular[0]
        rank = int((singular > tolerance).sum())
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
        diagnostic.update(rank=rank, condition_number=condition)
        if rank < nparams:
            diagnostic["reason"] = "rank_deficient"
            diagnostics.append(diagnostic)
            continue
        if condition > max_condition_number:
            diagnostic["reason"] = "ill_conditioned"
            diagnostics.append(diagnostic)
            continue
        # Each row of coefficient_map is a stock-weight portfolio whose return
        # exactly equals the corresponding constrained regression coefficient.
        coefficient_map = (null @ ((vt.T / singular) @ u.T)) * sqrt_w[None, :]
        f = coefficient_map @ y
        fitted = x @ f
        residual = y - fitted
        weighted_sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        redundant_direction = np.r_[1.0, -np.ones(len(sectors)), np.zeros(len(styles))]
        expected_projection = np.eye(x.shape[1]) + np.outer(redundant_direction, constraint)
        exposure_error = float(np.max(np.abs(coefficient_map @ x - expected_projection)))
        diagnostic.update(status="ok", reason="", constraint_error=float(abs(constraint @ f)),
                          max_mimicking_exposure_error=exposure_error,
                          weighted_r2=1 - float(np.sum(w * residual**2)) / weighted_sst
                          if weighted_sst > 0 else np.nan,
                          r2=1 - float(residual @ residual) / sst if sst > 0 else np.nan,
                          residual_std=float(np.sqrt((residual @ residual) / (n - nparams))),
                          max_mimicking_gross=float(np.abs(coefficient_map).sum(axis=1).max()))
        factors.loc[date, names] = f
        diagnostics.append(diagnostic)
        for i, symbol in enumerate(symbols):
            residual_records.append({"date": date, "symbol": symbol, "asof_date": asof,
                                     "observed_excess_return": y[i], "fitted_return": fitted[i],
                                     "residual": residual[i], "wls_weight": w[i]})
        for j, name in enumerate(names) if (store_mimicking_weights or store_portfolio_diagnostics) else []:
            current = pd.Series(coefficient_map[j], index=symbols)
            previous = previous_weights.get(name)
            change = np.nan if previous is None else float(
                current.subtract(previous, fill_value=0).abs().sum() / 2)
            if store_portfolio_diagnostics:
                portfolio_records.append({"date": date, "factor": name,
                                          "gross_leverage": float(current.abs().sum()),
                                          "net_weight": float(current.sum()),
                                          "half_l1_weight_change": change})
            previous_weights[name] = current
            if store_mimicking_weights:
                for symbol, weight in current.items():
                    weight_records.append({"date": date, "factor": name, "symbol": symbol,
                                           "weight": float(weight)})
    diagnostic_columns = ["date", "status", "reason", "asof_date", "n_snapshot", "n_with_return",
                          "n_stocks", "dropped_missing_return", "dropped_missing_exposure",
                          "coverage", "n_params", "rank", "condition_number", "residual_df",
                          "weighted_r2", "r2", "residual_std", "risk_free", "constraint_error",
                          "max_mimicking_exposure_error", "max_mimicking_gross"]
    return {
        "factor_returns": factors,
        "diagnostics": pd.DataFrame(diagnostics, columns=diagnostic_columns).set_index("date"),
        "residuals": pd.DataFrame(residual_records, columns=["date", "symbol", "asof_date",
            "observed_excess_return", "fitted_return", "residual", "wls_weight"]),
        "mimicking_weights": pd.DataFrame(weight_records, columns=["date", "factor", "symbol", "weight"]),
        "portfolio_diagnostics": pd.DataFrame(portfolio_records, columns=["date", "factor",
            "gross_leverage", "net_weight", "half_l1_weight_change"]),
    }
