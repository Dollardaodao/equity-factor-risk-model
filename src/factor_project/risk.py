"""Historical factor/specific covariance estimates for research attribution.

These are conditional covariance estimates, not out-of-sample risk validation.
They exclude transaction costs and assume exposures and daily covariances remain
constant when daily variances are scaled to 21 days or 252 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _daily_index(values, label):
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.hasnans:
        raise ValueError(f"{label} contains missing dates")
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _weighted_covariance(values: np.ndarray, ages: np.ndarray, half_life: float) -> np.ndarray:
    """Centered weighted sample covariance; ages count observed market sessions."""
    weights = np.exp2(-ages / half_life)
    weights /= weights.sum()
    centered = values - weights @ values
    correction = 1.0 - weights @ weights
    if correction <= np.finfo(float).eps:
        raise ValueError("half_life leaves fewer than two effective observations")
    return (centered.T * weights) @ centered / correction


def estimate_risk(
    factor_returns: pd.DataFrame,
    residuals: pd.DataFrame,
    latest_exposures: pd.DataFrame,
    asof_date,
    min_history: int = 126,
    lookback: int = 756,
    half_life: float = 63,
    shrinkage: float = 0.2,
) -> dict:
    """Estimate daily factor covariance, specific variance and stock risk.

    Inputs use DECIMAL daily excess-return coefficients and residuals from
    ``fit_factor_model``. ``latest_exposures`` must contain one full snapshot,
    with ``asof_date,symbol,sector`` and the model's style columns. A snapshot
    dated after the risk cutoff raises; no future factor/residual data is used.

    Factor covariance uses up to ``lookback`` model-session rows through the
    cutoff, drops rows missing any active factor, and requires ``min_history``
    complete rows. Exponential ages preserve gaps; missing industry returns are
    never zero-filled. Covariance is centered, adjusted for the effective sample
    size, and shrunk ``shrinkage`` toward its diagonal to preserve PSD.

    Specific variances use the last 252 model sessions with the same half-life,
    require 60 finite residual observations per symbol, and remain missing when
    unavailable. Such stocks are dropped from stock risk, not assigned average
    specific risk. ``stock_covariance`` is X F X' + diag(specific_variance).
    ``annual_vol`` and ``vol_21d`` are sqrt(252) and sqrt(21) scalings of daily
    model volatility; serial correlation and changing exposure are not modeled.
    """
    if not isinstance(min_history, int) or min_history < 2:
        raise ValueError("min_history must be an integer >= 2")
    if not isinstance(lookback, int) or lookback < min_history:
        raise ValueError("lookback must be an integer >= min_history")
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive and finite")
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    cutoff = _daily_index([asof_date], "asof_date")[0]
    factors = factor_returns.copy()
    factors.index = _daily_index(factors.index, "factor_returns")
    if factors.index.duplicated().any() or factors.columns.duplicated().any():
        raise ValueError("factor_returns dates and factor columns must be unique")
    if "market" not in factors or any(not isinstance(c, str) for c in factors.columns):
        raise ValueError("factor_returns must have named market/sector/style columns")
    factors = factors.sort_index().apply(pd.to_numeric, errors="raise")
    factors = factors.loc[factors.index <= cutoff].replace([np.inf, -np.inf], np.nan)
    required = {"date", "symbol", "residual"}
    if not required.issubset(residuals.columns):
        raise ValueError(f"missing residual columns: {sorted(required.difference(residuals.columns))}")
    residual = residuals.loc[:, ["date", "symbol", "residual"]].copy()
    residual["date"] = _daily_index(residual.date, "residuals")
    if residual.duplicated(["date", "symbol"]).any():
        raise ValueError("residuals has duplicate date/symbol keys")
    residual["residual"] = pd.to_numeric(residual.residual, errors="raise")
    residual = residual.loc[residual.date <= cutoff]
    styles = [c for c in factors.columns if c != "market" and not c.startswith("sector:")]
    required = {"asof_date", "symbol", "sector", *styles}
    if not required.issubset(latest_exposures.columns):
        raise ValueError(f"missing exposure columns: {sorted(required.difference(latest_exposures.columns))}")
    snapshot = latest_exposures.copy()
    snapshot["asof_date"] = _daily_index(snapshot.asof_date, "latest_exposures")
    if snapshot.asof_date.nunique() > 1:
        raise ValueError("latest_exposures must contain exactly one snapshot date")
    if (snapshot.asof_date > cutoff).any():
        raise ValueError("latest_exposures is dated after the risk cutoff")
    if snapshot.symbol.duplicated().any():
        raise ValueError("latest_exposures symbols must be unique")
    for column in styles:
        snapshot[column] = pd.to_numeric(snapshot[column], errors="raise")
    complete = np.isfinite(snapshot[styles].to_numpy(float)).all(axis=1)
    complete &= snapshot.sector.map(lambda s: isinstance(s, str) and bool(s.strip())).to_numpy()
    complete &= snapshot.symbol.map(lambda s: isinstance(s, str) and bool(s)).to_numpy()
    exposure_count = len(snapshot)
    snapshot = snapshot.loc[complete].set_index("symbol")
    active_sectors = sorted(snapshot.sector.unique())
    names = ["market", *[f"sector:{s}" for s in active_sectors], *styles]
    missing_sectors = set(names).difference(factors.columns)
    if missing_sectors:
        raise ValueError(f"current exposure sector has no historical factor column: {sorted(missing_sectors)}")
    window = factors[names].tail(lookback)
    valid = window.notna().all(axis=1).to_numpy()
    n_factor_obs = int(valid.sum())
    factor_covariance = pd.DataFrame(np.nan, index=names, columns=names)
    diagnostics = {
        "status": "ok", "asof_date": str(cutoff.date()),
        "exposure_asof_date": str(latest_exposures.asof_date.iloc[0]) if len(latest_exposures) else None,
        "estimate_type": "historical centered EWMA covariance; not out-of-sample validated",
        "daily_return_unit": "decimal", "factor_window_rows": len(window),
        "factor_complete_rows": n_factor_obs, "factor_rows_dropped_missing": int((~valid).sum()),
        "latest_factor_date": str(window.index[valid][-1].date()) if n_factor_obs else None,
        "min_history": min_history, "lookback": lookback, "half_life": half_life,
        "diagonal_shrinkage": shrinkage, "specific_window_sessions": 252,
        "specific_min_observations": 60, "exposure_rows": exposure_count,
        "dropped_incomplete_exposures": int(exposure_count - len(snapshot)),
        "dropped_missing_specific_risk": 0, "stocks_estimated": 0,
        "factor_covariance_min_eigenvalue": None, "stock_covariance_min_eigenvalue": None,
        "horizon_assumption": "constant daily covariance/exposures and uncorrelated daily shocks",
    }
    if n_factor_obs >= min_history:
        ages = np.arange(len(window) - 1, -1, -1, dtype=float)[valid]
        covariance = _weighted_covariance(window.to_numpy(float)[valid], ages, half_life)
        covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
        covariance = (covariance + covariance.T) / 2
        factor_covariance.loc[:, :] = covariance
        diagnostics["factor_covariance_min_eigenvalue"] = float(np.linalg.eigvalsh(covariance).min())
    else:
        diagnostics["status"] = "insufficient_factor_history"
    specific_sessions = factors.tail(252).index
    specific_ages = np.arange(len(specific_sessions) - 1, -1, -1, dtype=float)
    specific_records = []
    by_symbol = {symbol: frame.set_index("date").residual for symbol, frame in residual.groupby("symbol")}
    for symbol in snapshot.index:
        history = by_symbol.get(symbol, pd.Series(dtype=float)).reindex(specific_sessions)
        values = history.to_numpy(float)
        finite = np.isfinite(values)
        nobs = int(finite.sum())
        variance = np.nan
        if nobs >= 60:
            variance = float(_weighted_covariance(values[finite, None], specific_ages[finite], half_life)[0, 0])
        specific_records.append({"symbol": symbol, "daily_variance": variance, "n_obs": nobs,
                                 "latest_residual_date": specific_sessions[finite][-1] if nobs else pd.NaT})
    specific = pd.DataFrame(specific_records, columns=["symbol", "daily_variance", "n_obs", "latest_residual_date"])
    stock_risk = pd.DataFrame(columns=["symbol", "factor_daily_variance", "specific_daily_variance", "annual_vol", "vol_21d"])
    stock_covariance = pd.DataFrame(dtype=float)
    if len(snapshot) == 0:
        diagnostics["status"] = "no_complete_exposures"
    elif diagnostics["status"] == "ok":
        specific_by_symbol = specific.set_index("symbol")
        valid_specific = np.isfinite(specific_by_symbol.daily_variance) & (specific_by_symbol.daily_variance >= 0)
        diagnostics["dropped_missing_specific_risk"] = int((~valid_specific).sum())
        selected = snapshot.loc[valid_specific]
        if len(selected) == 0:
            diagnostics["status"] = "no_specific_risk_history"
        else:
            design = np.column_stack([np.ones(len(selected)),
                *[(selected.sector == s).to_numpy(float) for s in active_sectors],
                selected[styles].to_numpy(float)])
            common = design @ factor_covariance.to_numpy() @ design.T
            idiosyncratic = specific_by_symbol.loc[selected.index, "daily_variance"].to_numpy()
            covariance = common + np.diag(idiosyncratic)
            covariance = (covariance + covariance.T) / 2
            stock_covariance = pd.DataFrame(covariance, index=selected.index, columns=selected.index)
            total_daily = np.maximum(np.diag(covariance), 0)
            stock_risk = pd.DataFrame({"symbol": selected.index,
                "factor_daily_variance": np.diag(common), "specific_daily_variance": idiosyncratic,
                "annual_vol": np.sqrt(total_daily * 252), "vol_21d": np.sqrt(total_daily * 21)})
            diagnostics["stocks_estimated"] = len(selected)
            diagnostics["stock_covariance_min_eigenvalue"] = float(np.linalg.eigvalsh(covariance).min())
    return {"factor_covariance": factor_covariance, "specific_variance": specific,
            "stock_risk": stock_risk, "stock_covariance": stock_covariance, "diagnostics": diagnostics}
