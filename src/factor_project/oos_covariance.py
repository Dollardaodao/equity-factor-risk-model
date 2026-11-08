"""Monthly walk-forward forecasts of daily equity covariance.

Every forecast/weight uses data available at the origin close. Realized matrices
describe the following H daily observations, not a covariance estimated from a
single H-day cumulative return. Realized matrices are never inverted.
"""
from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULTS = {
    "origin_start": "2014-01-31", "origin_end": None, "holdout_start": "2022-01-01",
    "horizons": (21, 63), "factor_lookback": 756, "min_factor_history": 252,
    "factor_half_life": 63.0, "factor_shrinkage": 0.2,
    "specific_lookback": 252, "specific_half_life": 63.0, "min_specific_history": 126,
    "direct_lookback": 252, "direct_half_life": 63.0, "min_stocks": 35,
    "random_portfolios": 20, "random_seed": 1837, "random_concentration": 5.0,
    "minimum_variance_gross_limit": 2.0, "jitter_relative": 1e-6, "jitter_absolute": 1e-12,
    "include_factor_dof_adjusted": False, "store_forecasts": True,
    "storage_dtype": "float64", "return_basis": "excess_daily_decimal",
}


def _dates(values, label):
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.hasnans:
        raise ValueError(f"{label}: missing dates")
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _wide(frame, label):
    result = frame.copy()
    result.index = _dates(result.index, label)
    if result.index.duplicated().any() or result.columns.duplicated().any():
        raise ValueError(f"{label}: duplicate dates/columns")
    if any(not isinstance(c, str) or not c for c in result.columns):
        raise ValueError(f"{label}: columns must be nonempty strings")
    return result.sort_index().apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)


def _weights(length, half_life):
    return np.exp2(-np.arange(length - 1, -1, -1, dtype=float) / half_life)


def _ew_covariance(values, weights):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    correction = 1 - weights @ weights
    if correction <= np.finfo(float).eps:
        raise ValueError("Too few effective covariance observations")
    centered = values - weights @ values
    return (centered.T * weights) @ centered / correction


def _ew_variances(values, weights, minimum):
    finite = np.isfinite(values)
    nobs = finite.sum(axis=0)
    # Zero only masked arithmetic contributions; missing observations are not
    # imputed returns, and each column normalizes its observed weights.
    observed_weights = weights[:, None] * finite
    total = observed_weights.sum(axis=0)
    safe_values = np.where(finite, values, 0.0)
    mean = np.divide((observed_weights * safe_values).sum(axis=0), total,
                     out=np.zeros(values.shape[1]), where=total > 0)
    denominator = total - np.divide((observed_weights**2).sum(axis=0), total,
                                    out=np.zeros_like(total), where=total > 0)
    numerator = (observed_weights * (safe_values - mean)**2).sum(axis=0)
    variance = np.divide(numerator, denominator, out=np.full(values.shape[1], np.nan), where=denominator > 0)
    variance[nobs < minimum] = np.nan
    return variance, nobs


def _stabilize(covariance, settings, rank_hint=None):
    """Use a declared diagonal jitter only for loss/minimum-variance solves."""
    from scipy.linalg import cholesky
    covariance = (covariance + covariance.T) / 2
    n = len(covariance)
    scale = max(float(np.mean(np.diag(covariance))), 0.0)
    floor = max(settings["jitter_absolute"], settings["jitter_relative"] * scale)
    jitter = floor if rank_hint is not None and rank_hint < n else 0.0
    for _ in range(12):
        stabilized = covariance + np.eye(n) * jitter
        try:
            factor = cholesky(stabilized, lower=True, check_finite=False)
            rank = n if rank_hint is None and jitter == 0 else rank_hint
            if rank is None:
                rank = int(np.linalg.matrix_rank(covariance))
            return stabilized, factor, {"diagonal_jitter": float(jitter), "raw_rank": int(rank),
                "dimension": n, "jitter_relative_to_mean_variance": jitter / scale if scale else None,
                "raw_covariance_recovery": "stabilized covariance minus diagonal_jitter * identity"}
        except np.linalg.LinAlgError:
            jitter = floor if jitter == 0 else jitter * 10
    raise ValueError("Forecast covariance could not be stabilized")


def _common_portfolios(symbols, market_cap, settings):
    n = len(symbols)
    seed_material = f"{settings['random_seed']}\n" + "\n".join(symbols)
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    portfolios = {"equal_weight": np.full(n, 1 / n), "cap_weight": market_cap / market_cap.sum()}
    families = {"equal_weight": "equal_weight", "cap_weight": "cap_weight"}
    for i in range(settings["random_portfolios"]):
        name = f"random_longonly_{i:02d}"
        portfolios[name] = rng.dirichlet(np.full(n, settings["random_concentration"]))
        families[name] = "random_longonly"
        neutral = rng.normal(size=n)
        neutral -= neutral.mean()
        neutral /= np.abs(neutral).sum()
        name = f"dollar_neutral_random_{i:02d}"
        portfolios[name] = neutral
        families[name] = "dollar_neutral_random"
    return portfolios, families


def _minimum_variance(cholesky, gross_limit):
    from scipy.linalg import cho_solve
    n = len(cholesky)
    inv_one = cho_solve((cholesky, True), np.ones(n), check_finite=False)
    weights = inv_one / inv_one.sum()
    equal = np.full(n, 1 / n)
    if np.abs(weights).sum() <= gross_limit:
        return weights
    # Convex mixing with equal weight enforces a predeclared gross cap, without
    # an optimizer or any realized-return information.
    low, high = 0.0, 1.0
    for _ in range(45):
        alpha = (low + high) / 2
        candidate = equal + alpha * (weights - equal)
        if np.abs(candidate).sum() <= gross_limit:
            low = alpha
        else:
            high = alpha
    return equal + low * (weights - equal)


def _matrix_errors(forecast, realized):
    n = len(forecast)
    off = ~np.eye(n, dtype=bool)
    denominator = max(float(np.linalg.norm(realized)), 1e-30)
    forecast_std = np.sqrt(np.maximum(np.diag(forecast), 0))
    realized_std = np.sqrt(np.maximum(np.diag(realized), 0))
    divisor_f = np.outer(forecast_std, forecast_std)
    divisor_r = np.outer(realized_std, realized_std)
    valid = off & (divisor_f > 0) & (divisor_r > 0)
    corr_error = forecast[valid] / divisor_f[valid] - realized[valid] / divisor_r[valid]
    return {"relative_frobenius": float(np.linalg.norm(forecast - realized) / denominator),
            "offdiagonal_rmse": float(np.sqrt(np.mean((forecast[off] - realized[off])**2))),
            "offdiagonal_relative_frobenius": float(np.linalg.norm(forecast[off] - realized[off]) /
                                                   max(float(np.linalg.norm(realized[off])), 1e-30)),
            "correlation_rmse": float(np.sqrt(np.mean(corr_error**2))) if len(corr_error) else np.nan}


def walk_forward_covariance(returns: pd.DataFrame, factor_returns: pd.DataFrame,
                            residuals: pd.DataFrame, exposures: pd.DataFrame,
                            config: Mapping | None = None, progress_callback=None,
                            forecast_callback=None) -> dict:
    """Compare factor and direct covariance forecasts at monthly origins.

    All returns must use decimal DAILY units and the same basis as the fitted
    model (normally excess returns; caller subtracts RF). Forecast/eligibility
    histories end at the origin; exposures use the latest snapshot <= origin.
    All methods use the same ex-ante stocks: complete direct-return history,
    complete PIT exposures and sufficient historical factor residuals.

    Future missing prices never determine the forecast universe. An entire
    origin+horizon is explicitly skipped when its frozen universe has ANY
    missing next-H return or lacks the full H sessions. Coverage is recorded.

    Matrix-error scores use RAW forecast covariance and centered future sample
    covariance (ddof=1). Gaussian losses use stabilized forecast covariance and
    the uncentered future second moment, under a zero-mean-return convention.
    Min-variance allocations use that same stabilized covariance. Both realized
    matrices are saved; no realized covariance is inverted. Shrinkage, windows,
    jitter and portfolio gross caps are predeclared config, not holdout tuning.

    Returns DataFrames ``origins,matrix_scores,portfolio_scores,summary,
    portfolio_summary``, array mappings ``forecasts``, and ``metadata``. Mapping
    entries contain symbols, stabilized covariances, raw variance diagonals,
    frozen portfolio weights, realized matrices, and forecast metadata. Caller
    may export each entry to NPZ rather than huge stock-pair tables.

    ``forecast_callback(origin_iso, entry)`` is invoked once after all horizons
    for each forecastable origin. Set ``store_forecasts=False`` to stream these
    entries without retaining the full historical matrix collection in RAM.
    """
    from scipy.linalg import solve_triangular
    from sklearn.covariance import LedoitWolf
    supplied = dict(config or {})
    aliases = {"lookback": "factor_lookback", "min_history": "min_factor_history",
               "shrinkage": "factor_shrinkage", "return_lookback": "direct_lookback",
               "specific_min_history": "min_specific_history", "seed": "random_seed"}
    for old, new in aliases.items():
        if old in supplied and new not in supplied:
            supplied[new] = supplied[old]
    if "half_life" in supplied:
        for name in ["factor_half_life", "specific_half_life", "direct_half_life"]:
            supplied.setdefault(name, supplied["half_life"])
    settings = {**DEFAULTS, **supplied}
    integer_keys = ["factor_lookback", "min_factor_history", "specific_lookback", "min_specific_history",
                    "direct_lookback", "min_stocks", "random_portfolios"]
    if any(not isinstance(settings[k], int) or settings[k] < (0 if k == "random_portfolios" else 2) for k in integer_keys):
        raise ValueError("History/stocks counts must be integers >=2; random_portfolios must be >=0")
    if settings["factor_lookback"] < settings["min_factor_history"] or settings["specific_lookback"] < settings["min_specific_history"]:
        raise ValueError("Lookback must cover required history")
    for key in ["factor_half_life", "specific_half_life", "direct_half_life", "random_concentration", "jitter_relative", "jitter_absolute"]:
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= settings["factor_shrinkage"] <= 1 or settings["minimum_variance_gross_limit"] < 1:
        raise ValueError("Invalid shrinkage/gross limit")
    horizons = sorted(set(settings["horizons"]))
    if not horizons or any(not isinstance(h, int) or h < 2 for h in horizons):
        raise ValueError("horizons must contain integer session counts >=2")
    dtype = np.dtype(settings["storage_dtype"])
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("storage_dtype must be float32 or float64")
    data = _wide(returns, "returns")
    factors = _wide(factor_returns, "factor_returns")
    if "market" not in factors:
        raise ValueError("factor_returns must include market")
    styles = [c for c in factors if c != "market" and not c.startswith("sector:")]
    required = {"asof_date", "symbol", "sector", "market_cap", *styles}
    if not required.issubset(exposures.columns):
        raise ValueError(f"Missing exposures: {sorted(required.difference(exposures.columns))}")
    snapshots = exposures.copy()
    snapshots["asof_date"] = _dates(snapshots.asof_date, "exposures")
    if snapshots.duplicated(["asof_date", "symbol"]).any():
        raise ValueError("Duplicate exposure snapshot/symbol")
    for column in ["market_cap", *styles]:
        snapshots[column] = pd.to_numeric(snapshots[column], errors="raise")
    snapshot_map = {day: group.set_index("symbol") for day, group in snapshots.groupby("asof_date")}
    snapshot_dates = pd.DatetimeIndex(sorted(snapshot_map))
    if not {"date", "symbol", "residual"}.issubset(residuals.columns):
        raise ValueError("residuals requires date,symbol,residual")
    residual = residuals[["date", "symbol", "residual"]].copy()
    residual["date"] = _dates(residual.date, "residuals")
    if residual.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate residual date/symbol")
    residual["residual"] = pd.to_numeric(residual.residual, errors="raise")
    residual_wide = residual.pivot(index="date", columns="symbol", values="residual").replace([np.inf, -np.inf], np.nan)
    residual_count = residual_wide.notna().sum(axis=1)
    effective_params = factors.notna().sum(axis=1) - 1
    correction = residual_count / (residual_count - effective_params.reindex(residual_count.index))
    correction = correction.where(np.isfinite(correction) & (correction >= 1))
    calendar = data.index
    origin_series = pd.Series(calendar, index=calendar)
    origins = pd.DatetimeIndex(origin_series.groupby(calendar.to_period("M")).last().to_numpy())
    if settings["origin_start"] is not None:
        origins = origins[origins >= pd.Timestamp(settings["origin_start"])]
    if settings["origin_end"] is not None:
        origins = origins[origins <= pd.Timestamp(settings["origin_end"])]
    holdout = pd.Timestamp(settings["holdout_start"])
    origin_records, matrix_records, portfolio_records, stored = [], [], [], {}
    methods = ["factor", "sample_252_jitter", "ewma_252", "ledoit_wolf_252", "diagonal_ewma"]
    if settings["include_factor_dof_adjusted"]:
        methods.insert(1, "factor_dof_adjusted")
    previous_progress_year = None
    for origin_number, origin in enumerate(origins):
        if progress_callback is not None and (origin.year != previous_progress_year or origin_number == len(origins) - 1):
            progress_callback({"origin": str(origin.date()), "origin_number": origin_number + 1,
                               "total_origins": len(origins), "stage": "forecast"})
            previous_progress_year = origin.year
        base = {"origin": origin, "exposure_asof": pd.NaT, "n_snapshot": 0, "n_exante": 0,
                "n_factor_observations": 0}
        reason = None
        pos = int(calendar.get_loc(origin))
        past_sessions = calendar[:pos + 1]
        snapshot_pos = snapshot_dates.searchsorted(origin, side="right") - 1
        if snapshot_pos < 0:
            reason = "no_available_exposures"
        elif len(past_sessions) < settings["direct_lookback"]:
            reason = "insufficient_direct_history"
        if reason is not None:
            for h in horizons:
                origin_records.append({**base, "horizon": h, "status": reason})
            continue
        asof = snapshot_dates[snapshot_pos]
        snapshot = snapshot_map[asof]
        good = np.isfinite(snapshot[["market_cap", *styles]].to_numpy(float)).all(axis=1)
        good &= (snapshot.market_cap > 0).to_numpy()
        good &= snapshot.sector.map(lambda s: isinstance(s, str) and bool(s.strip())).to_numpy()
        good &= snapshot.index.isin(data.columns)
        snapshot = snapshot.loc[good].sort_index()
        base.update(exposure_asof=asof, n_snapshot=len(snapshot))
        direct_dates = past_sessions[-settings["direct_lookback"]:]
        direct = data.reindex(index=direct_dates, columns=snapshot.index)
        complete_prices = direct.notna().all(axis=0)
        snapshot = snapshot.loc[complete_prices]
        specific_dates = past_sessions[-settings["specific_lookback"]:]
        specific_history = residual_wide.reindex(index=specific_dates, columns=snapshot.index).to_numpy(float)
        specific, n_specific = _ew_variances(specific_history, _weights(len(specific_dates), settings["specific_half_life"]),
                                             settings["min_specific_history"])
        specific_adjusted = None
        good_specific = np.isfinite(specific) & (specific >= 0)
        if settings["include_factor_dof_adjusted"]:
            adjusted_history = specific_history * np.sqrt(correction.reindex(specific_dates).to_numpy())[:, None]
            specific_adjusted, _ = _ew_variances(adjusted_history, _weights(len(specific_dates), settings["specific_half_life"]),
                                                settings["min_specific_history"])
            good_specific &= np.isfinite(specific_adjusted) & (specific_adjusted >= 0)
        base.update(n_complete_price_history=len(snapshot), n_exante=int(good_specific.sum()))
        snapshot = snapshot.loc[good_specific]
        specific, n_specific = specific[good_specific], n_specific[good_specific]
        if specific_adjusted is not None:
            specific_adjusted = specific_adjusted[good_specific]
        if len(snapshot) < settings["min_stocks"]:
            for h in horizons:
                origin_records.append({**base, "horizon": h, "status": "insufficient_exante_stocks"})
            continue
        symbols = snapshot.index.tolist()
        sector_names = sorted(snapshot.sector.unique())
        factor_names = ["market", *[f"sector:{s}" for s in sector_names], *styles]
        missing = set(factor_names).difference(factors.columns)
        if missing:
            raise ValueError(f"Current sector absent from factor history: {sorted(missing)}")
        factor_dates = past_sessions[-settings["factor_lookback"]:]
        factor_history = factors.reindex(index=factor_dates, columns=factor_names)
        finite_factor = factor_history.notna().all(axis=1).to_numpy()
        base["n_factor_observations"] = int(finite_factor.sum())
        if finite_factor.sum() < settings["min_factor_history"]:
            for h in horizons:
                origin_records.append({**base, "horizon": h, "status": "insufficient_factor_history"})
            continue
        fcov = _ew_covariance(factor_history.to_numpy()[finite_factor],
            _weights(len(factor_dates), settings["factor_half_life"])[finite_factor])
        alpha = settings["factor_shrinkage"]
        fcov = (1 - alpha) * fcov + alpha * np.diag(np.diag(fcov))
        x = np.column_stack([np.ones(len(snapshot)),
            *[(snapshot.sector == s).to_numpy(float) for s in sector_names], snapshot[styles].to_numpy()])
        common = x @ fcov @ x.T
        history = data.reindex(index=direct_dates, columns=symbols).to_numpy()
        centered_history = history - history.mean(axis=0)
        sample_rank = int(np.linalg.matrix_rank(centered_history))
        sample_covariance = centered_history.T @ centered_history / (len(history) - 1)
        ewma = _ew_covariance(history, _weights(len(history), settings["direct_half_life"]))
        lw = LedoitWolf(assume_centered=False).fit(history)
        raw = {"factor": common + np.diag(specific), "sample_252_jitter": sample_covariance,
               "ewma_252": ewma, "ledoit_wolf_252": lw.covariance_, "diagonal_ewma": np.diag(np.diag(ewma))}
        if specific_adjusted is not None:
            raw["factor_dof_adjusted"] = common + np.diag(specific_adjusted)
        weights, families = _common_portfolios(symbols, snapshot.market_cap.to_numpy(), settings)
        cholesky, stabilized, method_metadata = {}, {}, {}
        for method, covariance in raw.items():
            rank_hint = sample_rank if method in {"sample_252_jitter", "ewma_252"} else None
            if method == "diagonal_ewma":
                rank_hint = int(np.count_nonzero(np.diag(covariance) > 0))
            stabilized[method], cholesky[method], method_metadata[method] = _stabilize(covariance, settings, rank_hint)
            name = f"minimum_variance::{method}"
            weights[name] = _minimum_variance(cholesky[method], settings["minimum_variance_gross_limit"])
            families[name] = "minimum_variance"
        forecast_metadata = {**base, "specific_observations_min": int(n_specific.min()),
            "specific_observations_median": float(np.median(n_specific)), "methods": method_metadata,
            "ledoit_wolf_shrinkage": float(lw.shrinkage_), "sample_covariance_raw_rank": sample_rank,
            "matrix_error_uses_raw_covariance": True, "loss_and_minvariance_use_stabilized_covariance": True}
        entry = {"symbols": symbols,
            "covariances": {m: c.astype(dtype) for m, c in stabilized.items()},
            "raw_variance_diagonals": {m: np.diag(c).astype(dtype) for m, c in raw.items()},
            "weights": {name: w.astype(dtype) for name, w in weights.items()},
            "realized": {}, "metadata": forecast_metadata}
        for horizon in horizons:
            future_dates = calendar[pos + 1:pos + 1 + horizon]
            log = {**base, "horizon": horizon, "n_future_sessions": len(future_dates),
                   "future_start": future_dates[0] if len(future_dates) else pd.NaT,
                   "future_end": future_dates[-1] if len(future_dates) else pd.NaT}
            if len(future_dates) != horizon:
                origin_records.append({**log, "status": "future_horizon_truncated"})
                continue
            future = data.reindex(index=future_dates, columns=symbols).to_numpy()
            future_complete = np.isfinite(future).all(axis=0)
            log.update(future_complete_stocks=int(future_complete.sum()),
                       future_asset_coverage=float(future_complete.mean()),
                       future_observation_coverage=float(np.isfinite(future).mean()))
            if not future_complete.all():
                origin_records.append({**log, "status": "future_missing_returns",
                    "future_missing_symbols": ",".join(np.array(symbols)[~future_complete])})
                continue
            split = "holdout" if origin >= holdout else "development" if future_dates[-1] < holdout else "split_boundary"
            log.update(status="scored", split=split)
            origin_records.append(log)
            centered = future - future.mean(axis=0)
            realized = centered.T @ centered / (horizon - 1)
            second_moment = future.T @ future / horizon
            entry["realized"][horizon] = {"centered_covariance": realized.astype(dtype),
                "second_moment": second_moment.astype(dtype), "start_date": future_dates[0],
                "end_date": future_dates[-1], "dates": future_dates}
            for method in methods:
                covariance, chol = stabilized[method], cholesky[method]
                whitened = solve_triangular(chol, future.T, lower=True, check_finite=False)
                logdet = float(2 * np.log(np.diag(chol)).sum())
                trace = float(np.sum(whitened**2) / horizon)
                matrix_records.append({"origin": origin, "future_end": future_dates[-1], "horizon": horizon,
                    "split": split, "method": method, "n_stocks": len(symbols),
                    **_matrix_errors(raw[method], realized),
                    "gaussian_qlike": logdet + trace,
                    "gaussian_qlike_per_asset": (logdet + trace) / len(symbols),
                    "gaussian_logdet": logdet, "gaussian_trace": trace,
                    "diagonal_jitter": method_metadata[method]["diagonal_jitter"],
                    "raw_rank": method_metadata[method]["raw_rank"]})
                names = [name for name in weights if not name.startswith("minimum_variance::")]
                names.append(f"minimum_variance::{method}")
                wmatrix = np.vstack([weights[name] for name in names])
                predicted = np.sum((wmatrix @ covariance) * wmatrix, axis=1)
                portfolio_returns = future @ wmatrix.T
                realized_var = portfolio_returns.var(axis=0, ddof=1)
                realized_second = np.mean(portfolio_returns**2, axis=0)
                for j, name in enumerate(names):
                    ratio = realized_second[j] / predicted[j]
                    portfolio_records.append({"origin": origin, "future_end": future_dates[-1],
                        "horizon": horizon, "split": split, "method": method,
                        "portfolio": name, "portfolio_family": families[name],
                        "weight_scope": "method_specific" if name.startswith("minimum_variance::") else "common",
                        "n_stocks": len(symbols), "net_weight": float(wmatrix[j].sum()),
                        "gross_weight": float(np.abs(wmatrix[j]).sum()),
                        "forecast_daily_variance": float(predicted[j]),
                        "realized_daily_variance": float(realized_var[j]),
                        "realized_daily_second_moment": float(realized_second[j]),
                        "calibration_ratio": float(ratio),
                        "centered_variance_calibration_ratio": float(realized_var[j] / predicted[j]),
                        "portfolio_gaussian_qlike": float(np.log(predicted[j]) + realized_second[j] / predicted[j])})
        origin_iso = str(origin.date())
        if forecast_callback is not None:
            forecast_callback(origin_iso, entry)
        if settings["store_forecasts"]:
            stored[origin_iso] = entry
    origin_frame = pd.DataFrame(origin_records)
    matrix_frame, portfolio_frame = pd.DataFrame(matrix_records), pd.DataFrame(portfolio_records)
    summary_records, portfolio_summary_records = [], []
    if len(matrix_frame):
        scored = matrix_frame.loc[matrix_frame.split != "split_boundary"]
        for key, group in scored.groupby(["split", "horizon", "method"]):
            summary_records.append({"split": key[0], "horizon": key[1], "method": key[2],
                "n_origins": len(group), "median_stocks": float(group.n_stocks.median()),
                **{f"mean_{column}": float(group[column].mean()) for column in
                   ["relative_frobenius", "offdiagonal_rmse", "offdiagonal_relative_frobenius",
                    "correlation_rmse", "gaussian_qlike_per_asset"]}})
        scored = portfolio_frame.loc[portfolio_frame.split != "split_boundary"]
        for key, group in scored.groupby(["split", "horizon", "method", "portfolio_family"]):
            by_origin = group.groupby("origin")[["realized_daily_second_moment", "forecast_daily_variance"]].sum()
            origin_ratios = by_origin.realized_daily_second_moment / by_origin.forecast_daily_variance
            portfolio_summary_records.append({"split": key[0], "horizon": key[1], "method": key[2],
                "portfolio_family": key[3], "n_origins": int(group.origin.nunique()),
                "n_portfolio_observations": len(group),
                "aggregate_calibration_ratio": float(group.realized_daily_second_moment.sum() / group.forecast_daily_variance.sum()),
                "aggregate_centered_variance_ratio": float(group.realized_daily_variance.sum() / group.forecast_daily_variance.sum()),
                "median_origin_calibration_ratio": float(origin_ratios.median()),
                "p10_origin_calibration_ratio": float(origin_ratios.quantile(.1)),
                "p90_origin_calibration_ratio": float(origin_ratios.quantile(.9)),
                "mean_portfolio_gaussian_qlike": float(group.portfolio_gaussian_qlike.mean()),
                "mean_forecast_daily_variance": float(group.forecast_daily_variance.mean()),
                "mean_realized_daily_second_moment": float(group.realized_daily_second_moment.mean()),
                "mean_realized_daily_variance": float(group.realized_daily_variance.mean())})
    metadata = {"config": settings, "methods": methods, "primary_horizon": 21,
        "forecast_unit": "daily return covariance, decimal squared",
        "realized_matrix": "centered sample covariance of next H DAILY returns, ddof=1",
        "gaussian_loss_target": "uncentered next-H daily second moment, zero conditional mean",
        "gaussian_loss_direction": "lower is better; no realized matrix inversion",
        "origin_rule": "monthly last observed return session, forecast at close; exposures dated<=origin",
        "eligibility": "same ex-ante universe for every method; complete historical direct-return window, finite PIT exposure and adequate specific history",
        "future_missing_rule": "skip whole origin+horizon; never reselect stocks using future availability",
        "development_rule": "future horizon end strictly before holdout_start",
        "holdout_rule": "origin on/after holdout_start; crossing windows labeled split_boundary",
        "overlap": "Monthly 21-session windows can overlap; 63-session windows overlap substantially. No IID standard errors or significance claims.",
        "portfolio_aggregation": "EW and cap-weight reported separately; random portfolios grouped by family rather than dominating a pooled score",
        "portfolio_calibration": "primary ratio sums zero-mean realized second moments / sums forecast variance; centered-variance ratio also reported",
        "portfolio_weight_interpretation": "frozen weight vector applied to each daily return (fixed-weight/daily-rebalanced diagnostic), not drifting buy-and-hold wealth",
        "minimum_variance": "unconstrained fully invested optimum, mixed with EW until configured gross cap; covariance and weights use origin data only",
        "matrix_jitter": "raw forecast used for matrix errors; stabilized covariance used for Gaussian loss, portfolio variance and minimum-variance weights; recover raw by subtracting recorded jitter*I",
        "diagonal_shrinkage": "factor covariance fixed diagonal shrink; Ledoit-Wolf intensity estimated from past direct returns only",
        "factor_specific_dof_variant": "optional fixed per-date n/(n-p) residual scaling; default disabled",
        "claims": "Historical walk-forward forecast assessment; observational coverage conditioning remains material. No return-alpha or trading-cost claims."}
    return {"origins": origin_frame, "matrix_scores": matrix_frame, "portfolio_scores": portfolio_frame,
            "summary": pd.DataFrame(summary_records), "portfolio_summary": pd.DataFrame(portfolio_summary_records),
            "forecasts": stored, "metadata": metadata}
