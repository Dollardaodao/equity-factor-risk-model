"""Independent, bounded-memory reconstruction of saved covariance forecasts.

Run only after the expanded pipeline finishes. This module does not invoke the
forecasting engine or its covariance/portfolio helpers. It independently uses
past price ratios, numpy.cov, SEC-dated exposures and saved residuals to rebuild
first, March-2020 and late forecast/realization pairs, one origin at a time.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_csv(path, predicate, **kwargs):
    pieces = [part.loc[predicate(part)] for part in pd.read_csv(path, chunksize=5000, **kwargs)]
    return pd.concat(pieces, ignore_index=True)


def _age_weights(dates, half_life):
    return np.exp(-np.log(2) * np.arange(len(dates) - 1, -1, -1) / half_life)


def _specific(history, weights, minimum):
    result = []
    for _, column in history.items():
        valid = column.notna().to_numpy()
        result.append(float(np.cov(column.to_numpy()[valid], aweights=weights[valid], ddof=1))
                      if valid.sum() >= minimum else np.nan)
    return np.asarray(result)


def _excess_returns(root, config, universe):
    benchmark = pd.read_csv(root / config["benchmark_file"], parse_dates=["date"]).set_index("date")
    benchmark = benchmark.loc[config["price_start"]:config["model_end"]]
    market_audit_path = root / "data/market_audit.json"
    market_audit = json.loads(market_audit_path.read_text()) if market_audit_path.is_file() else {}
    exclusions = {rule["symbol"] for rule in market_audit.get("model_identity_screen", {}).get("rules", [])
                  if rule["action"] == "exclude" and rule["model_rows"] == 0}
    prices = {}
    for symbol in universe.symbol:
        path = root / config["market_cache"] / f"{symbol}.csv"
        if symbol in exclusions:
            if path.exists():
                raise ValueError(f"Excluded symbol unexpectedly has a model price file: {symbol}")
            continue
        frame = pd.read_csv(path, usecols=["date", "adj_close"], parse_dates=["date"])
        if frame.date.duplicated().any():
            raise ValueError(f"Duplicate source price dates: {symbol}")
        prices[symbol] = frame.set_index("date").adj_close.reindex(benchmark.index)
    wide = pd.DataFrame(prices)
    # A missing session and its following return remain missing. There is no
    # forward filling and no use of the cache's precomputed return column.
    returns = (wide / wide.shift(1) - 1).loc[config["model_start"]:config["model_end"]]
    return returns.subtract(benchmark.RF, axis=0).reindex(returns.index), benchmark


def _max_filing(value):
    dates = []
    if isinstance(value, dict):
        if value.get("filed"):
            dates.append(pd.Timestamp(value["filed"]))
        dates.extend(_max_filing(child) for child in value.values() if isinstance(child, (dict, list)))
    elif isinstance(value, list):
        dates.extend(_max_filing(child) for child in value)
    return max(dates, default=pd.Timestamp.min)


def _financial_date_audit(path):
    checked, violations = 0, 0
    cached = lru_cache(maxsize=4096)(lambda raw: _max_filing(json.loads(raw)))
    for part in pd.read_csv(path, chunksize=1000, parse_dates=["asof_date"]):
        for field in [name for name in part if name.endswith("_provenance")]:
            valid = part[field].notna()
            observed = part.loc[valid, field].map(cached)
            violations += int((pd.to_datetime(observed) >= part.loc[valid, "asof_date"]).sum())
            checked += int(valid.sum())
    return {"provenance_cells_checked": checked, "filed_not_strictly_before_asof": violations}


def audit_export_scores(root: Path, origins) -> dict:
    """Verify saved descriptive scores, holdout labels and aggregate calibration."""
    output = Path(root) / "output"
    metadata = json.loads((output / "evaluation_metadata.json").read_text())
    cutoff = pd.Timestamp(metadata["config"]["holdout_start"])
    scores = pd.read_csv(output / "matrix_scores.csv", parse_dates=["origin", "future_end"])
    expected_split = np.where(scores.origin >= cutoff, "holdout", np.where(scores.future_end < cutoff, "development", "split_boundary"))
    failures = []
    split_errors = int((scores.split != expected_split).sum())
    if split_errors:
        failures.append("matrix_score_split_label_violations")
    families = scores.groupby(["origin", "horizon"])
    comparability_errors = int(((families.method.nunique() != len(metadata["methods"])) |
                                (families.n_stocks.nunique() != 1) |
                                (families.size() != len(metadata["methods"])) ).sum())
    if comparability_errors:
        failures.append("different_methods_or_stock_counts_for_same_origin_horizon")
    score_keys = ["split", "horizon", "method"]
    expected = scores.loc[scores.split != "split_boundary"].groupby(score_keys).gaussian_qlike_per_asset.agg(["size", "mean"])
    summary = pd.read_csv(output / "summary.csv").set_index(score_keys).sort_index()
    if not expected.index.equals(summary.index) or not np.array_equal(expected["size"], summary.n_origins) or not np.allclose(expected["mean"], summary.mean_gaussian_qlike_per_asset):
        failures.append("summary_does_not_match_nonboundary_matrix_scores")
    portfolio_keys = [*score_keys, "portfolio_family"]
    value_cols = ["realized_daily_second_moment", "forecast_daily_variance"]
    portfolios = pd.read_csv(output / "portfolio_scores.csv", usecols=portfolio_keys + value_cols)
    sums = portfolios.loc[portfolios.split != "split_boundary"].groupby(portfolio_keys)[value_cols].sum()
    expected_ratio = sums.realized_daily_second_moment / sums.forecast_daily_variance
    reported_ratio = pd.read_csv(output / "portfolio_summary.csv").set_index(portfolio_keys).aggregate_calibration_ratio.sort_index()
    if not expected_ratio.index.equals(reported_ratio.index) or not np.allclose(expected_ratio, reported_ratio, rtol=1e-10, atol=1e-12):
        failures.append("portfolio_summary_calibration_ratio_mismatch")

    fields = ["relative_frobenius", "offdiagonal_rmse", "offdiagonal_relative_frobenius", "correlation_rmse"]
    errors = dict.fromkeys(fields, 0.0)
    checked = 0
    for origin in pd.to_datetime(list(origins)):
        name = f"{origin:%Y-%m-%d}"
        path = output / "matrices" / f"{name}.json"
        matrix_meta = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as values:
            upper = np.triu_indices(len(values["symbols"]), k=1)
            for _, row in scores.loc[scores.origin == origin].iterrows():
                sample = values[f"realized_{row.horizon}__centered_covariance"]
                forecast = values[f"forecast__{row.method}"] - np.eye(len(sample)) * matrix_meta["metadata"]["methods"][row.method]["diagonal_jitter"]
                delta = forecast - sample
                sd_f = np.sqrt(np.maximum(np.diag(forecast), 0))
                sd_s = np.sqrt(np.maximum(np.diag(sample), 0))
                f_den, s_den = np.outer(sd_f, sd_f)[upper], np.outer(sd_s, sd_s)[upper]
                use = (f_den > 0) & (s_den > 0)
                corr_delta = forecast[upper][use] / f_den[use] - sample[upper][use] / s_den[use]
                expected_values = [np.linalg.norm(delta) / max(np.linalg.norm(sample), 1e-30),
                                   np.sqrt(np.mean(delta[upper]**2)),
                                   np.linalg.norm(delta[upper]) / max(np.linalg.norm(sample[upper]), 1e-30),
                                   np.sqrt(np.mean(corr_delta**2))]
                checked += 1
                for field, value in zip(fields, expected_values):
                    errors[field] = max(errors[field], float(abs(row[field] - value)))
                    if not np.isclose(row[field], value, rtol=3e-7, atol=1e-12):
                        failures.append(f"{name}:{row.horizon}:{row.method}:{field}")
    return {"matrix_rows_checked_for_split_and_comparability": len(scores), "split_label_violations": split_errors,
            "method_comparability_violations": comparability_errors, "boundary_score_rows_excluded": int((scores.split == "split_boundary").sum()),
            "summary_rows_checked": len(summary), "portfolio_calibration_summary_rows_checked": len(reported_ratio),
            "descriptive_score_rows_rebuilt_from_S": checked, "descriptive_score_max_absolute_errors": errors,
            "failures": failures}


def audit_oos(root: Path, origins=None) -> dict:
    """Rebuild real exported pairs and fail after writing a reviewable audit.

    Forecasts and realized proxies are DAILY covariance units. For a horizon H,
    M=R'R/H and S=(R-mean(R))'(R-mean(R))/(H-1); S has rank at most H-1.
    All reconstruction windows stop at the origin and use the stored symbol
    order. No data/network mutation or forecast retuning is performed.
    """
    from sklearn.covariance import LedoitWolf
    from threadpoolctl import threadpool_limits
    root = Path(root).resolve()
    output = root / "output"
    config = json.loads((root / "config.json").read_text())
    evaluation = json.loads((output / "evaluation_metadata.json").read_text())
    settings = evaluation["config"]
    universe = pd.read_csv(root / config["universe_file"])
    index = pd.read_csv(output / "matrix_index.csv", parse_dates=["origin"])
    paired = index.loc[index.realized_horizons.map(lambda value: {"21", "63"}.issubset(str(value).split(",")))].sort_values("origin")
    if paired.empty:
        raise ValueError("No fully scored 21/63-session pair is available")
    if origins is None:
        covid = (paired.origin - pd.Timestamp("2020-03-31")).abs().idxmin()
        origins = sorted(set([paired.origin.iloc[0], paired.loc[covid, "origin"], paired.origin.iloc[-1]]))
    else:
        origins = sorted(set(pd.to_datetime(origins)))
    factors = pd.read_csv(output / "factor_returns.csv", index_col="date", parse_dates=True)
    styles = [name for name in factors if name != "market" and not name.startswith("sector:")]
    returns, _ = _excess_returns(root, config, universe)
    calendar = returns.index
    exposure_columns = ["asof_date", "symbol", "sector", "market_cap", *styles,
                        "raw_close", "shares", "shares_filed", "shares_at_asof_estimate"]
    exposures = pd.read_csv(root / "data/processed/monthly_exposures.csv.gz", usecols=exposure_columns,
                            parse_dates=["asof_date", "shares_filed"])
    needed_dates = set()
    for origin in origins:
        past = calendar[calendar <= origin]
        needed_dates.update(past[-settings["specific_lookback"]:])
    residual = _selected_csv(output / "residuals.csv.gz", lambda part: part.date.isin(needed_dates),
                              usecols=["date", "symbol", "residual"], parse_dates=["date"])
    residual = residual.pivot(index="date", columns="symbol", values="residual")
    matrix_scores = _selected_csv(output / "matrix_scores.csv", lambda part: part.origin.isin(origins), parse_dates=["origin", "future_end"])
    portfolio_scores = _selected_csv(output / "portfolio_scores.csv", lambda part: part.origin.isin(origins), parse_dates=["origin", "future_end"])
    report = {"status": "passed", "origins": {}, "failures": [],
              "scope": "Independent source reconstruction at selected real origins; not certification of a prospective investment universe.",
              "checks": {"financial_dates": _financial_date_audit(root / "data/processed/point_in_time_fundamentals.csv.gz")},
              "source_sha256": {str(path.relative_to(root)): _sha(path) for path in
                                   [root / "config.json", output / "model_manifest.json", output / "evaluation_metadata.json",
                                    output / "factor_returns.csv", output / "residuals.csv.gz",
                                    root / "data/processed/monthly_exposures.csv.gz",
                                    root / "data/processed/point_in_time_fundamentals.csv.gz"]},
              "interpretation": [
                  "M and S are average daily proxies over the window, not covariance of cumulative horizon returns.",
                  "Calibration ratios compare second moment to variance; near 1 does not establish all pairwise correlations.",
                  "Common portfolios compare forecasts on the same weights. Minimum-variance portfolios are method-specific allocations.",
                  "Current holdings selection, static sectors and revised Yahoo adjustments remain retrospective limitations.",
                  "Relative Frobenius/correlation errors transform noisy targets; Gaussian score is the direct covariance scoring comparison.",
              ]}
    diagnostics = pd.read_csv(output / "regression_diagnostics.csv", usecols=["date", "status", "asof_date"], parse_dates=["date", "asof_date"])
    fitted = diagnostics.loc[diagnostics.status == "ok"]
    report["checks"]["regression_snapshots_not_strictly_prior"] = int((fitted.asof_date >= fitted.date).sum())

    def close(actual, expected, label, detail, rtol=3e-7, atol=1e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected))) if actual.size and actual.shape == expected.shape else None
        error = error if error is None or np.isfinite(error) else None
        detail[label] = {"max_absolute_error": error, "passed": bool(actual.shape == expected.shape and np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))}
        if not detail[label]["passed"]:
            report["failures"].append(label)

    with threadpool_limits(limits=1):
        for origin in origins:
            name = f"{origin:%Y-%m-%d}"
            detail = report["origins"][name] = {}
            try:
                metadata_path = output / "matrices" / f"{name}.json"
                metadata = json.loads(metadata_path.read_text())
                matrix_path = metadata_path.with_suffix(".npz")
                if _sha(matrix_path) != metadata["npz_sha256"]:
                    raise ValueError("NPZ hash mismatch")
                report["source_sha256"][str(matrix_path.relative_to(root))] = metadata["npz_sha256"]
                past = calendar[calendar <= origin]
                direct_dates = past[-settings["direct_lookback"]:]
                specific_dates = past[-settings["specific_lookback"]:]
                factor_dates = past[-settings["factor_lookback"]:]
                asof = exposures.loc[exposures.asof_date <= origin, "asof_date"].max()
                snapshot = exposures.loc[exposures.asof_date == asof].set_index("symbol").sort_index()
                snapshot = snapshot.loc[np.isfinite(snapshot[["market_cap", *styles]]).all(axis=1) & (snapshot.market_cap > 0)]
                snapshot = snapshot.loc[snapshot.sector.notna() & snapshot.index.isin(returns.columns)]
                direct = returns.reindex(index=direct_dates, columns=snapshot.index)
                snapshot = snapshot.loc[direct.notna().all(axis=0)]
                specific_history = residual.reindex(index=specific_dates, columns=snapshot.index)
                specific = _specific(specific_history, _age_weights(specific_dates, settings["specific_half_life"]), settings["min_specific_history"])
                good = np.isfinite(specific) & (specific >= 0)
                snapshot, specific = snapshot.loc[good], specific[good]
                with np.load(matrix_path, allow_pickle=False) as arrays:
                    symbols = arrays["symbols"].astype(str).tolist()
                    detail["frozen_universe"] = {"passed": symbols == snapshot.index.tolist(), "n_stocks": len(symbols)}
                    if symbols != snapshot.index.tolist():
                        report["failures"].append(name + ":frozen_universe")
                        continue
                    equal = np.full(len(symbols), 1 / len(symbols))
                    close(arrays["weights__equal_weight"], equal, name + ":equal_weights", detail)
                    close(arrays["weights__cap_weight"], snapshot.market_cap.to_numpy() / snapshot.market_cap.sum(), name + ":cap_weights", detail)
                    sectors = sorted(snapshot.sector.unique())
                    factor_names = ["market", *[f"sector:{sector}" for sector in sectors], *styles]
                    fh = factors.reindex(index=factor_dates, columns=factor_names)
                    valid = fh.notna().all(axis=1)
                    fw = _age_weights(factor_dates, settings["factor_half_life"])[valid]
                    factor_cov = np.cov(fh.loc[valid].to_numpy(), rowvar=False, aweights=fw, ddof=1)
                    alpha = settings["factor_shrinkage"]
                    factor_cov = (1 - alpha) * factor_cov + alpha * np.diag(np.diag(factor_cov))
                    loadings = np.column_stack([np.ones(len(snapshot)), *[(snapshot.sector == sector).to_numpy(float) for sector in sectors], snapshot[styles].to_numpy()])
                    history = returns.loc[direct_dates, symbols].to_numpy()
                    ewma = np.cov(history, rowvar=False, aweights=_age_weights(direct_dates, settings["direct_half_life"]), ddof=1)
                    raw = {"factor": loadings @ factor_cov @ loadings.T + np.diag(specific),
                           "sample_252_jitter": np.cov(history, rowvar=False, ddof=1), "ewma_252": ewma,
                           "ledoit_wolf_252": LedoitWolf(assume_centered=False).fit(history).covariance_,
                           "diagonal_ewma": np.diag(np.diag(ewma))}
                    if set(metadata["methods"]) != set(raw):
                        raise ValueError("Audit must explicitly support every configured forecast method")
                    detail.update(exposure_asof=str(asof.date()), past_direct_start=str(direct_dates.min().date()),
                                  past_direct_end=str(direct_dates.max().date()), past_factor_end=str(fh.loc[valid].index.max().date()))
                    if asof > origin or direct_dates.max() > origin or fh.loc[valid].index.max() > origin:
                        raise ValueError("A reconstruction window contains future data")
                    for method in metadata["methods"]:
                        saved = arrays[f"forecast__{method}"]
                        if not np.isfinite(saved).all():
                            raise ValueError("Saved forecast contains nonfinite values")
                        jitter = metadata["metadata"]["methods"][method]["diagonal_jitter"]
                        close(saved, raw[method] + np.eye(len(symbols)) * jitter, name + ":forecast:" + method, detail)
                        np.linalg.cholesky(saved)
                        saved_minimum = arrays[f"weights__minimum_variance::{method}"]
                        unrestricted = np.linalg.solve(saved, np.ones(len(symbols)))
                        unrestricted /= unrestricted.sum()
                        if np.abs(unrestricted).sum() <= settings["minimum_variance_gross_limit"]:
                            close(saved_minimum, unrestricted, name + ":minimum_variance:" + method, detail, rtol=1e-5, atol=1e-7)
                        else:
                            direction = unrestricted - equal
                            mixing = float(np.dot(saved_minimum - equal, direction) / np.dot(direction, direction))
                            close(saved_minimum, equal + mixing * direction, name + ":minimum_variance_mix:" + method, detail, rtol=1e-5, atol=1e-7)
                            if not -1e-8 <= mixing <= 1 + 1e-8:
                                raise ValueError("Minimum-variance weights are not an ex-ante EW/optimum mixture")
                        if abs(saved_minimum.sum() - 1) > 1e-7 or np.abs(saved_minimum).sum() > settings["minimum_variance_gross_limit"] + 1e-7:
                            raise ValueError("Minimum-variance allocation violates net/gross weight rules")
                        for horizon, realized_meta in metadata["realized"].items():
                            h = int(horizon)
                            dates = calendar[calendar > origin][:h]
                            if len(dates) != h or dates.tolist() != pd.DatetimeIndex(pd.to_datetime(realized_meta["dates"])).tolist():
                                raise ValueError(f"Future dates differ from exactly next {h} sessions")
                            future = returns.loc[dates, symbols].to_numpy()
                            if not np.isfinite(future).all():
                                raise ValueError("Saved realization has incomplete frozen-universe returns")
                            moment, sample = future.T @ future / h, np.cov(future, rowvar=False, ddof=1)
                            prefix = f"{name}:{h}:"
                            if method == metadata["methods"][0]:
                                close(arrays[f"realized_{h}__second_moment"], moment, prefix + "M", detail)
                                close(arrays[f"realized_{h}__centered_covariance"], sample, prefix + "S", detail)
                                rank = int(np.linalg.matrix_rank(sample, hermitian=True))
                                detail[prefix + "centered_rank"] = {"observed": rank, "upper_bound": h - 1}
                                if rank > h - 1:
                                    raise ValueError("Centered realized covariance violates expected rank bound")
                                if pd.Timestamp(realized_meta["start_date"]) != dates[0] or pd.Timestamp(realized_meta["end_date"]) != dates[-1]:
                                    raise ValueError("Future boundary metadata mismatch")
                            score = matrix_scores.loc[(matrix_scores.origin == origin) & (matrix_scores.horizon == h) & (matrix_scores.method == method)]
                            if len(score) != 1:
                                raise ValueError("Missing/duplicate matrix score")
                            sign, logdet = np.linalg.slogdet(saved)
                            if sign <= 0:
                                raise ValueError("Forecast determinant is not positive")
                            trace = float(np.sum(future.T * np.linalg.solve(saved, future.T)) / h)
                            close(score.gaussian_qlike.iloc[0], logdet + trace, prefix + method + ":gaussian", detail, atol=1e-6)
                            rows = portfolio_scores.loc[(portfolio_scores.origin == origin) & (portfolio_scores.horizon == h) & (portfolio_scores.method == method)]
                            weights = np.vstack([arrays[f"weights__{portfolio}"] for portfolio in rows.portfolio])
                            predicted = np.einsum("ij,jk,ik->i", weights, saved, weights)
                            future_portfolios = future @ weights.T
                            second = np.mean(future_portfolios**2, axis=0)
                            close(rows.forecast_daily_variance, predicted, prefix + method + ":portfolio_prediction", detail)
                            close(rows.realized_daily_second_moment, second, prefix + method + ":portfolio_M", detail)
                            close(rows.realized_daily_variance, np.var(future_portfolios, axis=0, ddof=1), prefix + method + ":portfolio_S", detail)
                            close(rows.calibration_ratio, second / predicted, prefix + method + ":portfolio_ratio", detail)
                            close(rows.portfolio_gaussian_qlike, np.log(predicted) + second / predicted, prefix + method + ":portfolio_gaussian", detail, atol=1e-7)
                            detail[prefix + method + ":portfolios_checked"] = len(rows)
                    split_examples = {}
                    for symbol in ["NVDA", "GOOG", "GOOGL"]:
                        selected = snapshot.loc[[symbol]] if symbol in snapshot.index else snapshot.iloc[:0]
                        if selected.empty:
                            continue
                        price = pd.read_csv(root / config["market_cache"] / f"{symbol}.csv", parse_dates=["date"]).set_index("date")
                        share = selected.iloc[0]
                        factor = price.loc[(price.index > share.shares_filed) & (price.index <= asof), "split"].prod()
                        close(share.shares_at_asof_estimate, share.shares * factor, name + ":split_shares:" + symbol, detail)
                        close(share.market_cap, share.raw_close * share.shares_at_asof_estimate, name + ":split_cap:" + symbol, detail)
                        close(share.raw_close, price.loc[asof, "split_adjusted_close"] * price.loc[asof, "future_split_factor"], name + ":raw_price_units:" + symbol, detail)
                        split_examples[symbol] = {"asof": str(asof.date()), "share_multiplier_through_asof": float(factor),
                                                  "raw_close": float(share.raw_close), "shares": float(share.shares),
                                                  "shares_at_asof": float(share.shares_at_asof_estimate), "market_cap": float(share.market_cap)}
                    detail["split_unit_examples"] = split_examples
            except Exception as error:
                report["failures"].append(f"{name}: {type(error).__name__}: {error}")
    if report["checks"]["financial_dates"]["filed_not_strictly_before_asof"]:
        report["failures"].append("financial_provenance_date_violation")
    if report["checks"]["regression_snapshots_not_strictly_prior"]:
        report["failures"].append("regression_snapshot_date_violation")
    with threadpool_limits(limits=1):
        report["checks"]["export_scores"] = audit_export_scores(root, origins)
    report["failures"].extend(report["checks"]["export_scores"]["failures"])
    report["status"] = "passed" if not report["failures"] else "failed"
    target = output / "oos_integrity_audit.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise RuntimeError(f"OOS integrity audit failed; see {target}: {report['failures']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "expanded")
    args = parser.parse_args()
    result = audit_oos(args.root)
    print(json.dumps({"status": result["status"], "origins": list(result["origins"]), "failures": result["failures"]}, indent=2))
