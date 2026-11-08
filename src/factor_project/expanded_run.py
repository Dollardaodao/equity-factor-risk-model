"""Long-history, size-stratified covariance forecasting experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .descriptors import STYLES, build_exposures, session_returns
from .fundamentals import build_fundamentals
from .model import fit_factor_model

BASE = Path(__file__).resolve().parent
ROOT = Path("research")

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    if isinstance(value, (np.ndarray, pd.Index)):
        return _clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_fingerprints(root, config, universe):
    """Invalidate cached factors on RF revisions or SEC cache refreshes."""
    sec = root / config["sec_cache"]
    facts = []
    for cik in sorted(universe.cik.astype(str).str.zfill(10).unique()):
        for suffix in [".json", "_submissions.json"]:
            path = sec / f"CIK{cik}{suffix}"
            state = path.stat() if path.exists() else None
            facts.append((path.name, state.st_size if state else None, state.st_mtime_ns if state else None))
    return {"benchmark_daily_returns": _sha(root / config["benchmark_file"]),
            "sec_cache_file_state": hashlib.sha256(json.dumps(facts).encode()).hexdigest()}


def prepare_model(root: Path = ROOT, reuse: bool = False) -> dict:
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text())
    output, processed = root / "output", root / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(root / config["universe_file"], dtype={"cik": str})
    benchmark = pd.read_csv(root / config["benchmark_file"], parse_dates=["date"]).set_index("date")
    benchmark = benchmark.loc[config["price_start"]:config["model_end"]]
    dates = benchmark.groupby(benchmark.index.to_period("M")).tail(1).index
    dates = dates[dates >= pd.Timestamp(config["model_start"]) - pd.offsets.MonthEnd(1)]
    fingerprints = {str(p.relative_to(root)): _sha(p) for p in
                    [root / "config.json", root / "universe.csv", root / "data/market_audit.json"]}
    fingerprints.update({f"code/{name}": _sha(BASE / name) for name in
                         ["descriptors.py", "fundamentals.py", "model.py"]})
    fingerprints.update(source_fingerprints(root, config, universe))
    market_columns = {"date", "symbol", "raw_close", "adj_close", "dividend", "split", "volume", "price_source"}
    def market_inputs():
        frames = []
        for symbol in universe.symbol:
            path = root / config["market_cache"] / f"{symbol}.csv"
            if path.exists():
                frames.append(pd.read_csv(path, parse_dates=["date"], usecols=lambda c: c in market_columns))
        result = pd.concat(frames, ignore_index=True)
        result["symbol"] = result.symbol.astype("category")
        result["price_source"] = result.price_source.astype("category")
        return result
    manifest_path = output / "model_manifest.json"
    if reuse:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprints") != fingerprints:
            raise ValueError("Cached model inputs/code changed; rerun without --reuse-model")
        exposures = pd.read_csv(processed / "monthly_exposures.csv.gz", parse_dates=["asof_date"])
        coverage = pd.read_csv(processed / "monthly_coverage.csv", parse_dates=["asof_date"])
        factors = pd.read_csv(output / "factor_returns.csv", index_col="date", parse_dates=True)
        residuals = pd.read_csv(output / "residuals.csv.gz", parse_dates=["date", "asof_date"])
        diagnostics = pd.read_csv(output / "regression_diagnostics.csv", index_col="date", parse_dates=True)
        market = market_inputs()
        returns = session_returns(market, benchmark.index).loc[config["model_start"]:config["model_end"]]
    else:
        began = time.monotonic()
        print("Preparing SEC snapshots in resumable 10-company batches before loading prices", flush=True)
        batch_dir = processed / "fundamentals_batches"
        batch_dir.mkdir(exist_ok=True)
        batches = []
        for start in range(0, len(universe), 10):
            selected = universe.iloc[start:start + 10]
            path = batch_dir / f"{start:04d}.csv.gz"
            meta_path = batch_dir / f"{start:04d}.json"
            signature = {"fingerprints": fingerprints, "symbols": selected.symbol.tolist(),
                         "first_asof": str(dates.min()), "last_asof": str(dates.max())}
            cached = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            cache_valid = path.exists() and cached.get("signature") == signature
            # Upgrade earlier checkpoints that predate RF/SEC cache-state keys,
            # only when every recorded fingerprint still matches and all raw
            # inputs are demonstrably older than the checkpoint metadata.
            old = cached.get("signature", {})
            if path.exists() and not cache_valid and old:
                old_fingerprints = old.get("fingerprints", {})
                same_recorded = all(fingerprints.get(k) == old_fingerprints.get(k)
                                    for k in ["code/fundamentals.py", "universe.csv"])
                same_grid = all(old.get(k) == signature[k] for k in ["symbols", "first_asof", "last_asof"])
                checkpoint_time = meta_path.stat().st_mtime_ns
                sources = [root / config["benchmark_file"]]
                for cik in selected.cik.astype(str).str.zfill(10):
                    sources.extend([root / config["sec_cache"] / f"CIK{cik}.json",
                                    root / config["sec_cache"] / f"CIK{cik}_submissions.json"])
                inputs_older = all(p.exists() and p.stat().st_mtime_ns <= checkpoint_time for p in sources)
                cache_valid = same_recorded and same_grid and inputs_older
            cache_valid = cache_valid and _sha(path) == cached.get("sha256")
            if not cache_valid:
                chunk = build_fundamentals(selected, dates, root / config["sec_cache"])
                chunk.to_csv(path, index=False, compression="gzip")
                _write_json(meta_path, {"signature": signature, "sha256": _sha(path)})
                del chunk
                gc.collect()
            batches.append(path)
            print(f"SEC snapshots: {min(start + 10, len(universe))}/{len(universe)} companies", flush=True)
        raw_path = processed / "point_in_time_fundamentals.csv.gz"
        numeric_frames, header = [], None
        for i, path in enumerate(batches):
            chunk = pd.read_csv(path)
            if header is None:
                header = chunk.columns.tolist()
            if set(chunk) != set(header):
                raise ValueError("Inconsistent SEC batch schema")
            chunk.reindex(columns=header).to_csv(raw_path, mode="w" if i == 0 else "a", header=i == 0,
                                                 compression="gzip", index=False)
            numeric_frames.append(chunk.drop(columns=[c for c in chunk if c.endswith("_provenance")]))
            del chunk
        fundamentals = pd.concat(numeric_frames, ignore_index=True)
        del numeric_frames
        # Full field-level provenance stays in the compressed financial source;
        # the repeated exposure table keeps its dates/keys rather than JSON copies.
        gc.collect()
        market = market_inputs()
        returns = session_returns(market, benchmark.index).loc[config["model_start"]:config["model_end"]]
        print(f"Building monthly exposures: {len(fundamentals):,} stock-month rows", flush=True)
        exposures, coverage = build_exposures(market, fundamentals, universe, benchmark, dates)
        exposures.to_csv(processed / "monthly_exposures.csv.gz", index=False)
        coverage.to_csv(processed / "monthly_coverage.csv", index=False)
        print("Estimating daily constrained cross-sectional factor returns", flush=True)
        styles = STYLES + (["dividend_yield"] if config["include_dividend_yield"] else [])
        fitted = fit_factor_model(returns, exposures, benchmark.RF, factor_columns=styles,
                                  min_stocks=config["minimum_stocks"],
                                  min_residual_df=config["minimum_residual_df"],
                                  store_mimicking_weights=False, store_portfolio_diagnostics=False)
        factors, residuals, diagnostics = fitted["factor_returns"], fitted["residuals"], fitted["diagnostics"]
        factors.to_csv(output / "factor_returns.csv")
        residuals.to_csv(output / "residuals.csv.gz", index=False)
        diagnostics.to_csv(output / "regression_diagnostics.csv")
        complete = diagnostics.loc[diagnostics.status == "ok"]
        if complete.empty:
            raise RuntimeError("Expanded model produced no estimable dates; inspect monthly coverage")
        manifest = {"fingerprints": fingerprints, "universe_stocks": len(universe),
                    "universe_size_buckets": universe.size_bucket.value_counts().to_dict(),
                    "estimated_days": len(complete), "attempted_days": len(returns),
                    "first_estimated_date": str(complete.index.min().date()),
                    "last_estimated_date": str(complete.index.max().date()),
                    "daily_stocks_min": int(complete.n_stocks.min()),
                    "daily_stocks_median": float(complete.n_stocks.median()),
                    "daily_stocks_max": int(complete.n_stocks.max()),
                    "median_weighted_r2": float(complete.weighted_r2.median()),
                    "max_condition": float(complete.condition_number.max()),
                    "max_constraint_error": float(complete.constraint_error.max()),
                    "max_mimicking_exposure_error": float(complete.max_mimicking_exposure_error.max()),
                    "elapsed_seconds": round(time.monotonic() - began, 2),
                    "price_sources": market.groupby("symbol", observed=True).price_source.first().value_counts().to_dict()}
        _write_json(manifest_path, manifest)
        del fitted, fundamentals
    return {"config": config, "universe": universe, "market": market,
            "benchmark": benchmark, "returns": returns,
            "excess_returns": returns.subtract(benchmark.RF, axis=0).reindex(returns.index),
            "exposures": exposures, "coverage": coverage,
            "factor_returns": factors, "residuals": residuals,
            "diagnostics": diagnostics, "manifest": manifest}


def save_forecasts(forecasts: dict, directory: Path) -> pd.DataFrame:
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for origin, forecast in forecasts.items():
        arrays = {"symbols": np.asarray(forecast["symbols"], dtype=str)}
        for method, covariance in forecast["covariances"].items():
            arrays[f"forecast__{method}"] = np.asarray(covariance)
        for portfolio, weights in forecast.get("weights", {}).items():
            arrays[f"weights__{portfolio}"] = np.asarray(weights)
        for horizon, observed in forecast.get("realized", {}).items():
            for key in ["centered_covariance", "second_moment"]:
                arrays[f"realized_{horizon}__{key}"] = np.asarray(observed[key])
        name = f"{pd.Timestamp(origin):%Y-%m-%d}"
        target = directory / f"{name}.npz"
        np.savez_compressed(target, **arrays)
        metadata = {"origin": origin, "symbols": forecast["symbols"],
                    "methods": list(forecast["covariances"]),
                    "metadata": forecast.get("metadata", {}),
                    "realized": {str(h): {k: v for k, v in obs.items()
                                         if k not in {"centered_covariance", "second_moment"}}
                                 for h, obs in forecast.get("realized", {}).items()},
                    "units": "decimal daily covariance/second moment; multiply by horizon only under stated constant-risk assumption",
                    "npz_sha256": _sha(target)}
        _write_json(directory / f"{name}.json", metadata)
        index.append({"origin": origin, "n_stocks": len(forecast["symbols"]),
                      "file": str(target.name), "methods": ",".join(forecast["covariances"]),
                      "realized_horizons": ",".join(str(h) for h in forecast.get("realized", {})),
                      "sha256": metadata["npz_sha256"]})
    result = pd.DataFrame(index)
    result.to_csv(directory.parent / "matrix_index.csv", index=False)
    return result


def load_forecast(directory: Path, origin: str) -> dict:
    """Load one comparison pair for reporting, not the full forecast history."""
    metadata = json.loads((directory / f"{origin}.json").read_text())
    with np.load(directory / f"{origin}.npz", allow_pickle=False) as arrays:
        entry = {"symbols": arrays["symbols"].tolist(),
                 "covariances": {k.split("__", 1)[1]: arrays[k] for k in arrays.files if k.startswith("forecast__")},
                 "weights": {k.split("__", 1)[1]: arrays[k] for k in arrays.files if k.startswith("weights__")},
                 "metadata": metadata["metadata"], "realized": {}}
        for horizon, observed in metadata["realized"].items():
            entry["realized"][int(horizon)] = {**observed,
                "centered_covariance": arrays[f"realized_{horizon}__centered_covariance"],
                "second_moment": arrays[f"realized_{horizon}__second_moment"]}
    return entry


def audit_inputs(model: dict, destination: Path) -> dict:
    exposure = model["exposures"]
    checks = {
        "duplicate_exposure_rows": int(exposure.duplicated(["asof_date", "symbol"]).sum()),
        "nonpositive_caps_in_complete_rows": int((exposure.loc[exposure.complete_required_exposures].market_cap <= 0).sum()),
        "complete_exposure_nonfinite": int((~np.isfinite(exposure.loc[exposure.complete_required_exposures, STYLES].to_numpy())).sum()),
    }
    for field in ["financial_filed", "shares_filed"]:
        checks[f"future_{field}"] = int((pd.to_datetime(exposure[field]) >= exposure.asof_date).sum())
    max_filing_cache = {}
    def max_filed(value):
        if isinstance(value, dict):
            dates = ([pd.Timestamp(value["filed"])] if value.get("filed") else [])
            dates.extend(max_filed(v) for v in value.values() if isinstance(v, (dict, list)))
            return max(dates, default=pd.Timestamp.min)
        if isinstance(value, list):
            return max((max_filed(v) for v in value), default=pd.Timestamp.min)
        return pd.Timestamp.min
    provenance_count, future_provenance = 0, 0
    raw_financials = destination.parent.parent / "data/processed/point_in_time_fundamentals.csv.gz"
    for chunk in pd.read_csv(raw_financials, chunksize=3000,
                             usecols=lambda c: c == "asof_date" or c.endswith("_provenance")):
        asof = pd.to_datetime(chunk.asof_date)
        for field in [c for c in chunk if c.endswith("_provenance")]:
            for raw in chunk[field].dropna().unique():
                if raw not in max_filing_cache:
                    max_filing_cache[raw] = max_filed(json.loads(raw))
            observed = chunk[field].map(max_filing_cache)
            future_provenance += int((pd.to_datetime(observed) >= asof).sum())
            provenance_count += int(chunk[field].notna().sum())
    checks["future_nested_provenance"] = future_provenance
    cap_ratio = exposure.market_cap / (exposure.raw_close * exposure.shares_at_asof_estimate) - 1
    checks["cap_identity_errors"] = int((cap_ratio.abs() > 1e-10).sum())
    known_good = model["diagnostics"].query("status == 'ok'")
    checks["return_exposure_date_not_strictly_prior"] = int((pd.to_datetime(known_good.asof_date) >= known_good.index).sum())
    failures = {k: v for k, v in checks.items() if v}
    report = {"status": "passed" if not failures else "failed", "checks": checks,
              "provenance_cells_checked": provenance_count,
              "unique_provenance_strings": len(max_filing_cache),
              "rows": len(exposure), "months": exposure.asof_date.nunique(),
              "missing_total_shares_symbols": exposure.groupby("symbol").shares.count().loc[lambda x: x == 0].index.tolist(),
              "failures": failures}
    _write_json(destination, report)
    if failures:
        raise RuntimeError(f"Expanded input audit failed: {failures}")
    return report


def run(root: Path = ROOT, reuse_model: bool = False) -> dict:
    root = Path(root).resolve()
    model = prepare_model(root, reuse=reuse_model)
    audit = audit_inputs(model, root / "output/input_audit.json")
    from .oos_covariance import walk_forward_covariance
    print("Forecasting covariance at monthly origins and evaluating future observations", flush=True)
    forecast_config = dict(model["config"]["forecast"])
    forecast_config["store_forecasts"] = False
    streamed = []
    matrix_dir = root / "output/matrices"
    def save_one(origin, entry):
        row = save_forecasts({origin: entry}, matrix_dir).iloc[0].to_dict()
        streamed.append(row)
        pd.DataFrame(streamed).to_csv(matrix_dir.parent / "matrix_index.csv", index=False)
    evaluated = walk_forward_covariance(model["excess_returns"], model["factor_returns"],
                                         model["residuals"], model["exposures"], forecast_config,
                                         progress_callback=lambda event: print(json.dumps(_clean(event)), flush=True),
                                         forecast_callback=save_one)
    output = root / "output"
    for key in ["origins", "matrix_scores", "portfolio_scores", "summary", "portfolio_summary"]:
        if key in evaluated:
            evaluated[key].to_csv(output / f"{key}.csv", index=False)
    index = pd.DataFrame(streamed)
    if index.empty:
        raise RuntimeError("No available forecast origins")
    index.to_csv(output / "matrix_index.csv", index=False)
    _write_json(output / "evaluation_metadata.json", evaluated["metadata"])
    summary = {"status": "completed", "model": model["manifest"], "input_audit": audit["status"],
               "forecast_origins": len(index), "evaluation": evaluated["metadata"],
               "limitations": [model["config"]["universe_type"], model["config"]["sector_type"],
                               "Public revised Yahoo total returns; historical delisting completeness is not established",
                               "SEC standardized annual statements and lagged total shares; source gaps remain missing",
                               "21/63 daily observations give a noisy, often singular proxy for latent future covariance",
                               "Fixed-weight portfolio diagnostics imply daily rebalancing; no trading costs modeled",
                               model["config"]["predeclared_evaluation"]]}
    _write_json(output / "experiment_summary.json", summary)
    paired = index.loc[index.realized_horizons.map(lambda value: "21" in str(value).split(","))]
    if paired.empty:
        raise RuntimeError("No fully observed 21-session forecast pair")
    latest_origin = str(paired.iloc[-1].origin)
    evaluated["forecasts"] = {latest_origin: load_forecast(matrix_dir, latest_origin)}
    from .covariance_report import make_covariance_report
    make_covariance_report(root, model, evaluated, index)
    print(json.dumps(_clean({"status": summary["status"], "forecast_origins": len(index),
                            "model": model["manifest"]}), indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reuse-model", action="store_true")
    args = parser.parse_args()
    run(args.root, reuse_model=args.reuse_model)


if __name__ == "__main__":
    main()
