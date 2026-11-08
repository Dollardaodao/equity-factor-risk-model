"""Prepare the shipped research cohort from public data, with reusable caches.

Run ``python -m factor_project.prepare_data --root research`` from the repository.
Live SEC requests require SEC_USER_AGENT with a descriptive name and contact
email. Existing valid caches are reused without credentials or a paid feed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .download_benchmarks import download as download_benchmarks
from .fundamentals import download_companyfacts
from .market_data import _atomic_bytes, _atomic_csv, _atomic_json, _sha, align_session_returns, fetch_market


def _inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
        raise ValueError(f"Data configuration path must stay inside the research root: {relative}")
    return root / path


def _benchmark(root: Path, cutoff: str) -> dict:
    data = root / "data"
    panel_path = data / "benchmark_returns_daily.csv"
    manifest_path = data / "benchmark_manifest.json"
    if panel_path.exists() or manifest_path.exists():
        if not panel_path.exists() or not manifest_path.exists():
            raise ValueError("Incomplete benchmark cache; use a fresh research data directory")
        manifest = json.loads(manifest_path.read_text())
        if _sha(panel_path) != manifest.get("panel_sha256"):
            raise ValueError("Benchmark cache hash mismatch")
        if manifest.get("as_of_cutoff") != str(pd.Timestamp(cutoff).date()):
            raise ValueError("Benchmark cache cutoff differs from price_end; use a fresh data directory")
        for source in manifest["sources"]:
            path = data / "raw" / source["url"].rsplit("/", 1)[-1]
            if not path.is_file() or _sha(path) != source["sha256"]:
                raise ValueError(f"Benchmark raw archive missing or changed: {path.name}")
        return manifest
    # The downloader writes into an isolated temporary root. An incomplete
    # network operation cannot replace a previously usable benchmark panel.
    with tempfile.TemporaryDirectory(prefix=".benchmark_download_", dir=data) as temporary:
        staging = Path(temporary)
        manifest = download_benchmarks(staging, cutoff)
        for source in sorted((staging / "data").rglob("*"),
                             key=lambda path: path.name == "benchmark_manifest.json"):
            if source.is_file():
                destination = data / source.relative_to(staging / "data")
                if destination.exists():
                    if _sha(destination) != _sha(source):
                        raise ValueError(f"Refusing to replace existing benchmark source: {destination.name}")
                else:
                    _atomic_bytes(destination, source.read_bytes())
        return manifest


def _validate_frame(frame: pd.DataFrame, symbol: str) -> None:
    required = {"date", "symbol", "raw_close", "adj_close", "dividend", "split", "volume", "total_return"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError(f"Incomplete market schema or empty history: {symbol}")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError(f"Duplicate or unsorted market dates: {symbol}")
    if set(frame.symbol) != {symbol}:
        raise ValueError(f"Market file has an incorrect symbol: {symbol}")
    prices = frame[["raw_close", "adj_close"]].to_numpy()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"Invalid market prices: {symbol}")
    if not np.isfinite(frame["split"]).all() or (frame["split"] <= 0).any():
        raise ValueError(f"Invalid share-split ratios: {symbol}")


def stage_market_identity(root: Path, config: dict, universe: pd.DataFrame, downloaded: dict) -> dict:
    """Validate and publish model prices without changing downloaded originals.

    Rules are applied to a new directory. The first return of each admitted
    segment is missing. An excluded candidate remains in the universe/audit,
    while its model price file is absent. No model checkpoints are required.
    """
    root = Path(root).resolve()
    source = root / "data/market_downloaded"
    destination = _inside(root, config["market_cache"])
    if destination.resolve() == source.resolve():
        raise ValueError("Downloaded prices and identity-screened model prices require separate directories")
    rules_path = root / "market_identity_rules.json"
    rulebook = json.loads(rules_path.read_text())
    rules = rulebook.get("rules", {})
    requested = list(dict.fromkeys([*universe.symbol.astype(str), "SPY"]))
    unknown_rules = set(rules) - set(requested)
    if unknown_rules:
        raise ValueError(f"Identity rules reference symbols outside the cohort: {sorted(unknown_rules)}")
    details = {item["symbol"]: item for item in downloaded["symbols"]}
    if downloaded.get("failed") or set(details) != set(requested):
        raise ValueError("Download audit does not contain a complete requested cohort")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".market_validated_", dir=destination.parent))
    backup = staging.with_name(staging.name + "_previous")
    audit_path = root / "data/market_audit.json"
    old_audit = audit_path.read_bytes() if audit_path.exists() else None
    promoted = False
    try:
        spy = pd.read_csv(source / "SPY.csv", parse_dates=["date"])
        _validate_frame(spy, "SPY")
        sessions = pd.DatetimeIndex(spy.date)
        applied, model_details = [], []
        for symbol in requested:
            path = source / f"{symbol}.csv"
            if not path.is_file() or _sha(path) != details[symbol]["processed_sha256"]:
                raise ValueError(f"Downloaded price file missing or hash mismatch: {symbol}")
            frame = pd.read_csv(path, parse_dates=["date"])
            _validate_frame(frame, symbol)
            source_rows = len(frame)
            rule = rules.get(symbol)
            if rule:
                action = rule.get("action")
                if action == "exclude":
                    frame = frame.iloc[0:0].copy()
                elif action == "start_at":
                    frame = frame.loc[frame.date >= pd.Timestamp(rule["date"])].copy()
                    if frame.empty:
                        raise ValueError(f"Identity cutoff leaves no usable history: {symbol}")
                else:
                    raise ValueError(f"Unknown identity action for {symbol}: {action}")
                applied.append({"symbol": symbol, **rule, "source_rows": source_rows,
                                "model_rows": len(frame), "removed_rows": source_rows - len(frame)})
            if frame.empty:
                continue
            frame["total_return"] = align_session_returns(frame, sessions)
            _atomic_csv(staging / path.name, frame)
            model_details.append({**details[symbol], "processed_file": path.name,
                                  "downloaded_processed_sha256": details[symbol]["processed_sha256"],
                                  "processed_sha256": _sha(staging / path.name), "rows": len(frame),
                                  "first_date": str(frame.date.min().date()),
                                  "last_date": str(frame.date.max().date())})
        audit = {**downloaded, "status": "complete", "succeeded": len(model_details),
                 "symbols": model_details,
                 "downloaded_requested_symbols": downloaded["requested_symbols"],
                 "model_identity_screen": {"policy": rulebook["policy"], "rules": applied,
                                            "rules_sha256": _sha(rules_path)},
                 "note": "Downloaded raw/source cache is retained unchanged in data/market_downloaded. Model files apply dated identity rules; excluded candidates have no model price CSV."}
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staging, destination)
        promoted = True
        _atomic_json(audit_path, audit)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        return audit
    except Exception:
        if promoted:
            shutil.rmtree(destination)
        if backup.exists():
            os.replace(backup, destination)
        if old_audit is not None:
            _atomic_bytes(audit_path, old_audit)
        elif audit_path.exists():
            audit_path.unlink()
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def prepare_data(root: Path) -> dict:
    root = Path(root).resolve()
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = data / "preparation_manifest.json"
    result = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(manifest_path, result)
    try:
        config = json.loads((root / "config.json").read_text())
        if config["sec_cache"] != "data/sec" or config["benchmark_file"] != "data/benchmark_returns_daily.csv":
            raise ValueError("Portable setup requires sec_cache=data/sec and benchmark_file=data/benchmark_returns_daily.csv")
        universe_path = _inside(root, config["universe_file"])
        universe = pd.read_csv(universe_path, dtype={"cik": str})
        if not {"symbol", "cik"}.issubset(universe) or universe.symbol.duplicated().any():
            raise ValueError("Universe requires unique symbols and CIK identifiers")
        benchmark = _benchmark(root, config["price_end"])
        sec = download_companyfacts(universe, data / "sec")
        if sec.get("failed") or any("error" in item for item in sec["identities"]):
            raise RuntimeError("SEC acquisition is incomplete; see data/sec/download_summary.json")
        rulebook = json.loads((root / "market_identity_rules.json").read_text())
        excluded = {symbol for symbol, rule in rulebook["rules"].items() if rule["action"] == "exclude"}
        unverified = [item["symbol"] for item in sec["identities"]
                      if item["symbol"] not in excluded and (not item.get("verified") or item.get("ticker_match") is not True)]
        if unverified:
            raise ValueError(f"SEC issuer identity requires review before model staging: {unverified}")
        downloaded = fetch_market(universe, data / "market_downloaded", config["price_start"], config["price_end"],
                                  audit_path=data / "market_download_audit.json")
        market = stage_market_identity(root, config, universe, downloaded)
        result.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "universe_sha256": _sha(universe_path), "config_sha256": _sha(root / "config.json"),
                       "benchmark": {key: benchmark[key] for key in ["panel_sha256", "rows", "first_date", "last_date"]},
                       "sec_files": len(sec["files"]), "downloaded_market_symbols": downloaded["succeeded"],
                       "model_market_symbols": market["succeeded"], "excluded_symbols": sorted(excluded),
                       "market_audit_sha256": _sha(data / "market_audit.json"),
                       "source_vintage_note": "Public historical data is retrieved at setup time and can be revised. The resulting hashes identify this run's actual input snapshot."})
        _atomic_json(manifest_path, result)
        return result
    except Exception as exc:
        result.update({"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                       "error_type": type(exc).__name__, "error": str(exc)})
        _atomic_json(manifest_path, result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research"))
    args = parser.parse_args()
    result = prepare_data(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
