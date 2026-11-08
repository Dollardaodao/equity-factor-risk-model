"""Daily prices and corporate actions with explicit source provenance.

Yahoo chart data supplies reconstructed then-share closes, corporate actions
and adjusted total returns. Raw responses are immutable cache snapshots, and
processed files record their public source and content hashes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import tempfile
import time

import numpy as np
import pandas as pd


# Fractional share splits are verified by issuer/SEC disclosures. Yahoo also
# encodes distributions as apparent splits, so a ratio alone is insufficient.
VERIFIED_FRACTIONAL_SPLITS = {
    ("AAON", "2023-08-17"): (1.5, "https://www.sec.gov/Archives/edgar/data/824142/000082414224000030/aaon-20231231.htm"),
    ("APG", "2025-07-01"): (1.5, "https://ir.apigroupcorp.com/News/press-releases/news-details/2025/APi-Group-Reports-Second-Quarter-2025-Financial-Results-and-Raises-Full-Year-2025-Outlook/default.aspx"),
    ("BCPC", "2010-01-21"): (1.5, "https://www.sec.gov/Archives/edgar/data/9326/000091431709002241/ex99-1.txt"),
    ("CPK", "2014-09-09"): (1.5, "https://www.chpk.com/wp-content/uploads/FAQ-CPK-Stock-Split.pdf"),
    ("MMSI", "2011-05-06"): (1.25, "https://meritmedicalsystemsinc.gcs-web.com/static-files/36e4d8ce-2728-4a3f-aadf-a9d61567e087"),
    ("UGI", "2014-09-08"): (1.5, "https://www.ugicorp.com/investors/investor-resources/dividends"),
    ("WTRG", "2013-09-03"): (1.25, "https://www.essential.co/stock-information/investor-faqs"),
}


def share_split_classification(symbol, day, ratio):
    verified = VERIFIED_FRACTIONAL_SPLITS.get((symbol, str(pd.Timestamp(day).date())))
    if verified and np.isclose(ratio, verified[0]):
        return "verified fractional share split", verified[1]
    if ((ratio >= 2 and np.isclose(ratio, round(ratio)))
            or (ratio <= 0.5 and np.isclose(1 / ratio, round(1 / ratio)))):
        return "standard integral share split", None
    return "ambiguous source adjustment; not applied to shares", None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Publish a complete file without exposing partially written output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode())


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False, date_format="%Y-%m-%d",
                                     float_format="%.12g").encode())


def _date(value, zone="America/New_York"):
    return pd.Timestamp(value, unit="s", tz="UTC").tz_convert(zone).normalize().tz_localize(None)


def chart_to_daily(payload: dict, symbol: str, start, end) -> tuple[pd.DataFrame, dict]:
    """Recover then-share price/cash units from Yahoo's split-adjusted history.

    All Yahoo price-adjustment events through retrieval must be present,
    including splits or pseudo-split distribution adjustments after end.
    Total returns use Yahoo adjusted-close ratios and are never calendar-filled.
    """
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart unavailable for {symbol}")
    result = chart["result"][0]
    zone = result.get("meta", {}).get("exchangeTimezoneName", "America/New_York")
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adjusted is None:
        raise ValueError(f"No adjusted-close series for {symbol}")
    frame = pd.DataFrame({
        "date": [_date(x, zone) for x in timestamps],
        "split_adjusted_close": quote["close"],
        "adj_close": adjusted, "source_volume": quote["volume"],
    }).sort_values("date")
    if frame["date"].duplicated().any():
        raise ValueError(f"Duplicate daily dates for {symbol}")
    events = result.get("events", {})
    splits = {}
    for event in events.get("splits", {}).values():
        day = _date(event["date"], zone)
        ratio = float(event["numerator"]) / float(event["denominator"])
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid split for {symbol} on {day.date()}")
        splits[day] = splits.get(day, 1.0) * ratio
    dividends = {}
    for event in events.get("dividends", {}).values():
        day = _date(event["date"], zone)
        dividends[day] = dividends.get(day, 0.0) + float(event["amount"])
    future_factor = np.ones(len(frame))
    for day, ratio in splits.items():
        future_factor *= np.where(frame["date"] < day, ratio, 1.0)
    frame["future_split_factor"] = future_factor
    frame["raw_close"] = frame["split_adjusted_close"] * future_factor
    frame["dividend"] = frame["date"].map(dividends).fillna(0.0) * future_factor
    frame["source_adjustment_ratio"] = frame["date"].map(splits).fillna(1.0)
    # Yahoo also stores spinoff price adjustments as nonstandard "splits".
    # Preserve nonstandard ratios for raw-price reconstruction. Only explicitly
    # verified fractional share splits may also scale SEC share counts.
    standard_splits = {day: ratio for day, ratio in splits.items()
                       if share_split_classification(symbol, day, ratio)[0]
                       != "ambiguous source adjustment; not applied to shares"}
    frame["split"] = frame["date"].map(standard_splits).fillna(1.0)
    frame["volume"] = frame["source_volume"] / future_factor
    frame["dollar_volume"] = frame["split_adjusted_close"] * frame["source_volume"]
    invalid = frame[["raw_close", "adj_close"]].isna().any(axis=1)
    invalid |= (frame[["raw_close", "adj_close"]] <= 0).any(axis=1)
    dropped = [str(d.date()) for d in frame.loc[invalid, "date"]]
    # Compute on the source grid before dropping invalid rows: no return may
    # bridge an explicit missing quote and masquerade as a single-day return.
    frame.loc[invalid, "adj_close"] = np.nan
    frame["total_return"] = frame["adj_close"].pct_change(fill_method=None)
    frame = frame.loc[~invalid].copy()
    if not np.isfinite(frame[["raw_close", "adj_close"]].to_numpy()).all():
        raise ValueError(f"Nonfinite prices for {symbol}")
    frame["symbol"] = symbol
    frame["price_source"] = "Yahoo Finance public chart daily"
    frame = frame.loc[frame.date.between(pd.Timestamp(start), pd.Timestamp(end))].copy()
    order = ["date", "symbol", "raw_close", "adj_close", "dividend", "split", "volume",
             "total_return", "split_adjusted_close", "source_volume", "dollar_volume",
             "future_split_factor", "source_adjustment_ratio", "price_source"]
    audit = {
        "symbol": symbol, "rows": len(frame),
        "first_date": str(frame.date.min().date()) if len(frame) else None,
        "last_date": str(frame.date.max().date()) if len(frame) else None,
        "invalid_price_dates_dropped": dropped,
        "split_events": [{"date": str(day.date()), "ratio": ratio,
                          "share_split_applied": day in standard_splits,
                          "classification": share_split_classification(symbol, day, ratio)[0],
                          "verification_source": share_split_classification(symbol, day, ratio)[1]}
                         for day, ratio in sorted(splits.items())],
        "cash_dividend_events_in_source": len(dividends),
        "exchange_timezone": zone,
        "instrument_type": result.get("meta", {}).get("instrumentType"),
    }
    return frame[order].reset_index(drop=True), audit


def align_session_returns(frame: pd.DataFrame, sessions) -> pd.Series:
    """Return one-session changes, leaving gaps and the following return missing."""
    prices = frame.set_index("date")["adj_close"]
    daily = prices.reindex(pd.DatetimeIndex(sessions)).pct_change(fill_method=None)
    return frame["date"].map(daily)


def _fetch_symbol(symbol, cache_dir, start, end):
    cache_dir = Path(cache_dir)
    raw = cache_dir / "raw" / f"{symbol}_chart.json"
    meta_path = cache_dir / "raw" / f"{symbol}_source.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    # Yahoo OHLC carries source split/distribution adjustments through retrieval,
    # rather than only through the requested estimation end date.
    retrieval_end = max(pd.Timestamp(end) + pd.Timedelta(days=1),
                        pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None) + pd.Timedelta(days=1))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "period2": int(retrieval_end.tz_localize("UTC").timestamp()),
        "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true",
    }
    requested_meta = {"url": url, "parameters": params}
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    use_cache = raw.exists() or meta_path.exists()
    if use_cache:
        if not raw.exists() or not meta_path.exists():
            raise ValueError(f"Incomplete immutable raw cache for {symbol}; use a fresh cache directory")
        cached_parameters = existing.get("parameters", {})
        if (existing.get("url") != url
                or cached_parameters.get("period1", float("inf")) > params["period1"]
                or cached_parameters.get("period2", 0) < int((pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
                or cached_parameters.get("interval") != "1d"
                or cached_parameters.get("events") != "div,splits"
                or cached_parameters.get("includeAdjustedClose") != "true"):
            raise ValueError(f"Immutable raw cache does not cover the request for {symbol}; use a fresh cache directory")
    if not use_cache:
        from curl_cffi import requests
        error = None
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, impersonate="chrome", timeout=45)
                response.raise_for_status()
                payload = response.json()
                frame, info = chart_to_daily(payload, symbol, start, end)
                if frame.empty:
                    raise ValueError(f"No in-range prices for {symbol}")
                content = response.content
                source = {**requested_meta, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                          "raw_sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                _atomic_bytes(raw, content)
                _atomic_json(meta_path, source)
                break
            except Exception as exc:
                error = exc
                if attempt == 2:
                    raise RuntimeError(f"Public daily data failed for {symbol}: {type(error).__name__}") from error
                time.sleep(1 + attempt)
    else:
        source = existing
        if _sha(raw) != source.get("raw_sha256"):
            raise ValueError(f"Cached raw data hash mismatch: {symbol}")
        frame, info = chart_to_daily(json.loads(raw.read_text()), symbol, start, end)
    if frame.empty:
        raise ValueError(f"No in-range prices for {symbol}")
    processed = cache_dir / f"{symbol}.csv"
    _atomic_csv(processed, frame)
    return {**info, **source, "processed_sha256": _sha(processed), "processed_file": processed.name}


def fetch_market(universe, cache_dir, start="2010-01-01", end="2026-09-30", *, audit_path=None) -> dict:
    """Acquire public Yahoo snapshots, reusing verified immutable raw caches."""
    if isinstance(universe, (str, Path)):
        universe = pd.read_csv(universe)
    symbols = list(dict.fromkeys(universe["symbol"].tolist() if isinstance(universe, pd.DataFrame) else universe))
    if "SPY" not in symbols:
        symbols.append("SPY")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(audit_path) if audit_path is not None else cache_dir.parent / "market_audit.json"
    details, failures = [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_fetch_symbol, symbol, cache_dir, start, end): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                details.append(result)
                print(f"market {symbol}: {result['rows']} daily observations", flush=True)
            except Exception as exc:
                failures.append({"symbol": symbol, "error_type": type(exc).__name__})
    spy_path = cache_dir / "SPY.csv"
    if "SPY" in {item["symbol"] for item in details}:
        sessions = pd.read_csv(spy_path, parse_dates=["date"])["date"]
        for item in details:
            path = cache_dir / item["processed_file"]
            frame = pd.read_csv(path, parse_dates=["date"])
            frame["total_return"] = align_session_returns(frame, sessions)
            _atomic_csv(path, frame)
            item["processed_sha256"] = _sha(path)
            item["missing_spy_session_dates"] = [str(day.date()) for day in
                                                pd.DatetimeIndex(sessions).difference(frame.date)]
    audit = {
        "status": "failed" if failures else "complete",
        "source": "Yahoo Finance public chart daily",
        "requested_start": str(start), "requested_end": str(end),
        "requested_symbols": symbols, "succeeded": len(details), "failed": failures,
        "schema": {
            "raw_close": "Historical close in then-share units; reverse later recorded splits from Yahoo Close",
            "adj_close": "Yahoo dividend-and-split-adjusted close; current revised source snapshot",
            "dividend": "Cash dividend on ex-date, dollars per then-share; reverse later recorded splits",
            "split": "Standard integral share split/reciprocal or issuer-verified fractional share split; 1 otherwise. Ambiguous source ratios never scale SEC shares.",
            "source_adjustment_ratio": "All original Yahoo split/pseudo-split ratios, used to reconstruct raw historical prices",
            "volume": "Then-share volume: source split-adjusted volume divided by later split factor",
            "total_return": "adj_close / previous SPY-session adj_close - 1; missing-session returns remain missing",
        },
        "limitations": [
            "Current survivor-selected pilot universe; no delisted securities or historical universe membership.",
            "Yahoo revised historical data and corporate actions are not a historical-vintage dataset.",
            "raw_close reconstruction relies on recorded split events; complex spinoffs/distributions require additional validation.",
            "Ambiguous Yahoo nonstandard ratios (including spinoff adjustments) stay in source_adjustment_ratio; fractional share splits apply only for issuer-verified date/symbol events.",
            "Total returns use adjusted-close ratios; complex corporate actions and special distributions may be revised.",
        ],
        "symbols": sorted(details, key=lambda item: item["symbol"]),
    }
    audit["fractional_share_split_verifications"] = [
        {"symbol": item["symbol"], **event}
        for item in audit["symbols"] for event in item["split_events"]
        if event.get("classification") == "verified fractional share split"]
    _atomic_json(audit_path, audit)
    if failures:
        raise RuntimeError(f"Market acquisition failed for {[x['symbol'] for x in failures]}; see {audit_path}")
    return audit


def load_market(cache_dir) -> pd.DataFrame:
    files = sorted(Path(cache_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No processed daily price files in {cache_dir}")
    frames = [pd.read_csv(path, parse_dates=["date"]) for path in files]
    frame = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/symbol in market cache")
    sessions = frame.loc[frame.symbol == "SPY", "date"]
    if len(sessions):
        for _, group in frame.groupby("symbol"):
            frame.loc[group.index, "total_return"] = align_session_returns(group, sessions).to_numpy()
    return frame.reset_index(drop=True)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-09-30")
    args = parser.parse_args()
    result = fetch_market(args.universe, args.cache, args.start, args.end)
    print(f"Saved {result['succeeded']} public market histories", flush=True)
