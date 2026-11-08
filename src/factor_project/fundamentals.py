"""Conservative SEC Company Facts inputs for a point-in-time factor prototype.

Only facts filed strictly before the requested as-of date are eligible. Annual
flows require a 10-K (or 10-K/A) and 330--400 days; every financial metric uses
the same latest available annual period end. Original and amended observations
remain in the cache, so a later restatement cannot leak into an earlier panel.

Company Facts contains standardized, entity-wide XBRL only. Missing extensions,
inconsistent tags, and stale observations remain missing rather than becoming
zero. This is a research extractor, not a full commercial PIT accounting feed.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time
from typing import Iterable
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts"
def _sec_user_agent() -> str:
    """Require a descriptive identity and contact email only for live SEC access."""
    value = os.environ.get("SEC_USER_AGENT", "").strip()
    email = re.search(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", value)
    description = re.sub(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", "", value).strip(" <>-()")
    if (not email or len(description) < 3 or "\n" in value or "\r" in value):
        raise ValueError("Live SEC downloads require SEC_USER_AGENT with a descriptive name and contact email, for example 'Your Research Name contact@your-domain.com'")
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, value) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode())
ANNUAL_FORMS = {"10-K", "10-K/A"}
SHARE_FORMS = ANNUAL_FORMS | {"10-Q", "10-Q/A"}
FINANCIAL_MAX_AGE_DAYS = 550
SHARES_MAX_AGE_DAYS = 180

# A current ticker can move to a new registrant while the old CIK continues
# filing as a subsidiary. Explicit effective dates prevent a ticker-only join
# from silently carrying the predecessor's facts into the successor's history.
# This registry blocks old CIKs; it never backfills from a successor automatically.
CIK_VALIDITY = {
    "0000034088": {
        "unavailable_from": "2026-07-01", "successor_cik": "0002115436",
        "source_url": "https://www.sec.gov/Archives/edgar/data/2115436/000003408826000093/xom-20260630.htm",
        "event": "ExxonMobil Holdings became the listed parent and successor registrant",
    },
    "0001602065": {
        "unavailable_from": "2025-08-19", "successor_cik": "0002074176",
        "source_url": "https://www.viperenergy.com/static-files/e138789d-d54e-4f09-8f56-76074744e0bb",
        "event": "New Viper became the listed parent; the former issuer became VNOM Sub, Inc.",
    },
}

# Ordered aliases prefer narrower and more common definitions. The selected
# exact tag, filing and period are included with each output field.
TAGS = {
    "assets": ("Assets",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ),
    "operating_cashflow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": (
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
    ),
    "debt_current": (
        "LongTermDebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
    ),
    "debt_noncurrent": (
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    ),
    "debt_shortterm": ("ShortTermBorrowings", "CommercialPaper"),
}
FLOWS = {"net_income", "revenue", "operating_cashflow", "gross_profit", "cost_of_revenue"}
BASE_FIELDS = [
    "assets", "equity", "net_income", "revenue", "operating_cashflow",
    "gross_profit", "debt", "shares",
]


def _date(value) -> date | None:
    if value is None:
        return None
    # SEC JSON dates are ISO strings. Avoid constructing millions of pandas
    # Timestamp objects when parsing hundreds of company histories.
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    if type(value) is date:
        return value
    if pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        return None


def _cik(value) -> str:
    """Normalize strings or numeric CSV CIKs without dropping leading zeros."""
    return f"{int(float(value)):010d}"


def download_companyfacts(universe: pd.DataFrame, cache_dir: Path) -> dict:
    """Download SEC JSON serially, <=2.5 requests/sec, using a factual user-agent.

    Existing valid JSON files are reused. Three attempts handle transient
    transport/5xx errors. An SEC 403/429 stops this batch (no access-control
    workaround). No credentials or paid service is required.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = {"downloaded": [], "cached": [], "failed": [], "blocked": False}
    last_request = 0.0
    for _, company in universe.iterrows():
        symbol = str(company["symbol"])
        try:
            cik = _cik(company["cik"])
        except (TypeError, ValueError, OverflowError):
            summary["failed"].append({"symbol": symbol, "error": "missing/invalid CIK"})
            continue
        target = cache_dir / f"CIK{cik}.json"
        if target.exists():
            try:
                payload = json.loads(target.read_text())
                if "facts" in payload and _cik(payload["cik"]) == cik:
                    summary["cached"].append(symbol)
                    continue
            except (ValueError, KeyError, TypeError):
                pass
            summary["failed"].append({"symbol": symbol, "error": "invalid immutable companyfacts cache; use a fresh cache directory"})
            continue
        if summary["blocked"]:
            summary["failed"].append({"symbol": symbol, "error": "batch stopped after SEC access/rate refusal"})
            continue
        error = None
        for attempt in range(3):
            pause = 0.4 - (time.monotonic() - last_request)
            if pause > 0:
                time.sleep(pause)
            request = urllib.request.Request(
                f"{SEC_BASE}/CIK{cik}.json",
                headers={"User-Agent": _sec_user_agent(), "Accept": "application/json"},
            )
            last_request = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    content = response.read()
                payload = json.loads(content)
                if "facts" not in payload or _cik(payload.get("cik", -1)) != cik:
                    raise ValueError("SEC response lacks facts or has a mismatched CIK")
                _atomic_bytes(target, content)
                summary["downloaded"].append(symbol)
                error = None
                break
            except urllib.error.HTTPError as exc:
                error = f"HTTP {exc.code}"
                if exc.code in {403, 429}:
                    summary["blocked"] = True
                    break
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
        if error:
            summary["failed"].append({"symbol": symbol, "error": error})
    summary["identities"] = _verify_identities(universe, cache_dir, blocked=summary["blocked"])
    summary["identity_mismatches"] = [item for item in summary["identities"]
                                      if item.get("ticker_match") is False or item.get("cik_match") is False]
    summary["source"] = SEC_BASE
    summary["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    summary["status"] = "failed" if summary["failed"] or any("error" in item for item in summary["identities"]) else "complete"
    valid_ciks = set()
    for value in universe.cik:
        try:
            valid_ciks.add(_cik(value))
        except (TypeError, ValueError, OverflowError):
            continue
    summary["files"] = [{"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for cik in sorted(valid_ciks)
                        for path in [cache_dir / f"CIK{cik}.json", cache_dir / f"CIK{cik}_submissions.json"]
                        if path.is_file()]
    _atomic_json(cache_dir / "download_summary.json", summary)
    return summary


def _verify_identities(universe: pd.DataFrame, cache_dir: Path, blocked: bool = False) -> list[dict]:
    """Cross-check manually entered CIKs against SEC submissions' current tickers.

    A current ticker check cannot itself establish historical symbol mappings.
    A mismatch is flagged and build_fundamentals excludes the entity's values.
    """
    identities = []
    last_request = 0.0
    for _, company in universe.iterrows():
        symbol = str(company["symbol"])
        try:
            cik = _cik(company["cik"])
        except (ValueError, TypeError, OverflowError):
            identities.append({"symbol": symbol, "verified": False, "error": "invalid CIK"})
            continue
        target = cache_dir / f"CIK{cik}_submissions.json"
        payload = None
        error = None
        if target.exists():
            try:
                payload = json.loads(target.read_text())
                if not isinstance(payload, dict):
                    payload = None
                    raise ValueError("SEC submissions cache is not an object")
            except ValueError:
                error = "invalid immutable submissions cache; use a fresh cache directory"
        if payload is None and not blocked and not target.exists():
            for attempt in range(3):
                pause = 0.4 - (time.monotonic() - last_request)
                if pause > 0:
                    time.sleep(pause)
                request = urllib.request.Request(
                    f"https://data.sec.gov/submissions/CIK{cik}.json",
                    headers={"User-Agent": _sec_user_agent(), "Accept": "application/json"})
                last_request = time.monotonic()
                try:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        content = response.read()
                    candidate = json.loads(content)
                    if not isinstance(candidate, dict) or _cik(candidate.get("cik", -1)) != cik:
                        raise ValueError("SEC submissions response has a mismatched CIK")
                    _atomic_bytes(target, content)
                    payload = candidate
                    break
                except urllib.error.HTTPError as exc:
                    error = f"HTTP {exc.code}"
                    if exc.code in {403, 429}:
                        blocked = True
                        break
                    if exc.code < 500:
                        break
                except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    time.sleep(attempt + 1)
        if payload is None:
            identities.append({"symbol": symbol, "cik": cik, "verified": False,
                               "error": error or "SEC access stopped earlier in batch"})
        else:
            tickers = payload.get("tickers", [])
            normalize = lambda ticker: str(ticker).upper().replace(".", "-")
            match = normalize(symbol) in {normalize(ticker) for ticker in tickers} if tickers else None
            cik_match = _cik(payload.get("cik", -1)) == cik
            identities.append({"symbol": symbol, "cik": cik, "verified": bool(tickers) and cik_match,
                               "entity_name": payload.get("name"), "tickers": tickers,
                               "ticker_match": match,
                               "cik_match": cik_match})
    _atomic_json(cache_dir / "identity_summary.json", identities)
    return identities


def _records(payload: dict, namespace: str, tags: Iterable[str], unit: str) -> list[dict]:
    records = []
    for rank, tag in enumerate(tags):
        values = payload.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {}).get(unit, [])
        for value in values:
            end, filed, start = _date(value.get("end")), _date(value.get("filed")), _date(value.get("start"))
            numeric = value.get("val")
            if not end or not filed or not isinstance(numeric, (float, int)) or not np.isfinite(numeric):
                continue
            records.append({**value, "end": end, "filed": filed, "start": start,
                            "tag": f"{namespace}:{tag}", "unit": unit, "rank": rank})
    return records


def _eligible(records: list[dict], asof: date, flow: bool = False, shares: bool = False) -> list[dict]:
    forms = SHARE_FORMS if shares else ANNUAL_FORMS
    valid = []
    for record in records:
        if record.get("form") not in forms or record["filed"] >= asof or record["end"] >= asof:
            continue
        if flow:
            start = record["start"]
            if start is None or not 330 <= (record["end"] - start).days <= 400:
                continue
        elif record["start"] is not None:
            continue  # Instantaneous balances/shares only, never weighted averages.
        valid.append(record)
    return valid


def _select(records: list[dict], period_end: date) -> dict | None:
    same = [record for record in records if record["end"] == period_end]
    if not same:
        return None
    # Prefer the latest available filing first; an old alias should not override
    # a new disclosure. For same filing date, use the documented alias order.
    return max(same, key=lambda r: (r["filed"], -r["rank"], str(r.get("accn", ""))))


def _provenance(record: dict | None) -> str | None:
    if record is None:
        return None
    if "_provenance" not in record:
        record["_provenance"] = json.dumps({key: record.get(key) for key in
                                           ["tag", "unit", "start", "end", "filed", "accn", "form", "val"]},
                                          default=str, sort_keys=True)
    return record["_provenance"]


def _put(row: dict, field: str, record: dict | None) -> None:
    row[field] = float(record["val"]) if record is not None else np.nan
    row[f"{field}_provenance"] = _provenance(record)


def _financial_snapshot(all_records: dict[str, list[dict]], asof: date, period_end: date) -> tuple[dict, dict]:
    chosen = {field: _select(_eligible(records, asof, flow=field in FLOWS), period_end)
              for field, records in all_records.items()}
    return _financial_values(chosen)


def _financial_values(chosen: dict[str, dict | None]) -> tuple[dict, dict]:
    row = {}
    for field in ["assets", "equity", "net_income", "revenue", "operating_cashflow", "gross_profit"]:
        _put(row, field, chosen[field])
    if chosen["gross_profit"] is None:
        revenue, cost = chosen["revenue"], chosen["cost_of_revenue"]
        if revenue and cost and revenue["start"] == cost["start"]:
            row["gross_profit"] = float(revenue["val"] - cost["val"])
            row["gross_profit_provenance"] = json.dumps({
                "method": "revenue minus cost_of_revenue", "revenue": json.loads(_provenance(revenue)),
                "cost_of_revenue": json.loads(_provenance(cost)),
            }, sort_keys=True)
    # Strict total debt: all three components must be disclosed. Non-disclosure
    # is not interpreted as zero; root can use alternative leverage descriptors.
    debt_parts = {field: chosen[field] for field in ["debt_current", "debt_noncurrent", "debt_shortterm"]}
    for field, record in debt_parts.items():
        _put(row, field, record)
    if all(record is not None for record in debt_parts.values()):
        row["debt"] = sum(float(record["val"]) for record in debt_parts.values())
    else:
        row["debt"] = np.nan
    row["debt_provenance"] = json.dumps({
        "method": "current long-term debt + noncurrent long-term debt + short-term borrowings",
        "components": {field: json.loads(_provenance(record)) if record else None
                       for field, record in debt_parts.items()},
        "missing_components": [field for field, record in debt_parts.items() if record is None],
    }, sort_keys=True)
    return row, chosen


class _FilingIndex:
    """Advance monotonically through filing dates, retaining PIT revisions.

    For each period end, the best *already filed* fact is updated only when its
    filing becomes strictly earlier than asof. Static form/duration checks run
    once, instead of rescanning the entire history at every monthly snapshot.
    """
    def __init__(self, records: list[dict], *, flow: bool = False, shares: bool = False):
        valid = _eligible(records, date.max, flow=flow, shares=shares)
        if shares:
            valid = [record for record in valid if record["val"] > 0]
        self.records = sorted(valid, key=lambda r: r["filed"])
        self.cursor = 0
        self.by_end: dict[date, dict] = {}
        self.shares = shares
        self.last_asof = date.min

    def _score(self, record):
        if self.shares:
            return (record["filed"], record["tag"].startswith("dei:"))
        return (record["filed"], -record["rank"], str(record.get("accn", "")))

    def advance(self, asof: date) -> None:
        if asof < self.last_asof:
            raise ValueError("Filing indexes must advance in chronological order")
        self.last_asof = asof
        while self.cursor < len(self.records) and self.records[self.cursor]["filed"] < asof:
            record = self.records[self.cursor]
            previous = self.by_end.get(record["end"])
            if previous is None or self._score(record) > self._score(previous):
                self.by_end[record["end"]] = record
            self.cursor += 1

    def eligible_ends(self, asof: date) -> set[date]:
        # Preserve the original separate period-end cutoff, including for any
        # malformed source record whose end happens to be after its filing.
        return {end for end in self.by_end if end < asof}


def build_fundamentals(universe: pd.DataFrame, asof_dates: Iterable, cache_dir: Path) -> pd.DataFrame:
    """Return PIT annual financial values and actual shares with field provenance.

    Financial rows older than 550 days at as-of are entirely unavailable. Shares
    require an observation within 180 days and remain independently selected.
    Revenue growth is log(revenue_t / revenue_t-3)/3 (both must be positive).
    Earnings change is (NI_t - NI_t-3) / assets_t-3, allowing negative earnings.
    A 3-year comparison allows a +/-45 day fiscal-calendar shift.
    """
    cache_dir = Path(cache_dir)
    dates = sorted({_date(value) for value in asof_dates} - {None})
    rows = []
    for _, company in universe.iterrows():
        symbol = str(company["symbol"])
        target = cache_dir / f"CIK{_cik(company['cik'])}.json"
        payload = json.loads(target.read_text()) if target.exists() else {}
        identity_file = cache_dir / f"CIK{_cik(company['cik'])}_submissions.json"
        identity = json.loads(identity_file.read_text()) if identity_file.exists() else {}
        normalize = lambda ticker: str(ticker).upper().replace(".", "-")
        identity_match = normalize(symbol) in {normalize(ticker) for ticker in identity.get("tickers", [])} if identity.get("tickers") else None
        identity_cik_match = _cik(identity.get("cik", -1)) == _cik(company["cik"]) if identity else None
        if identity_match is False or identity_cik_match is False or (payload and _cik(payload.get("cik", -1)) != _cik(company["cik"])):
            payload = {}  # Known mismatches must not enter research outputs.
        all_records = {field: _records(payload, "us-gaap", tags, "USD") for field, tags in TAGS.items()}
        share_records = _records(payload, "dei", ["EntityCommonStockSharesOutstanding"], "shares")
        share_records += _records(payload, "us-gaap", ["CommonStockSharesOutstanding"], "shares")
        indexes = {field: _FilingIndex(records, flow=field in FLOWS) for field, records in all_records.items()}
        share_index = _FilingIndex(share_records, shares=True)
        financial_cache, growth_cache = {}, {}
        validity = CIK_VALIDITY.get(_cik(company["cik"]), {})
        unavailable_from = _date(validity.get("unavailable_from"))

        def snapshot(period_end):
            chosen = {field: index.by_end.get(period_end) for field, index in indexes.items()}
            signature = tuple(id(record) if record is not None else 0 for record in chosen.values())
            if signature not in financial_cache:
                financial_cache[signature] = _financial_values(chosen)
            return financial_cache[signature]

        for asof in dates:
            for index in indexes.values():
                index.advance(asof)
            share_index.advance(asof)
            unavailable_due_to_successor = unavailable_from is not None and asof >= unavailable_from
            row = {"symbol": symbol, "asof_date": pd.Timestamp(asof), "cik": _cik(company["cik"]),
                   "entity_name": identity.get("name", payload.get("entityName")),
                   "identity_verified": bool(identity.get("tickers")) and bool(identity_cik_match), "identity_match": identity_match,
                   "identity_cik_verified": bool(identity_cik_match),
                   "source": str(target), "financial_period_end": pd.NaT, "financial_filed": pd.NaT,
                   "financial_stale": False, "source_missing": not bool(payload),
                   "source_unavailable_due_to_successor": unavailable_due_to_successor,
                   "cik_valid_until_exclusive": pd.Timestamp(unavailable_from) if unavailable_from else pd.NaT,
                   "successor_cik": validity.get("successor_cik"),
                   "cik_validity_source": validity.get("source_url"),
                   "cik_validity_event": validity.get("event"),
                   "revenue_3y_growth": np.nan, "earnings_3y_change": np.nan,
                   "growth_base_period_end": pd.NaT, "growth_provenance": None}
            for field in BASE_FIELDS:
                _put(row, field, None)
            # Require an actual annual flow to anchor the latest fiscal year;
            # instantaneous balances repeated in a 10-K are not sufficient.
            annual_ends = set() if unavailable_due_to_successor else set().union(*(indexes[field].eligible_ends(asof) for field in FLOWS))
            if annual_ends:
                period_end = max(annual_ends)
                row["financial_period_end"] = pd.Timestamp(period_end)
                row["financial_stale"] = (asof - period_end).days > FINANCIAL_MAX_AGE_DAYS
                if not row["financial_stale"]:
                    financial, chosen = snapshot(period_end)
                    row.update(financial)
                    filed = [record["filed"] for record in chosen.values() if record]
                    row["financial_filed"] = pd.Timestamp(max(filed)) if filed else pd.NaT
                    target_end = period_end - timedelta(days=round(365.25 * 3))
                    previous_ends = {end for end in annual_ends if abs((end - target_end).days) <= 45}
                    if previous_ends:
                        previous_end = min(previous_ends, key=lambda end: abs((end - target_end).days))
                        previous, old_chosen = snapshot(previous_end)
                        row["growth_base_period_end"] = pd.Timestamp(previous_end)
                        growth_signature = tuple(id(chosen[field]) for field in ["revenue", "net_income"])
                        growth_signature += tuple(id(old_chosen[field]) for field in ["revenue", "net_income", "assets"])
                        if growth_signature not in growth_cache:
                            growth = {"revenue_3y_growth": np.nan, "earnings_3y_change": np.nan}
                            if row["revenue"] > 0 and previous["revenue"] > 0:
                                growth["revenue_3y_growth"] = np.log(row["revenue"] / previous["revenue"]) / 3
                            if previous["assets"] > 0 and np.isfinite(row["net_income"]) and np.isfinite(previous["net_income"]):
                                growth["earnings_3y_change"] = (row["net_income"] - previous["net_income"]) / previous["assets"]
                            growth["growth_provenance"] = json.dumps({
                                "method": {"revenue": "log(revenue_t/revenue_t_minus_3)/3",
                                           "earnings": "(net_income_t-net_income_t_minus_3)/assets_t_minus_3"},
                                "current": {field: json.loads(_provenance(chosen[field])) if chosen[field] else None
                                            for field in ["revenue", "net_income"]},
                                "base": {field: json.loads(_provenance(old_chosen[field])) if old_chosen[field] else None
                                         for field in ["revenue", "net_income", "assets"]},
                            }, sort_keys=True)
                            growth_cache[growth_signature] = growth
                        row.update(growth_cache[growth_signature])
            shares = [record for end, record in share_index.by_end.items()
                      if not unavailable_due_to_successor and end < asof
                      and 0 <= (asof - end).days <= SHARES_MAX_AGE_DAYS and record["val"] > 0]
            share = max(shares, key=lambda r: (r["end"], r["filed"], r["tag"].startswith("dei:"))) if shares else None
            _put(row, "shares", share)
            row["shares_period_end"] = pd.Timestamp(share["end"]) if share else pd.NaT
            row["shares_filed"] = pd.Timestamp(share["filed"]) if share else pd.NaT
            row["shares_age_days"] = (asof - share["end"]).days if share else np.nan
            rows.append(row)
    frame = pd.DataFrame(rows)
    for field in ["asof_date", "financial_period_end", "financial_filed", "shares_period_end", "shares_filed", "growth_base_period_end", "cik_valid_until_exclusive"]:
        if field in frame:
            frame[field] = pd.to_datetime(frame[field])
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    result = download_companyfacts(pd.read_csv(args.universe), args.cache_dir)
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)
