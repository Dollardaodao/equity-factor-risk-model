"""Download official US equity benchmark factor returns (not stock exposures).

Run from the workspace: .venv/bin/python factor_project/download_benchmarks.py
Requires pandas. Original files express returns in percent; output uses decimals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
SOURCES = {
    "ff5": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]),
    "momentum": ("F-F_Momentum_Factor_daily_CSV.zip", ["Mom"]),
}


def parse_daily_archive(payload: bytes, expected_columns: list[str]) -> tuple[pd.DataFrame, str, str]:
    """Read the first daily data block, rejecting ambiguous or incomplete data."""
    with ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in archive, found {members}")
        member = members[0]
        content = archive.read(member).decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(content)))
    header_index = next((i for i, row in enumerate(rows)
                         if [cell.strip() for cell in row[1:]] == expected_columns), None)
    if header_index is None:
        raise ValueError(f"Cannot locate expected header {expected_columns}")
    records = []
    for row in rows[header_index + 1:]:
        if not row or not re.fullmatch(r"\d{8}", row[0].strip()):
            if records:
                break
            continue
        if len(row) != len(expected_columns) + 1:
            raise ValueError(f"Incorrect daily row length: {row[0]}")
        records.append([cell.strip() for cell in row])
    if not records:
        raise ValueError("No daily rows found")
    frame = pd.DataFrame(records, columns=["date", *expected_columns])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate source dates")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Source dates are not increasing")
    percent = frame[expected_columns].apply(pd.to_numeric, errors="raise")
    missing = percent.isin([-99.99, -999.0]) | percent.isna()
    if missing.any().any():
        raise ValueError(f"Source has {int(missing.sum().sum())} missing or sentinel values")
    if not np.isfinite(percent.to_numpy()).all():
        raise ValueError("Source has non-finite values")
    frame[expected_columns] = percent / 100.0
    source_creation = next((line.strip() for line in content.splitlines()
                            if "created" in line.lower()), "not supplied")
    return frame.set_index("date"), member, source_creation


def download(output: Path, as_of: str) -> dict:
    cutoff = pd.Timestamp(as_of).normalize()
    raw_dir = output / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    provenance = []
    for name, (filename, columns) in SOURCES.items():
        url = BASE + filename
        request = Request(url, headers={"User-Agent": "EquityFactorResearch/1.0"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        fetched_at = datetime.now(timezone.utc).isoformat()
        frame, member, source_creation = parse_daily_archive(payload, columns)
        source_first, source_last, source_rows = frame.index.min(), frame.index.max(), len(frame)
        excluded_future_rows = int((frame.index > cutoff).sum())
        frame = frame.loc[frame.index <= cutoff]
        if frame.empty:
            raise ValueError(f"No observations on or before {as_of}: {name}")
        (raw_dir / filename).write_bytes(payload)
        frames.append(frame)
        provenance.append({
            "name": name, "url": url, "fetched_at_utc": fetched_at,
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "archive_member": member, "source_creation_line": source_creation,
            "source_units": "percent daily return", "output_units": "decimal daily return",
            "source_rows": source_rows, "source_first_date": str(source_first.date()),
            "source_last_date": str(source_last.date()),
            "cutoff_excluded_rows": excluded_future_rows,
        })
    panel = pd.concat(frames, axis=1, join="inner").sort_index()
    if panel.empty or panel.isna().any().any() or panel.index.duplicated().any():
        raise ValueError("Merged benchmark panel is empty, incomplete, or has duplicate dates")
    if panel.index.max() > cutoff:
        raise ValueError("Panel contains observations later than requested cutoff")
    path = output / "data" / "benchmark_returns_daily.csv"
    panel.to_csv(path, date_format="%Y-%m-%d", float_format="%.8f")
    manifest = {
        "dataset": "Official US equity benchmark factor returns",
        "as_of_cutoff": str(cutoff.date()), "rows": len(panel),
        "first_date": str(panel.index.min().date()), "last_date": str(panel.index.max().date()),
        "units": "decimal daily return", "columns": list(panel.columns),
        "panel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "definitions": {
            "Mkt-RF": "Market excess return, already net of RF",
            "SMB": "Small minus big, FF five-factor construction",
            "HML": "High minus low book-to-market",
            "RMW": "Robust minus weak operating profitability (not comprehensive quality)",
            "CMA": "Conservative minus aggressive investment (not growth)",
            "RF": "Risk-free return, separately retained",
            "Mom": "Winners minus losers, not additionally reduced by RF",
        },
        "notes": [
            "These are published portfolio-based factor benchmarks, not our cross-sectional regression returns.",
            "The panel uses the common observed daily dates; no calendar filling or synthetic returns.",
            "RF is never subtracted from SMB, HML, RMW, CMA, or Mom.",
            "A retrieval-date snapshot of revised historical data is not a historical-vintage dataset.",
            "Missing sentinels -99.99/-999, duplicates, non-finite values and invalid dates are rejected.",
        ],
        "sources": provenance,
    }
    (output / "data" / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--as-of", default="2026-11-04",
                        help="Inclusive observation-date cutoff; source is the current historical vintage")
    args = parser.parse_args()
    result = download(args.output, args.as_of)
    print(json.dumps({key: result[key] for key in ("rows", "first_date", "last_date", "columns", "units")}, indent=2))
