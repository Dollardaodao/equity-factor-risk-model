"""Render saved covariance results without repeating financial/model estimation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .expanded_run import ROOT, load_forecast
from .covariance_report import make_covariance_report


def render(root: Path):
    output = root / "output"
    model = {"universe": pd.read_csv(root / "universe.csv"),
             "exposures": pd.read_csv(root / "data/processed/monthly_exposures.csv.gz", parse_dates=["asof_date"]),
             "manifest": json.loads((output / "model_manifest.json").read_text())}
    evaluated = {key: pd.read_csv(output / f"{key}.csv") for key in
                 ["origins", "matrix_scores", "portfolio_scores", "summary", "portfolio_summary"]}
    index = pd.read_csv(output / "matrix_index.csv")
    paired = index.loc[index.realized_horizons.map(lambda value: "21" in str(value).split(","))]
    latest = str(paired.iloc[-1].origin)
    evaluated["forecasts"] = {latest: load_forecast(output / "matrices", latest)}
    return make_covariance_report(root, model, evaluated, index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(render(args.root))
