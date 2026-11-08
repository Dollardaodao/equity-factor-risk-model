"""Verify the frozen study artifacts and reproduce the portfolio-risk figure offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def verify_results(results: Path) -> dict:
    """Check file hashes, matrix orders and the latest risk calculation."""
    results = results.resolve()
    manifest = json.loads((results / "artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (results.parent / relative).resolve()
        if not path.is_relative_to(results.parent):
            raise ValueError(f"Artifact path escapes the repository: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {relative}")

    pair = json.loads((results / "latest_matrix_pair.json").read_text())
    with np.load(results / "latest_forecast_archive.npz", allow_pickle=False) as archive:
        symbols = archive["symbols"].tolist()
        order = [symbols.index(symbol) for symbol in pair["symbols"]]
        forecast = archive["forecast__factor"]
        second_moment = archive["realized_21__second_moment"]
        realized = archive["realized_21__centered_covariance"]
        expected_matrices = {
            "latest_forecast_covariance.csv": forecast,
            "latest_realized_covariance.csv": realized,
            "latest_realized_second_moment.csv": second_moment,
            "latest_covariance_difference.csv": forecast - realized,
        }
        for name, expected in expected_matrices.items():
            frame = pd.read_csv(results / name, index_col=0)
            if frame.index.tolist() != pair["symbols"] or frame.columns.tolist() != pair["symbols"]:
                raise ValueError(f"Inconsistent matrix order: {name}")
            np.testing.assert_allclose(frame, expected[np.ix_(order, order)], rtol=1e-11, atol=1e-15)
        np.linalg.cholesky(forecast)
        weights = np.full(len(symbols), 1 / len(symbols))
        predicted = float(weights @ forecast @ weights)
        observed = float(weights @ second_moment @ weights)
        scores = pd.read_csv(results / "matrix_scores.csv")
        for method in ["factor", "ledoit_wolf_252"]:
            covariance = archive[f"forecast__{method}"]
            sign, logdet = np.linalg.slogdet(covariance)
            if sign != 1:
                raise ValueError(f"Non-positive-definite forecast: {method}")
            loss = (logdet + np.trace(np.linalg.solve(covariance, second_moment))) / len(symbols)
            row = scores.loc[(scores.origin == pair["origin"]) & (scores.horizon == 21) & (scores.method == method)].iloc[0]
            np.testing.assert_allclose(loss, row.gaussian_qlike_per_asset, rtol=1e-10)
    return {"origin": pair["origin"], "stocks": len(symbols),
            "forecast_risk_21": np.sqrt(21 * predicted),
            "realized_risk_21": np.sqrt(21 * observed),
            "realized_centered_rank": int(np.linalg.matrix_rank(realized)),
            "verified_files": len(manifest["files"])}


def plot_risk_history(results: Path, output: Path) -> Path:
    """Plot identical equal-weight portfolios across covariance estimators."""
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Use a resident system font when available; Matplotlib supplies a portable fallback.
    font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    data = pd.read_csv(results / "equal_weight_risk.csv", parse_dates=["origin"])
    actual = data.loc[data.method == "factor"].sort_values("origin")
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot(actual.origin, 100 * np.sqrt(21 * actual.realized_daily_second_moment),
            color="#283b42", linewidth=1.7, label="Next 21 sessions: realized risk proxy")
    methods = {"factor": ("Factor model", "#176c78"),
               "ewma_252": ("EWMA 252", "#4c84c5"),
               "ledoit_wolf_252": ("Ledoit-Wolf 252", "#dc8a43")}
    for method, (label, color) in methods.items():
        selected = data.loc[data.method == method].sort_values("origin")
        ax.plot(selected.origin, 100 * np.sqrt(21 * selected.forecast_daily_variance),
                color=color, linewidth=1.3, label=label)
    ax.axvline(pd.Timestamp("2022-01-01"), color="#a5afb2", linestyle="--", linewidth=1)
    ax.set(title="Equal-weight portfolio: predicted versus subsequent risk",
           ylabel="21-session risk proxy (%)", xlabel="Monthly forecast origin")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    path = output / "predicted_vs_realized_risk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if not (args.results / "artifact_manifest.json").is_file():
        parser.error("Run from the repository root or pass --results /path/to/results")
    latest = verify_results(args.results)
    summary = pd.read_csv(args.results / "summary.csv")
    selected = summary.loc[(summary.split == "holdout") & summary.method.isin(["factor", "ledoit_wolf_252"]),
                           ["horizon", "method", "n_origins", "mean_relative_frobenius", "mean_gaussian_qlike_per_asset"]]
    print("Frozen study results, evaluation origins from 2022 onward")
    print(selected.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(json.dumps(latest, indent=2))
    print(f"Figure written to {plot_risk_history(args.results, args.output)}")


if __name__ == "__main__":
    main()
