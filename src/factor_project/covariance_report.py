"""Report covariance forecasts against strictly subsequent realized observations."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

LABELS = {"factor": "Factor model", "sample_252_jitter": "Sample 252 + jitter",
          "ewma_252": "EWMA 252", "ledoit_wolf_252": "Ledoit-Wolf 252", "diagonal_ewma": "Diagonal EWMA"}
COLORS = {"factor": "#176c78", "sample_252_jitter": "#858e94", "ewma_252": "#4c84c5",
          "ledoit_wolf_252": "#dc8a43", "diagonal_ewma": "#9b65a5"}
FAMILIES = {"equal_weight": "Equal weight", "cap_weight": "Cap weight",
            "random_longonly": "Random long-only", "dollar_neutral_random": "Dollar-neutral random"}


def _embed(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _correlation(covariance):
    scale = np.sqrt(np.maximum(np.diag(covariance), 0))
    return np.divide(covariance, np.outer(scale, scale), out=np.full_like(covariance, np.nan),
                     where=np.outer(scale, scale) > 0)


def make_covariance_report(root: Path, model: dict, evaluated: dict, matrix_index: pd.DataFrame) -> Path:
    output, figures = root / "output", root / "output/figures"
    figures.mkdir(exist_ok=True)
    matrix, portfolios = evaluated["matrix_scores"], evaluated["portfolio_scores"]
    summary, calibration = evaluated["summary"], evaluated["portfolio_summary"]
    if matrix.empty or portfolios.empty:
        raise RuntimeError("No scored out-of-sample horizons; review origins.csv")
    system_font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    family = "DejaVu Sans"
    if system_font.exists():
        font_manager.fontManager.addfont(str(system_font))
        family = font_manager.FontProperties(fname=str(system_font)).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white"})
    # Plot fixed EW weights: they are identical across estimators at each origin.
    p = portfolios.query("portfolio_family == 'equal_weight' and horizon == 21").copy()
    p["origin"] = pd.to_datetime(p.origin)
    actual = p.loc[p.method == "factor"].sort_values("origin")
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    ax.plot(actual.origin, np.sqrt(actual.realized_daily_second_moment * 21) * 100,
            color="#283b42", alpha=.75, linewidth=1.7, label="Next 21 sessions: realized risk proxy")
    for method in ["factor", "ewma_252", "ledoit_wolf_252"]:
        group = p.loc[p.method == method].sort_values("origin")
        ax.plot(group.origin, np.sqrt(group.forecast_daily_variance * 21) * 100,
                color=COLORS[method], linewidth=1.3, label=LABELS[method])
    ax.axvline(pd.Timestamp("2022-01-01"), color="#a5afb2", linestyle="--", linewidth=1)
    ax.set(title="Equal-weight portfolio: predicted versus subsequent risk", ylabel="21-session risk proxy (%)")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    path = figures / "predicted_vs_realized_risk.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    history_image = _embed(path)

    hold = calibration.query("split == 'holdout' and horizon == 21 and portfolio_family != 'minimum_variance'")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), constrained_layout=True)
    primary_methods = ["factor", "sample_252_jitter", "ewma_252", "ledoit_wolf_252"]
    shown = hold.loc[hold.method.isin(primary_methods), "aggregate_calibration_ratio"]
    padding = max(0.06, .13 * (shown.max() - shown.min()))
    limits = (max(0, min(1., shown.min()) - padding), max(1., shown.max()) + padding)
    for ax, family in zip(axes, FAMILIES):
        group = hold.loc[hold.portfolio_family == family].set_index("method")
        for i, method in enumerate(primary_methods):
            if method in group.index:
                ax.scatter(i, group.loc[method, "aggregate_calibration_ratio"], color=COLORS[method], s=65)
        ax.axhline(1, color="#263b42", linestyle="--", linewidth=1)
        ax.set_ylim(*limits)
        ax.set_xticks(range(len(primary_methods)), ["Factor", "Sample", "EWMA", "Ledoit–Wolf"], rotation=40, ha="right", fontsize=9)
        ax.set_title(FAMILIES[family], fontsize=11)
    axes[0].set_ylabel("Realized second moment / predicted variance")
    fig.suptitle("2022 onward, 21-session calibration: target = 1", fontsize=15)
    path = figures / "portfolio_calibration.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    calibration_image = _embed(path)

    # Preserve exact full matrices; figures show their easier-to-read correlations.
    available = [(day, entry) for day, entry in evaluated["forecasts"].items() if 21 in entry["realized"]]
    latest_day, latest = available[-1]
    universe = model["universe"].set_index("symbol")
    order = sorted(range(len(latest["symbols"])), key=lambda i: (universe.loc[latest["symbols"][i], "sector"], latest["symbols"][i]))
    symbols = [latest["symbols"][i] for i in order]
    forecast = latest["covariances"]["factor"][np.ix_(order, order)]
    observed = latest["realized"][21]
    realized = observed["centered_covariance"][np.ix_(order, order)]
    second = observed["second_moment"][np.ix_(order, order)]
    for filename, covariance in [("latest_forecast_covariance.csv", forecast),
                                  ("latest_realized_covariance.csv", realized),
                                  ("latest_realized_second_moment.csv", second),
                                  ("latest_covariance_difference.csv", forecast - realized)]:
        pd.DataFrame(covariance, index=symbols, columns=symbols).rename_axis("symbol").to_csv(output / filename)
    fc, rc = _correlation(forecast), _correlation(realized)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, values, title, limit in zip(axes, [fc, rc, fc - rc],
                                       ["Forecast correlation", "Next 21-day sample correlation", "Forecast minus realized"], [1, 1, 2]):
        heat = ax.imshow(values, vmin=-limit, vmax=limit, cmap="RdBu_r", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Stocks ordered by sector"); ax.set_ylabel("Stocks ordered by sector")
        ax.grid(False); fig.colorbar(heat, ax=ax, shrink=.75)
    fig.suptitle(f"Origin {latest_day} | N={len(symbols)} | realized centered rank <= 20", fontsize=14)
    path = figures / "forecast_realized_correlation.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    matrix_image = _embed(path)
    # Exact export metadata keeps the CSV pair's date, units and symbol order explicit.
    pair_meta = {"origin": latest_day, "horizon": 21, "n_stocks": len(symbols),
                 "future_start": str(observed["start_date"]), "future_end": str(observed["end_date"]),
                 "symbols": symbols, "units": "decimal daily covariance; centered realized sample ddof=1",
                 "forecast": "factor covariance, with any diagonal jitter recorded in corresponding matrices JSON",
                 "difference": "forecast minus centered realized sample covariance"}
    (output / "latest_matrix_pair.json").write_text(json.dumps(pair_meta, ensure_ascii=False, indent=2) + "\n")
    methods = summary.method.unique().tolist()
    matrix_table = summary.loc[summary.split == "holdout", ["horizon", "method", "n_origins", "median_stocks",
                                    "mean_relative_frobenius", "mean_correlation_rmse", "mean_gaussian_qlike_per_asset"]].copy()
    matrix_table.method = matrix_table.method.map(LABELS)
    calibration_table = calibration.loc[(calibration.split == "holdout") & calibration.portfolio_family.isin(FAMILIES),
        ["horizon", "method", "portfolio_family", "n_origins", "aggregate_calibration_ratio", "aggregate_centered_variance_ratio"]].copy()
    calibration_table.method = calibration_table.method.map(LABELS)
    calibration_table.portfolio_family = calibration_table.portfolio_family.map(FAMILIES)
    minvar = portfolios.query("split == 'holdout' and horizon == 21 and portfolio_family == 'minimum_variance'")
    minvar_summary = minvar.groupby("method").agg(n_origins=("origin", "nunique"),
        mean_actual_daily_second_moment=("realized_daily_second_moment", "mean"),
        mean_predicted_daily_variance=("forecast_daily_variance", "mean"), mean_gross=("gross_weight", "mean"))
    minvar_summary["actual_annualized_risk_proxy"] = np.sqrt(minvar_summary.mean_actual_daily_second_moment * 252)
    minvar_summary.to_csv(output / "minimum_variance_comparison.csv")
    origins = evaluated["origins"]
    scored = origins.query("status == 'scored' and horizon == 21")
    ex = model["exposures"]
    joined = ex.loc[ex.complete_required_exposures, ["asof_date", "symbol", "market_cap"]].merge(
        model["universe"][["symbol", "size_bucket"]], on="symbol")
    latest_caps = joined.loc[joined.asof_date == joined.asof_date.max()]
    cap_table = latest_caps.groupby("size_bucket").market_cap.agg(["count", "median", "min", "max"]) / [1, 1e9, 1e9, 1e9]
    cap_table.rename(columns={"count": "stocks", "median": "median_cap_bn", "min": "min_cap_bn", "max": "max_cap_bn"}, inplace=True)
    hold_eq = hold.loc[hold.portfolio_family == "equal_weight"].set_index("method").aggregate_calibration_ratio
    latest_ew = p.loc[(p.method == "factor") & (p.origin == pd.Timestamp(latest_day))].iloc[0]
    latest_predicted = np.sqrt(21 * latest_ew.forecast_daily_variance)
    latest_actual = np.sqrt(21 * latest_ew.realized_daily_second_moment)

    def html_table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")

    status_table = origins.groupby(["horizon", "status"]).size().rename("origins").reset_index()
    root_manifest = model["manifest"]
    text = f'''<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Equity covariance forecast evaluation</title>
<style>
body{{margin:0;background:#f3f6f6;color:#263940;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:40px 24px}}
h1{{font-size:36px;line-height:1.2}} h2{{font-size:24px}}
section{{background:white;border:1px solid #dce5e6;border-radius:12px;padding:24px;margin:20px 0}}
.note{{background:#e9f1f2;border-left:4px solid #176c78;padding:16px 20px}}
img{{max-width:100%;height:auto}} .scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:9px 11px;border-bottom:1px solid #e4ecec;text-align:left}}
th{{background:#edf3f3}} a{{color:#176c78}} select{{font:inherit;padding:7px}}
</style>
<main>
<h1>Equity covariance forecasts and realized risk</h1>
<p>{len(model['universe'])} candidate stocks · {len(scored)} scored monthly 21-session origins ·
{int(scored.n_exante.min())}–{int(scored.n_exante.max())} eligible stocks per origin · {len(methods)} estimators.</p>
<p class="note">The cohort uses the November 2026 holdings snapshot. Current membership and static sectors
introduce survivor and retrospective selection bias. Forecasts use data through each origin.
Financial filings must be strictly earlier than their exposure snapshot.</p>
<section>
<h2>Portfolio risk calibration</h2>
<p>2022-onward 21-session equal-weight calibration is {hold_eq['factor']:.3f} for the factor model,
{hold_eq['ewma_252']:.3f} for EWMA, and {hold_eq['ledoit_wolf_252']:.3f} for Ledoit–Wolf.
The ratio is the sum of realized daily second moments divided by the sum of predicted daily variances.
Its target is one. Values below one indicate aggregate variance overprediction.</p>
<img alt="Predicted and subsequent portfolio risk" src="{history_image}">
<p>Risk is displayed as sqrt(21 times average daily variance or second moment), in percent.
Fixed weights are applied to each daily excess return, implying daily rebalancing.</p>
<img alt="Portfolio variance calibration by estimator" src="{calibration_image}">
<p>The plot compares four cross-stock covariance estimators. Diagonal EWMA is included in the tables.</p>
<select id="horizon" aria-label="Evaluation horizon">
<option value="21">Next 21 trading sessions</option><option value="63">Next 63 trading sessions</option>
</select>
<div class="scroll" id="calibration-table">{html_table(calibration_table)}</div>
<p>Method-specific minimum-variance portfolios have a gross-weight limit of two.
Their weights differ across estimators. See the <a href="minimum_variance_comparison.csv">separate allocation comparison</a>.</p>
</section>
<section>
<h2>Matrix forecast errors</h2>
<div class="scroll">{html_table(matrix_table)}</div>
<p>Relative Frobenius error and correlation RMSE compare the forecast with centered future sample covariance.
Gaussian loss uses log det(forecast) + trace(forecast inverse times future second moment), divided by the number of stocks.
Lower scores are better. The conditional mean is assumed to be zero.</p>
<p>Centered realized matrices have rank at most 20 or 62 for these horizons. The evaluation inverts only forecasts.
Sample and EWMA precision losses can be very large near the 252-observation dimension limit.
Predeclared diagonal jitter is recorded for inverse-based calculations. Raw covariance errors use the original estimates.</p>
</section>
<section>
<h2>Latest complete matrix pair</h2>
<p>Origin {latest_day}, next-session window {str(pd.Timestamp(observed['start_date']).date())} to
{str(pd.Timestamp(observed['end_date']).date())}, {len(symbols)} stocks. Factor-model equal-weight risk was
{latest_predicted:.2%} predicted and {latest_actual:.2%} realized.</p>
<img alt="Forecast and realized correlations and their difference" src="{matrix_image}">
<p>Heatmaps show correlations with stocks sorted by sector. CSVs contain daily covariance in decimal-return-squared units.</p>
<p><a href="latest_forecast_covariance.csv">Forecast covariance</a> ·
<a href="latest_realized_covariance.csv">Realized covariance</a> ·
<a href="latest_covariance_difference.csv">Forecast minus realized</a> ·
<a href="latest_matrix_pair.json">Dates, units and stock order</a></p>
<p><a href="matrix_index.csv">All forecast dates and archive hashes</a>.
Complete archives are stored in matrices/YYYY-MM-DD.npz and contain forecasts, future observations and portfolio weights.</p>
</section>
<section>
<h2>Coverage and data integrity</h2>
<div class="scroll">{html_table(cap_table.reset_index())}</div>
<p>Daily factor estimation succeeded on {root_manifest['estimated_days']:,} sessions, with median cross-sectional size
{root_manifest['daily_stocks_median']:.0f}. The candidate cohort has 100 names in each current large/mid/small fund bucket.</p>
<div class="scroll">{html_table(status_table)}</div>
<p>A missing future return skips the entire frozen-stock horizon. Historical price or accounting gaps reduce ex-ante eligibility.
<a href="../market_identity_rules.json">Dated security-identity rules</a> separate reorganized CHRD equity, the listed TGTX history,
and the excluded SUPN source mismatch. Source returns remain uncapped.</p>
<p><a href="../data/processed/monthly_coverage.csv">Monthly exposure coverage</a> ·
<a href="input_audit.json">Input audit</a> · <a href="origins.csv">Forecast eligibility log</a></p>
</section>
<section>
<h2>Specification and interpretation</h2>
<p>Factor covariance uses up to 756 past sessions, exponential half-life 63 and 20% diagonal shrinkage.
Specific variances use 252 past residual sessions with at least 126 valid observations.
Direct estimators use 252 past stock-return sessions.</p>
<p>The chronological evaluation segment starts in 2022. Windows crossing its boundary are excluded from segment summaries.
Monthly windows can overlap. Current-survivor selection, public source revisions, accounting gaps and previous pilot analysis
limit prospective interpretation. Trading and financing costs are outside the portfolio diagnostics.</p>
<p><a href="summary.csv">Matrix summary</a> · <a href="portfolio_summary.csv">Portfolio summary</a> ·
<a href="evaluation_metadata.json">Metric definitions and settings</a> · <a href="experiment_summary.json">Run summary</a></p>
</section>
</main>
<script>
function filterHorizon(){{
  const selected = document.getElementById('horizon').value;
  document.querySelectorAll('#calibration-table tbody tr').forEach(row => {{
    row.style.display = row.cells[0].textContent.trim() === selected ? '' : 'none';
  }});
}}
document.getElementById('horizon').addEventListener('change', filterHorizon);
filterHorizon();
</script>
</html>'''
    path = output / "report.html"
    path.write_text(text)
    return path
