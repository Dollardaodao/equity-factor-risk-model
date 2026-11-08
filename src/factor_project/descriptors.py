"""Monthly, historically available characteristic exposures for the pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import standardize_descriptors

STYLES = ["size", "momentum", "quality", "growth", "value", "low_vol", "beta_control"]


def session_returns(market: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Do not turn a missing quote/holiday gap into an apparent one-day return."""
    prices = market.pivot(index="date", columns="symbol", values="adj_close").reindex(sessions)
    return prices.pct_change(fill_method=None)


def _split_adjusted_shares(shares: float, filed, asof, prices: pd.DataFrame) -> float:
    """Share disclosures are assumed recast for splits already known at filing.

    Adjust only for subsequent splits. Applying every split after the financial
    period end would double-count splits between that end and a recast filing.
    SEC share counts remain lagged estimates, not daily CRSP shares outstanding.
    """
    if not np.isfinite(shares) or shares <= 0 or pd.isna(filed):
        return np.nan
    filed = pd.Timestamp(filed).normalize()
    if filed >= asof:
        raise ValueError("Share disclosure is not available before the exposure date")
    events = prices.loc[(prices.index > filed) & (prices.index <= asof), "split"]
    return float(shares * events.prod())


def _dividend_yield(prices: pd.DataFrame, asof, raw_close: float) -> float:
    if not np.isfinite(raw_close) or raw_close <= 0:
        return np.nan
    if prices.index.min() > asof - pd.Timedelta(days=365):
        return np.nan  # An incomplete year is not evidence of zero distributions.
    window = prices.loc[(prices.index > asof - pd.Timedelta(days=365)) & (prices.index <= asof)]
    cash = 0.0
    for day, dividend in window.loc[window.dividend != 0, "dividend"].items():
        later_splits = prices.loc[(prices.index > day) & (prices.index <= asof), "split"].prod()
        cash += float(dividend / later_splits)
    return cash / raw_close


def build_exposures(market: pd.DataFrame, fundamentals: pd.DataFrame,
                    universe: pd.DataFrame, benchmark: pd.DataFrame,
                    asof_dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return monthly exposures and raw descriptor/coverage audit.

    Composite component coverage is explicit. Quality requires >=3 of 4
    components, value >=2 of 3, and growth BOTH observed three-year measures.
    No missing descriptor is replaced with a neutral score. Leverage uses
    (assets-equity)/assets, explicitly a broad balance-sheet obligations proxy.
    Financials and REITs are excluded from this first accounting specification.
    """
    prices = market.copy()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    if prices.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate price date/symbol")
    sessions = pd.DatetimeIndex(benchmark.index).sort_values()
    returns = session_returns(prices, sessions)
    adjusted = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(sessions)
    rf, mkt = benchmark["RF"].reindex(sessions), benchmark["Mkt-RF"].reindex(sessions)
    momentum = adjusted.shift(21) / adjusted.shift(252) - 1
    volatility = returns.rolling(252, min_periods=200).std(ddof=1) * np.sqrt(252)
    beta = pd.DataFrame(index=sessions, columns=returns.columns, dtype=float)
    for symbol in returns:
        excess = returns[symbol] - rf
        # Pairwise market variance must use exactly the same observed days.
        paired_market = mkt.where(excess.notna())
        cov = excess.rolling(252, min_periods=200).cov(paired_market)
        var = paired_market.rolling(252, min_periods=200).var(ddof=1)
        beta[symbol] = cov / var.where(var > 0)
    fundamentals = fundamentals.copy()
    fundamentals["asof_date"] = pd.to_datetime(fundamentals.asof_date).dt.normalize()
    if fundamentals.duplicated(["asof_date", "symbol"]).any():
        raise ValueError("Duplicate fundamentals date/symbol")
    financials = fundamentals.set_index(["asof_date", "symbol"])
    sectors = universe.set_index("symbol").sector.to_dict()
    records = []
    by_symbol = {s: g.set_index("date").sort_index() for s, g in prices.groupby("symbol")}
    for asof in pd.DatetimeIndex(asof_dates):
        if asof not in sessions:
            raise ValueError("Exposure dates must be actual observed market sessions")
        for symbol, sector in sectors.items():
            row = {"asof_date": asof, "symbol": symbol, "sector": sector,
                   "raw_close": np.nan, "market_cap": np.nan,
                   "momentum_raw": np.nan, "low_vol_raw": np.nan,
                   "beta_control_raw": np.nan, "dividend_yield_raw": np.nan}
            if (asof, symbol) in financials.index:
                row.update(financials.loc[(asof, symbol)].to_dict())
            history = by_symbol.get(symbol)
            if history is not None and asof in history.index:
                close = float(history.loc[asof, "raw_close"])
                row["raw_close"] = close
                shares = _split_adjusted_shares(float(row.get("shares", np.nan)),
                                               row.get("shares_filed", pd.NaT), asof, history)
                row["shares_at_asof_estimate"] = shares
                row["market_cap"] = close * shares if close > 0 else np.nan
                row["dividend_yield_raw"] = _dividend_yield(history, asof, close)
                if symbol in returns:
                    row["momentum_raw"] = momentum.loc[asof, symbol]
                    row["low_vol_raw"] = -volatility.loc[asof, symbol]
                    row["beta_control_raw"] = beta.loc[asof, symbol]
            records.append(row)
    raw = pd.DataFrame(records)
    for field in ["assets", "equity", "net_income", "gross_profit", "operating_cashflow",
                  "revenue_3y_growth", "earnings_3y_change"]:
        if field not in raw:
            raw[field] = np.nan
        raw[field] = pd.to_numeric(raw[field], errors="coerce")
    assets = raw.assets.where(raw.assets > 0)
    cap = raw.market_cap.where(raw.market_cap > 0)
    raw["size_raw"] = -np.log(cap)
    raw["roa"] = raw.net_income / assets
    raw["gross_profitability"] = raw.gross_profit / assets
    raw["cash_earnings_quality"] = (raw.operating_cashflow - raw.net_income) / assets
    raw["negative_obligations"] = -(assets - raw.equity) / assets
    raw["book_to_price"] = raw.equity / cap
    raw["earnings_to_price"] = raw.net_income / cap
    raw["cashflow_to_price"] = raw.operating_cashflow / cap
    # Negative equity is not an interpretable cheap book-to-price observation.
    raw.loc[raw.equity <= 0, "book_to_price"] = np.nan
    composites = {
        "quality": (["roa", "gross_profitability", "cash_earnings_quality", "negative_obligations"], 3),
        "value": (["book_to_price", "earnings_to_price", "cashflow_to_price"], 2),
        "growth": (["revenue_3y_growth", "earnings_3y_change"], 2),
    }
    output, audit = [], []
    for asof, group in raw.groupby("asof_date", sort=True):
        group = group.copy()
        components = [c for cols, _ in composites.values() for c in cols]
        transformed = standardize_descriptors(group, components)
        for factor, (columns, minimum) in composites.items():
            n_components = transformed[columns].notna().sum(axis=1)
            group[f"{factor}_n_components"] = n_components
            group[f"{factor}_raw"] = transformed[columns].mean(axis=1).where(n_components >= minimum)
        for style in STYLES:
            group[style] = group[f"{style}_raw"]
        group["dividend_yield"] = group.dividend_yield_raw
        # Use one common estimation universe for centering/scaling ALL styles.
        # Rows without complete required inputs remain in the audit but aren't
        # permitted to affect the final model's exposure normalization.
        required = ["market_cap", *STYLES]
        complete = group[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        complete &= group.market_cap > 0
        normalized = standardize_descriptors(group.loc[complete], [*STYLES, "dividend_yield"])
        for style in [*STYLES, "dividend_yield"]:
            group[style] = np.nan
            group.loc[complete, style] = normalized[style]
        group["complete_required_exposures"] = complete
        output.append(group)
        audit.append({"asof_date": asof, "n_universe": len(group), "n_complete": int(complete.sum()),
                      **{f"available_{s}": int(group[f"{s}_raw"].notna().sum()) for s in STYLES},
                      "n_missing_shares": int(group.get("shares", pd.Series(np.nan, index=group.index)).isna().sum()),
                      "n_stale_shares": int((pd.to_numeric(group.get("shares_age_days", pd.Series(np.nan, index=group.index)), errors="coerce") > 180).sum()),
                      "normalization": json.dumps(normalized.attrs["standardization_report"], default=str)})
    return pd.concat(output, ignore_index=True), pd.DataFrame(audit)


def save_exposures(exposures: pd.DataFrame, audit: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    exposures.to_csv(directory / "monthly_exposures.csv", index=False)
    audit.to_csv(directory / "monthly_coverage.csv", index=False)
