"""Annual fundamentals of the hyperscaler peers (Yahoo Finance) - used ONLY to anchor capital-intensity and
margin ranges in the valuation, never as model inputs for ORCL itself.

Yahoo exposes about four fiscal years of annual statements per company. Fields kept:
revenue, EBITDA, operating income, net PP&E, capex. All are reported historical figures.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from .manifest import Manifest

log = logging.getLogger(__name__)
HYPERSCALERS = ("MSFT", "GOOGL", "AMZN")


def _pick(df: pd.DataFrame, row: str, col) -> float:
    try:
        v = df.loc[row, col]
        return float(v) if pd.notna(v) else np.nan
    except KeyError:
        return np.nan


def fetch_peer_fundamentals(tickers=HYPERSCALERS + ("NVDA",)) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        try:
            fin, bs, cf = tk.financials, tk.balance_sheet, tk.cashflow
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("peer %s unavailable: %s", t, exc)
            continue
        for col in fin.columns:
            rev = _pick(fin, "Total Revenue", col)
            capex = _pick(cf, "Capital Expenditure", col) if col in cf.columns else np.nan
            rows.append({
                "ticker": t, "fiscal_year_end": pd.Timestamp(col).normalize(),
                "revenue": rev, "ebitda": _pick(fin, "EBITDA", col), "operating_income": _pick(fin, "Operating Income", col),
                "net_ppe": _pick(bs, "Net PPE", col) if col in bs.columns else np.nan,
                "capex": -capex if pd.notna(capex) else np.nan,
            })
    df = pd.DataFrame(rows).dropna(subset=["revenue"])
    df["ppe_per_revenue"] = df["net_ppe"] / df["revenue"]
    df["capex_per_revenue"] = df["capex"] / df["revenue"]
    df["ebitda_margin"] = df["ebitda"] / df["revenue"]
    df["operating_margin"] = df["operating_income"] / df["revenue"]
    return df.sort_values(["ticker", "fiscal_year_end"]).reset_index(drop=True)


def run(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    df = fetch_peer_fundamentals()
    out = cfg.processed_dir / "peer_fundamentals.csv"
    df.to_csv(out, index=False)
    manifest.record(
        "peer_fundamentals", kind="HISTORICAL", file=out,
        source="Yahoo Finance annual statements (MSFT, GOOGL, AMZN, NVDA) - ~4 fiscal years each",
        notes="Anchors for terminal capital intensity (net PP&E / revenue) and margin ranges; not a forecast.",
    )
    return df


def latest_hyperscaler_capital_intensity(df: pd.DataFrame) -> pd.Series:
    """Latest-FY net PP&E / revenue for MSFT, GOOGL, AMZN."""
    sub = df[df["ticker"].isin(HYPERSCALERS)].sort_values("fiscal_year_end").groupby("ticker").tail(1)
    return sub.set_index("ticker")["ppe_per_revenue"]
