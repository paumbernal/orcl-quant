"""Risk-free rates from FRED (St. Louis Fed).

Uses the public CSV endpoint (no key needed). If ``FRED_API_KEY`` is set in the environment the
JSON API is used instead, which is more robust against endpoint changes.
"""
from __future__ import annotations

import io
import os

import pandas as pd

from ..config import Config
from .http import make_session
from .manifest import Manifest

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
FRED_API = "https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json&observation_start={start}"


def fetch_fred(series_id: str, start: str = "2009-01-01") -> pd.Series:
    # NB: FRED's CSV endpoint stalls on some custom User-Agent strings, so the library default is used.
    s = make_session()
    key = os.environ.get("FRED_API_KEY")
    if key:
        obs = s.get(FRED_API.format(sid=series_id, key=key, start=start), timeout=60).json()["observations"]
        df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"date": "observation_date"})
        df = df.set_index("observation_date")
    else:
        text = s.get(FRED_CSV.format(sid=series_id, start=start), timeout=30).text
        df = pd.read_csv(io.StringIO(text))
        df = df.rename(columns={df.columns[0]: "observation_date", df.columns[1]: "value"}).set_index("observation_date")
    out = pd.to_numeric(df["value"], errors="coerce")
    out.index = pd.to_datetime(out.index)
    out.name = series_id
    return out.dropna() / 100.0  # percent -> decimal annual rate


def build_riskfree(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    short, long = cfg.data.risk_free.short, cfg.data.risk_free.long
    start = str(cfg.data.start_date)
    raw = pd.concat([fetch_fred(short, start), fetch_fred(long, start)], axis=1)
    raw.index.name = "date"
    raw.to_csv(cfg.path("data", "raw", create=True) / "fred_rates.csv")
    prices = pd.read_csv(cfg.processed_dir / "prices_adj_close.csv", index_col=0, parse_dates=True)
    # Align to trading days; forward-fill holidays/weekends (the last published rate is the one in force).
    rf = raw.reindex(raw.index.union(prices.index)).ffill().reindex(prices.index).dropna()
    rf = rf.rename(columns={short: "rf_3m", long: "rf_10y"})
    out = cfg.processed_dir / "risk_free.csv"
    rf.to_csv(out)
    manifest.record(
        "risk_free",
        kind="HISTORICAL",
        source=f"FRED ({short}, {long}) - Board of Governors of the Federal Reserve System",
        file=out,
        df=rf,
        notes="Annualised decimal rates, forward-filled to trading days. rf_3m is used for Sharpe/Sortino/alpha, rf_10y for the DCF.",
    )
    return rf
