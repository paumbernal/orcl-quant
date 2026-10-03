"""Analyst-consensus snapshot (Yahoo Finance) - the ONLY forecast data in the repository.

Free sources do not provide *historical* consensus estimates, so a snapshot is taken at run time,
stamped with its retrieval date, stored as JSON, and appended to ``data/snapshots/consensus_history.csv``.
Re-running the pipeline over time therefore builds a genuine point-in-time revision history that
can be used by the signal model without look-ahead bias. Everything here is labelled ``CONSENSUS``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Config
from .manifest import Manifest, utc_now

log = logging.getLogger(__name__)

INFO_FIELDS = [
    "currentPrice", "marketCap", "enterpriseValue", "sharesOutstanding", "forwardEps", "trailingEps",
    "forwardPE", "trailingPE", "enterpriseToEbitda", "enterpriseToRevenue", "priceToSalesTrailing12Months",
    "beta", "targetMeanPrice", "targetHighPrice", "targetLowPrice", "recommendationMean", "numberOfAnalystOpinions",
    "totalDebt", "totalCash", "ebitda", "totalRevenue", "dividendYield", "payoutRatio",
]


def _frame_to_dict(df) -> dict:
    if df is None or getattr(df, "empty", True):
        return {}
    out = df.copy()
    out.index = out.index.astype(str)
    return json.loads(out.to_json(orient="index"))


def fetch_snapshot(cfg: Config) -> dict:
    import yfinance as yf

    t = cfg.project.ticker
    tk = yf.Ticker(t)
    snap: dict = {"ticker": t, "retrieved_at_utc": utc_now(), "kind": "CONSENSUS", "source": "Yahoo Finance (aggregated sell-side consensus)"}
    for name, attr in (("earnings_estimate", "earnings_estimate"), ("revenue_estimate", "revenue_estimate"),
                       ("eps_trend", "eps_trend"), ("eps_revisions", "eps_revisions"), ("growth_estimates", "growth_estimates")):
        try:
            snap[name] = _frame_to_dict(getattr(tk, attr))
        except Exception as exc:  # pragma: no cover - network dependent
            log.warning("%s unavailable: %s", name, exc)
            snap[name] = {}
    try:
        snap["price_targets"] = dict(tk.analyst_price_targets or {})
    except Exception as exc:  # pragma: no cover
        log.warning("price targets unavailable: %s", exc)
        snap["price_targets"] = {}
    try:
        info = tk.info
        snap["info"] = {k: info.get(k) for k in INFO_FIELDS}
    except Exception as exc:  # pragma: no cover
        log.warning("info unavailable: %s", exc)
        snap["info"] = {}
    try:
        ed = tk.get_earnings_dates(limit=12)
        fut = ed[ed.index > pd.Timestamp.now(tz=ed.index.tz)]
        snap["next_earnings_date"] = str(fut.index.min().date()) if len(fut) else None
        # Yahoo's '0q' / '+1q' estimate rows can lag a quarter after a release, so the next-quarter EPS estimate is read from the earnings calendar
        nxt = fut.sort_index().iloc[0] if len(fut) else None
        snap["next_quarter_eps_estimate"] = float(nxt["EPS Estimate"]) if nxt is not None and pd.notna(nxt["EPS Estimate"]) else None
    except Exception as exc:  # pragma: no cover
        log.warning("next earnings date unavailable: %s", exc)
        snap["next_earnings_date"] = None
    peers = {}
    for p in cfg.data.peers:
        try:
            pi = yf.Ticker(p).info
            peers[p] = {k: pi.get(k) for k in INFO_FIELDS}
        except Exception as exc:  # pragma: no cover
            log.warning("peer %s unavailable: %s", p, exc)
    snap["peers"] = peers
    return snap


def snapshot_to_rows(snap: dict) -> pd.DataFrame:
    """Flatten the estimate tables into long rows for the append-only history file."""
    date = snap["retrieved_at_utc"][:10]
    rows = []
    ee, re_, tr, rv = snap.get("earnings_estimate", {}), snap.get("revenue_estimate", {}), snap.get("eps_trend", {}), snap.get("eps_revisions", {})
    for period in ee:
        rows.append({
            "snapshot_date": date, "period": period,
            "eps_avg": ee[period].get("avg"), "eps_low": ee[period].get("low"), "eps_high": ee[period].get("high"),
            "eps_n_analysts": ee[period].get("numberOfAnalysts"), "eps_year_ago": ee[period].get("yearAgoEps"),
            "rev_avg": re_.get(period, {}).get("avg"), "rev_low": re_.get(period, {}).get("low"), "rev_high": re_.get(period, {}).get("high"),
            "eps_7d_ago": tr.get(period, {}).get("7daysAgo"), "eps_30d_ago": tr.get(period, {}).get("30daysAgo"),
            "eps_60d_ago": tr.get(period, {}).get("60daysAgo"), "eps_90d_ago": tr.get(period, {}).get("90daysAgo"),
            "up_30d": rv.get(period, {}).get("upLast30days"), "down_30d": rv.get(period, {}).get("downLast30days"),
        })
    return pd.DataFrame(rows)


def run(cfg: Config, manifest: Manifest) -> dict:
    snap = fetch_snapshot(cfg)
    out = cfg.processed_dir / "consensus_snapshot.json"
    out.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    hist = cfg.path("data", "snapshots", create=True) / "consensus_history.csv"
    new = snapshot_to_rows(snap)
    if hist.exists():
        old = pd.read_csv(hist)
        new = pd.concat([old[old["snapshot_date"] != new["snapshot_date"].iloc[0]], new], ignore_index=True) if len(new) else old
    new.to_csv(hist, index=False)
    manifest.record(
        "consensus_snapshot", kind="CONSENSUS", file=out,
        source="Yahoo Finance analyst consensus (forward EPS / revenue, revisions, price targets) - a dated SNAPSHOT, not history",
        notes="Forecasts are third-party opinions, not data. consensus_history.csv appends one row per period per run.",
    )
    return snap
