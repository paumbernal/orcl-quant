"""Point-in-time valuation-multiple history for ORCL.

For every trading day t the multiple uses only fundamentals whose ``avail_date`` <= t (``merge_asof``), so the
history is exactly what an investor could have computed on that day:

    P/E (GAAP, trailing)  = close_t / EPS_ttm
    EV / EBITDA (trailing) = (close_t x diluted shares + debt incl. finance leases - cash + preferred + minorities) / EBITDA_ttm
    EV / Revenue (trailing) = EV / revenue_ttm

Prices are the as-quoted (split-adjusted, NOT dividend-adjusted) closes - multiples must be computed on the price
an investor actually paid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def multiples_history(close: pd.Series, q: pd.DataFrame) -> pd.DataFrame:
    f = q.reset_index().rename(columns={"period_end": "period"})
    f = f[["period", "avail_date", "eps_diluted_ttm", "ebitda_ttm", "revenue_ttm", "total_debt_incl_leases",
           "cash_and_securities", "preferred_stock", "minority_interest", "diluted_shares"]].sort_values("avail_date")
    px = close.rename("close").reset_index().rename(columns={close.index.name or "index": "date"})
    px.columns = ["date", "close"]
    m = pd.merge_asof(px.sort_values("date"), f, left_on="date", right_on="avail_date", direction="backward").set_index("date")
    mcap = m["close"] * m["diluted_shares"]
    net_debt = m["total_debt_incl_leases"] - m["cash_and_securities"]
    ev = mcap + net_debt + m["preferred_stock"].fillna(0) + m["minority_interest"].fillna(0)
    out = pd.DataFrame({
        "close": m["close"],
        "market_cap": mcap,
        "enterprise_value": ev,
        "pe_trailing": (m["close"] / m["eps_diluted_ttm"]).where(m["eps_diluted_ttm"] > 0),
        "ev_ebitda": (ev / m["ebitda_ttm"]).where(m["ebitda_ttm"] > 0),
        "ev_revenue": ev / m["revenue_ttm"],
        "fundamentals_as_of": m["period"],
    })
    return out.dropna(subset=["pe_trailing", "ev_ebitda", "ev_revenue"], how="all")


def multiple_stats(hist: pd.DataFrame, years: int, percentiles: dict[str, float]) -> pd.DataFrame:
    """Current value, percentile anchors and the current percentile rank, over the last `years` years."""
    cutoff = hist.index[-1] - pd.DateOffset(years=years)
    h = hist[hist.index >= cutoff]
    rows = {}
    for col in ("pe_trailing", "ev_ebitda", "ev_revenue"):
        s = h[col].dropna()
        row = {"current": float(s.iloc[-1]), "history_start": s.index[0]}
        for name, p in percentiles.items():
            row[name] = float(s.quantile(p))
        row["current_percentile_rank"] = float((s <= s.iloc[-1]).mean())
        row["min"], row["max"] = float(s.min()), float(s.max())
        rows[col] = row
    return pd.DataFrame(rows).T
