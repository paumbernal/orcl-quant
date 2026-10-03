"""Point-in-time (PIT) daily feature panel shared by the factor model, the investment score and the signal.

Every column at date t is computable from information that was public at the close of t:

* price features use data up to and including t;
* fundamental features are taken from the latest quarter whose ``avail_date`` <= t (``merge_asof``), i.e. no quarter is
  used before the session that reflected its earnings release;
* "percentile" features are *expanding-window* ranks - the rank of today's value within all history up to today - so the
  normalisation itself never peeks at the future (a full-sample z-score or percentile would).

The truncation-invariance test in ``tests/test_pit.py`` checks this property directly: features computed on data
truncated at date T must equal the same rows computed on the full data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..analytics import returns as R
from ..config import Config
from ..data.store import Store
from ..valuation.multiples import multiples_history

QUARTERLY_FEATURES = [
    "revenue_yoy", "eps_ttm_yoy", "eps_diluted_yoy", "ocf_yoy", "oci_growth", "cloud_growth", "rpo_yoy", "rpo_to_ttm_revenue",
    "fcf_margin_ttm", "fcf_margin_yoy_change", "capex_to_ocf_ttm", "operating_margin_ttm", "gross_margin_proxy",
    "net_debt_to_ebitda_ttm", "interest_coverage_ttm", "eps_surprise_pct", "operating_income_yoy",
]


def expanding_pct_rank(s: pd.Series, min_periods: int) -> pd.Series:
    """Rank of each value among all values observed so far (including itself), in (0, 1]."""
    return s.expanding(min_periods=min_periods).rank(pct=True)


def quarterly_ranks(q: pd.DataFrame, cols: list[str], min_quarters: int) -> pd.DataFrame:
    """Expanding percentile ranks computed on the QUARTERLY series (one observation per release, not per day)."""
    out = pd.DataFrame(index=q.index)
    for c in cols:
        out[f"{c}__rank"] = expanding_pct_rank(q[c], min_quarters)
    return out


def asof_join(daily_index: pd.DatetimeIndex, q: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Attach quarterly columns to daily dates using the release-availability date (no look-ahead)."""
    f = q[cols].copy()
    f["avail_date"] = q["avail_date"]
    f = f.sort_values("avail_date").drop_duplicates("avail_date", keep="last")
    d = pd.DataFrame({"date": daily_index}).sort_values("date")
    m = pd.merge_asof(d, f, left_on="date", right_on="avail_date", direction="backward")
    m = m.set_index("date").drop(columns="avail_date")
    return m.reindex(daily_index)


def price_features(px: pd.DataFrame, cfg: Config, rf_annual: pd.Series) -> pd.DataFrame:
    t = cfg.project.ticker
    spy, qqq, smh = cfg.data.benchmarks.sp500, cfg.data.benchmarks.nasdaq100, cfg.data.benchmarks.semis
    rets = R.simple_returns(px)
    f = pd.DataFrame(index=px.index)
    f["close"] = px[t]
    for name, n in (("ret_1m", 21), ("ret_3m", 63), ("ret_6m", 126), ("ret_12m", 252)):
        f[name] = px[t].pct_change(n)
    f["mom_12_1"] = px[t].shift(21) / px[t].shift(252) - 1.0
    for n in cfg.analytics.vol_windows:
        f[f"vol_{n}d"] = R.rolling_vol(rets[t], n).reindex(px.index)
    f["beta_252d"] = R.rolling_beta(rets[t], rets[spy], cfg.analytics.rolling_beta_window).reindex(px.index)
    f["drawdown_52w"] = px[t] / px[t].rolling(252, min_periods=126).max() - 1.0
    f["price_vs_200dma"] = px[t] / px[t].rolling(200, min_periods=200).mean() - 1.0
    f["rel_strength_6m_vs_qqq"] = f["ret_6m"] - px[qqq].pct_change(126)
    f["spy_ret_3m"], f["qqq_ret_3m"], f["smh_ret_3m"] = px[spy].pct_change(63), px[qqq].pct_change(63), px[smh].pct_change(63)
    f["rf_3m"] = rf_annual.reindex(px.index).ffill()
    return f


def build_feature_panel(st: Store, cfg: Config) -> pd.DataFrame:
    """Daily PIT panel: price features + as-of fundamentals + valuation multiples + expanding percentile ranks."""
    px = st.prices()
    q = st.fundamentals()
    ev = st.events()
    rf = st.risk_free()["rf_3m"]
    f = price_features(px, cfg, rf)

    # ---- valuation multiples (trailing, PIT) ----
    mh = multiples_history(st.closes()[cfg.project.ticker], q)
    for c in ("pe_trailing", "ev_ebitda", "ev_revenue"):
        f[c] = mh[c].reindex(f.index)

    # ---- fundamentals (as-of avail_date) ----
    qq = q.copy()
    qq["eps_surprise_pct"] = qq["eps_surprise_pct"]
    cols = [c for c in QUARTERLY_FEATURES if c in qq.columns]
    fund = asof_join(f.index, qq, cols)
    f = f.join(fund)
    f["days_since_release"] = (f.index.to_series() - asof_join(f.index, qq.assign(_d=qq["release_date"]), ["_d"])["_d"]).dt.days

    # ---- expanding percentile ranks ----
    mq = cfg.score.min_history_quarters
    qr = quarterly_ranks(qq, cols, mq)
    qr["avail_date"] = qq["avail_date"]
    qr_daily = asof_join(f.index, qr, [c for c in qr.columns if c != "avail_date"])
    f = f.join(qr_daily)
    min_days = 3 * 252
    for c in ("mom_12_1", "price_vs_200dma", "rel_strength_6m_vs_qqq", "vol_63d", "beta_252d", "drawdown_52w",
              "pe_trailing", "ev_ebitda", "ev_revenue"):
        f[f"{c}__rank"] = expanding_pct_rank(f[c], min_days)
    return f
