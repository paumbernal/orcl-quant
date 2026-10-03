"""Fundamentals assembly and derived ratios.

Inputs (all HISTORICAL): SEC XBRL quarterly table, earnings-release KPIs (cloud / OCI) and the earnings
event calendar. Output: one row per fiscal quarter with every metric requested in the brief:

    revenue, operating income, net income, EPS, gross margin (proxy), operating margin, operating cash
    flow, capex, FCF, total debt, cash, net debt, interest expense, shares, RPO, cloud revenue,
    OCI (cloud infrastructure) revenue, growth rates, leverage and coverage ratios.

Units: USD amounts in **$ millions**, shares in **millions**, EPS in USD/share, ratios as decimals.

``avail_date`` is the first trading session whose close reflects the quarter's release (the earnings
reaction day, or - where no 8-K matched - the filing date + 1 business day). Any strategy or score must
only use a quarter from ``avail_date`` onward; this is what keeps the point-in-time analysis free of
look-ahead bias.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..config import Config

MONEY_COLS = [
    "revenue", "operating_income", "net_income", "costs_and_expenses", "sales_marketing", "research_development",
    "general_admin", "restructuring", "acquisition_related", "amortization_intangibles_is", "interest_expense",
    "investment_income", "income_tax", "ocf", "capex", "depreciation", "sbc", "dividends_paid", "buybacks",
    "cash", "marketable_securities", "debt_current", "commercial_paper", "debt_noncurrent", "finance_lease_liability",
    "operating_lease_liability", "preferred_stock", "minority_interest", "total_assets", "total_equity",
    "current_assets", "current_liabilities", "receivables", "payables", "contract_liabilities", "ppe_net", "rpo",
]
SHARE_COLS = ["diluted_shares", "basic_shares", "shares_outstanding_cover"]


def _safe_growth(cur: pd.Series, prior: pd.Series) -> pd.Series:
    """YoY growth that is NaN (not nonsense) when the base is non-positive."""
    return (cur / prior - 1.0).where(prior > 0)


def build_fundamentals(q_xbrl: pd.DataFrame, events: pd.DataFrame, kpis: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    with warnings.catch_warnings():   # ~150 derived columns are added one at a time; fragmentation is harmless here
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _build_fundamentals(q_xbrl, events, kpis, cfg).copy()


def _build_fundamentals(q_xbrl: pd.DataFrame, events: pd.DataFrame, kpis: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    q = q_xbrl.copy()
    q[MONEY_COLS] = q[MONEY_COLS] / 1e6
    q[SHARE_COLS] = q[SHARE_COLS] / 1e6
    q["filed"] = pd.to_datetime(q["filed"])

    # ---------------- availability (point-in-time) ----------------
    ev = events.copy()
    ev.index = pd.to_datetime(ev.index)
    react = pd.to_datetime(ev["reaction_date"]) if "reaction_date" in ev else pd.Series(dtype="datetime64[ns]")
    q["avail_date"] = react.reindex(q.index)
    fallback = q["filed"] + pd.offsets.BDay(1)
    q["avail_date"] = q["avail_date"].fillna(fallback)
    q["release_date"] = pd.to_datetime(ev["release_date"]).reindex(q.index) if "release_date" in ev else pd.NaT

    # ---------------- KPIs from press releases ----------------
    k = kpis.copy()
    k.index = pd.to_datetime(k.index)
    for key, nice in (("cloud", "cloud"), ("iaas", "oci"), ("saas", "saas")):
        q[f"{nice}_revenue"] = k.get(f"pr_{key}_rev_bn", pd.Series(dtype=float)).reindex(q.index) * 1000.0
        q[f"{nice}_growth"] = k.get(f"pr_{key}_growth_pct", pd.Series(dtype=float)).reindex(q.index) / 100.0
        q[f"{nice}_source"] = k.get(f"pr_{key}_source", pd.Series(dtype=object)).reindex(q.index)
    for c in ("guid_rev_growth_lo", "guid_rev_growth_hi", "guid_cloud_growth_lo", "guid_cloud_growth_hi",
              "guid_eps_nongaap_lo", "guid_eps_nongaap_hi"):
        q[c] = ev[c].reindex(q.index) if c in ev else np.nan
    for c in ("eps_estimate", "eps_actual", "eps_surprise_pct"):
        q[c] = ev[c].reindex(q.index) if c in ev else np.nan
    q = q.rename(columns={"eps_actual": "eps_nongaap_actual", "eps_estimate": "eps_nongaap_consensus"})

    # ---------------- profitability ----------------
    q["d_and_a"] = q["depreciation"].fillna(0) + q["amortization_intangibles_is"].fillna(0)
    q["ebitda"] = q["operating_income"] + q["d_and_a"]
    q["cost_of_revenue_proxy"] = (q["costs_and_expenses"] - q["sales_marketing"] - q["research_development"]
                                  - q["general_admin"] - q["amortization_intangibles_is"].fillna(0)
                                  - q["restructuring"].fillna(0) - q["acquisition_related"].fillna(0))
    q["gross_margin_proxy"] = 1.0 - q["cost_of_revenue_proxy"] / q["revenue"]
    q["operating_margin"] = q["operating_income"] / q["revenue"]
    q["ebitda_margin"] = q["ebitda"] / q["revenue"]
    q["net_margin"] = q["net_income"] / q["revenue"]
    q["eps_diluted"] = q["eps_diluted"].astype(float)

    # ---------------- cash flow ----------------
    q["fcf"] = q["ocf"] - q["capex"]
    q["fcf_margin"] = q["fcf"] / q["revenue"]
    q["ocf_margin"] = q["ocf"] / q["revenue"]
    q["capex_to_ocf"] = (q["capex"] / q["ocf"]).where(q["ocf"] > 0)
    q["capex_to_revenue"] = q["capex"] / q["revenue"]
    q["capex_to_da"] = (q["capex"] / q["d_and_a"]).where(q["d_and_a"] > 0)

    # ---------------- trailing-twelve-month aggregates ----------------
    for c in ("revenue", "operating_income", "net_income", "ebitda", "ocf", "capex", "fcf", "interest_expense",
              "d_and_a", "income_tax", "sbc", "dividends_paid", "eps_diluted"):
        q[f"{c}_ttm"] = q[c].rolling(4, min_periods=4).sum()
    q["fcf_margin_ttm"] = q["fcf_ttm"] / q["revenue_ttm"]
    q["operating_margin_ttm"] = q["operating_income_ttm"] / q["revenue_ttm"]
    q["capex_to_ocf_ttm"] = (q["capex_ttm"] / q["ocf_ttm"]).where(q["ocf_ttm"] > 0)
    q["capex_to_revenue_ttm"] = q["capex_ttm"] / q["revenue_ttm"]
    pretax_ttm = q["net_income_ttm"] + q["income_tax_ttm"]
    q["effective_tax_rate_ttm"] = (q["income_tax_ttm"] / pretax_ttm).where(pretax_ttm > 0)

    # ---------------- balance sheet ----------------
    q["total_debt"] = q["debt_current"].fillna(0) + q["debt_noncurrent"].fillna(0) + q["commercial_paper"].fillna(0)
    q["total_debt_incl_leases"] = q["total_debt"] + q["finance_lease_liability"].fillna(0)
    q["cash_and_securities"] = q["cash"].fillna(0) + q["marketable_securities"].fillna(0)
    q["net_debt"] = q["total_debt"] - q["cash_and_securities"]
    q["net_debt_incl_leases"] = q["total_debt_incl_leases"] - q["cash_and_securities"]
    q["debt_to_ebitda_ttm"] = (q["total_debt"] / q["ebitda_ttm"]).where(q["ebitda_ttm"] > 0)
    q["net_debt_to_ebitda_ttm"] = (q["net_debt"] / q["ebitda_ttm"]).where(q["ebitda_ttm"] > 0)
    q["interest_coverage_ttm"] = (q["operating_income_ttm"] / q["interest_expense_ttm"]).where(q["interest_expense_ttm"] > 0)
    q["ebitda_interest_cover_ttm"] = (q["ebitda_ttm"] / q["interest_expense_ttm"]).where(q["interest_expense_ttm"] > 0)
    q["net_debt_to_equity"] = (q["net_debt"] / q["total_equity"]).where(q["total_equity"] > 0)
    q["nwc_proxy"] = q["receivables"] - q["payables"] - q["contract_liabilities"]

    # ---------------- growth rates (YoY) ----------------
    for c in ("revenue", "operating_income", "net_income", "ebitda", "ocf", "capex", "eps_diluted", "rpo", "ppe_net", "d_and_a"):
        q[f"{c}_yoy"] = _safe_growth(q[c], q[c].shift(4))
    q["revenue_ttm_yoy"] = _safe_growth(q["revenue_ttm"], q["revenue_ttm"].shift(4))
    q["eps_ttm_yoy"] = _safe_growth(q["eps_diluted_ttm"], q["eps_diluted_ttm"].shift(4))
    q["fcf_margin_yoy_change"] = q["fcf_margin_ttm"] - q["fcf_margin_ttm"].shift(4)
    q["fcf_ttm_change_yoy"] = q["fcf_ttm"] - q["fcf_ttm"].shift(4)   # $m; FCF is often negative so % growth is undefined
    q["rpo_qoq"] = _safe_growth(q["rpo"], q["rpo"].shift(1))
    q["rpo_to_ttm_revenue"] = q["rpo"] / q["revenue_ttm"]
    q["rpo_to_next_q_revenue_run_rate"] = q["rpo"] / (q["revenue"] * 4)
    q["cloud_share_of_revenue"] = q["cloud_revenue"] / q["revenue"]
    q["oci_share_of_revenue"] = q["oci_revenue"] / q["revenue"]
    # continuity: where the release gave no growth rate but levels exist for t and t-4, compute it
    for nice in ("cloud", "oci", "saas"):
        lvl = q[f"{nice}_revenue"]
        calc = _safe_growth(lvl, lvl.shift(4))
        q[f"{nice}_growth"] = q[f"{nice}_growth"].fillna(calc)
    q["incremental_oci_per_capex_lag4"] = (q["oci_revenue"].diff(4) / q["capex_ttm"].shift(4)).where(q["capex_ttm"].shift(4) > 0)

    q.index.name = "period_end"
    return q


def build_annual(a_xbrl: pd.DataFrame) -> pd.DataFrame:
    a = a_xbrl.copy()
    for c in MONEY_COLS:
        if c in a:
            a[c] = a[c] / 1e6
    for c in SHARE_COLS:
        if c in a:
            a[c] = a[c] / 1e6
    a["d_and_a"] = a["depreciation"].fillna(0) + a["amortization_intangibles_is"].fillna(0)
    a["ebitda"] = a["operating_income"] + a["d_and_a"]
    a["fcf"] = a["ocf"] - a["capex"]
    a["operating_margin"] = a["operating_income"] / a["revenue"]
    a["fcf_margin"] = a["fcf"] / a["revenue"]
    a["capex_to_ocf"] = (a["capex"] / a["ocf"]).where(a["ocf"] > 0)
    a["revenue_yoy"] = _safe_growth(a["revenue"], a["revenue"].shift(1))
    a["eps_yoy"] = _safe_growth(a["eps_diluted"], a["eps_diluted"].shift(1))
    a["capex_yoy"] = _safe_growth(a["capex"], a["capex"].shift(1))
    a["total_debt"] = a["debt_current"].fillna(0) + a["debt_noncurrent"].fillna(0) + a["commercial_paper"].fillna(0)
    a["cash_and_securities"] = a["cash"].fillna(0) + a["marketable_securities"].fillna(0)
    a["net_debt"] = a["total_debt"] - a["cash_and_securities"]
    a["net_debt_to_ebitda"] = (a["net_debt"] / a["ebitda"]).where(a["ebitda"] > 0)
    a["interest_coverage"] = (a["operating_income"] / a["interest_expense"]).where(a["interest_expense"] > 0)
    a.index.name = "period_end"
    return a
