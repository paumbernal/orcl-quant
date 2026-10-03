"""RPO -> Revenue -> Cash-flow conversion framework  (the central question of the project).

    Contracted backlog (RPO)  ->  recognised revenue (cloud / OCI)  ->  operating income  ->  operating cash flow
                                                                          ->  free cash flow (after capex)  ->  EPS

For each link we ask: is the growth at the top of the chain *arriving* at the bottom, how quickly, and at what
capital cost? All statistics are descriptive. With roughly 28 quarters of RPO history and one structural break
(the step-change in RPO in FY26 Q1) none of the estimates below can establish causality - see ``LIMITATIONS``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

LIMITATIONS = [
    "Small sample: RPO is only reported (in XBRL) from FY19 Q1, so growth-rate relationships rest on roughly 28 quarters.",
    "Structural break: RPO rose by >$300bn in a single quarter (FY26 Q1). A handful of observations dominates every correlation.",
    "RPO is contracted, not recognised: it can be cancelled/renegotiated, depends on customers' financing and on Oracle delivering "
    "data-centre capacity, and its timing profile is only partly disclosed.",
    "Customer concentration: a small number of very large AI-infrastructure contracts can account for a large share of RPO; "
    "this cannot be verified from public XBRL data.",
    "OCI revenue is disclosed only in the earnings releases (rounded to $0.1bn before FY26; exact in the FY26+ tables), starting FY23 Q1.",
    "Overlapping year-on-year growth rates are strongly autocorrelated; p-values and confidence intervals are therefore adjusted "
    "with an effective-sample-size correction, and still should be read as indicative.",
    "Correlation and Granger-style 'predictive precedence' are NOT causal evidence.",
]


# ---------------------------------------------------------------------------- helpers
def effective_n(x: pd.Series, y: pd.Series) -> float:
    """Bartlett-style effective sample size for the correlation of two autocorrelated series."""
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 6:
        return float(n)
    rx = df.iloc[:, 0].autocorr(1)
    ry = df.iloc[:, 1].autocorr(1)
    prod = float(np.clip(rx * ry, -0.95, 0.95))
    return max(4.0, n * (1 - prod) / (1 + prod))


def corr_with_ci(x: pd.Series, y: pd.Series, method: str = "spearman") -> dict[str, float]:
    df = pd.concat([x, y], axis=1).dropna()
    n = len(df)
    if n < 6:
        return {"corr": np.nan, "n": n, "n_eff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_adj": np.nan}
    r = df.iloc[:, 0].corr(df.iloc[:, 1], method=method)
    neff = effective_n(df.iloc[:, 0], df.iloc[:, 1])
    se = 1.0 / np.sqrt(max(neff - 3.0, 1.0))
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    p = 2 * (1 - stats.norm.cdf(abs(z) / se))
    return {"corr": float(r), "n": n, "n_eff": float(neff), "ci_low": float(lo), "ci_high": float(hi), "p_adj": float(p)}


@dataclass
class RPOAnalysis:
    table: pd.DataFrame
    growth_corr_spearman: pd.DataFrame
    growth_corr_pearson: pd.DataFrame
    growth_corr_post_break: pd.DataFrame
    lead_lag: pd.DataFrame
    granger: pd.DataFrame
    conversion: pd.DataFrame
    scorecard: pd.DataFrame
    coverage: dict[str, float]
    findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=lambda: list(LIMITATIONS))


# ---------------------------------------------------------------------------- main
GROWTH_VARS = {
    "rpo_yoy": "RPO growth",
    "oci_growth": "OCI growth",
    "cloud_growth": "Cloud growth",
    "revenue_yoy": "Revenue growth",
    "operating_income_yoy": "Operating income growth",
    "eps_diluted_yoy": "EPS growth (GAAP)",
    "ocf_yoy": "Operating cash flow growth",
    "capex_yoy": "Capex growth",
}


def growth_matrix(q: pd.DataFrame, cols: dict[str, str] | None = None) -> pd.DataFrame:
    cols = cols or GROWTH_VARS
    g = q[list(cols)].copy().rename(columns=cols)
    g["FCF margin change (TTM, pp)"] = q["fcf_margin_yoy_change"]
    return g


def correlation_table(g: pd.DataFrame, method: str) -> pd.DataFrame:
    return g.corr(method=method, min_periods=8)


def lead_lag_table(q: pd.DataFrame, lead: str = "rpo_yoy", lagged: str = "revenue_yoy", max_lag: int = 8, method: str = "spearman") -> pd.DataFrame:
    """corr( lead_{t-k}, lagged_t ) for k = 0..max_lag quarters, with effective-N confidence intervals."""
    rows = []
    for k in range(0, max_lag + 1):
        res = corr_with_ci(q[lead].shift(k), q[lagged], method)
        rows.append({"lag_quarters": k, **res})
    return pd.DataFrame(rows).set_index("lag_quarters")


def granger_table(q: pd.DataFrame, cause: str = "rpo_qoq", effect: str = "revenue_yoy", maxlag: int = 3) -> pd.DataFrame:
    """Does past `cause` help predict `effect` beyond past `effect`?  (F-test; predictive precedence only)."""
    from statsmodels.tsa.stattools import grangercausalitytests

    df = q[[effect, cause]].dropna()
    df = df.diff().dropna()                       # differencing -> closer to stationarity
    if len(df) < 4 * maxlag + 6:
        return pd.DataFrame(columns=["lag", "F", "p_value", "n"])
    out = grangercausalitytests(df, maxlag=maxlag)
    rows = [{"lag": l, "F": res[0]["ssr_ftest"][0], "p_value": res[0]["ssr_ftest"][1], "n": len(df)} for l, res in out.items()]
    return pd.DataFrame(rows).set_index("lag")


def forward_conversion(q: pd.DataFrame) -> pd.DataFrame:
    """Realised revenue in the following 4 / 8 quarters as a share of RPO at date t (a *lower bound* on conversion,
    since total revenue also includes non-RPO transactional revenue and RPO may be recognised beyond 8 quarters)."""
    rev = q["revenue"]
    fwd4 = sum(rev.shift(-i) for i in range(1, 5))
    fwd8 = sum(rev.shift(-i) for i in range(1, 9))
    out = pd.DataFrame({
        "rpo": q["rpo"],
        "rpo_pct_next_12m_disclosed": q.get("rpo_pct_next_12m"),
        "fwd4q_revenue": fwd4.where(rev.shift(-4).notna()),
        "fwd8q_revenue": fwd8.where(rev.shift(-8).notna()),
    })
    out["fwd4q_revenue_pct_of_rpo"] = out["fwd4q_revenue"] / out["rpo"]
    out["fwd8q_revenue_pct_of_rpo"] = out["fwd8q_revenue"] / out["rpo"]
    return out


def scorecard(q: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """Latest quarters side by side: does growth at the top of the chain reach the bottom?"""
    cols = {
        "rpo_yoy": "RPO growth", "oci_growth": "OCI growth", "revenue_yoy": "Revenue growth",
        "operating_income_yoy": "Operating income growth", "ocf_yoy": "Operating cash flow growth",
        "eps_diluted_yoy": "EPS growth (GAAP)", "fcf_margin_ttm": "FCF margin (TTM)", "capex_to_ocf_ttm": "Capex / OCF (TTM)",
        "operating_margin": "Operating margin", "gross_margin_proxy": "Gross margin (proxy)",
    }
    t = q[list(cols)].tail(n).rename(columns=cols)
    t.index = q["label"].tail(n)
    return t.T


def coverage_metrics(q: pd.DataFrame, consensus: dict | None = None) -> dict[str, float]:
    last = q.iloc[-1]
    ttm = float(last["revenue_ttm"])
    out = {
        "rpo_bn": float(last["rpo"]) / 1e3,
        "rpo_to_ttm_revenue_x": float(last["rpo"] / ttm),
        "years_of_revenue_at_run_rate": float(last["rpo"] / (last["revenue"] * 4)),
        "rpo_yoy": float(last["rpo_yoy"]),
        "oci_share_of_revenue": float(last["oci_share_of_revenue"]) if pd.notna(last.get("oci_share_of_revenue")) else np.nan,
    }
    if consensus:
        rv = consensus.get("revenue_estimate", {})
        fy0, fy1 = rv.get("0y", {}).get("avg"), rv.get("+1y", {}).get("avg")
        if fy0 and fy1:
            out["consensus_fy0_revenue_bn"] = fy0 / 1e9
            out["consensus_fy1_revenue_bn"] = fy1 / 1e9
            out["consensus_two_year_revenue_pct_of_rpo"] = (fy0 + fy1) / (last["rpo"] * 1e6)
            out["consensus_fy0_revenue_pct_of_rpo"] = fy0 / (last["rpo"] * 1e6)
    disclosed = q["rpo_pct_next_12m"].dropna() if "rpo_pct_next_12m" in q else pd.Series(dtype=float)
    if len(disclosed):
        out["disclosed_12m_conversion_median"] = float(disclosed.median())
        out["disclosed_12m_conversion_last_date"] = str(disclosed.index[-1].date())
    return out


def cash_conversion(q: pd.DataFrame) -> pd.DataFrame:
    """How much of each revenue dollar survives as profit and cash, and what growth cost in capex."""
    t = pd.DataFrame({
        "revenue_ttm": q["revenue_ttm"],
        "operating_margin_ttm": q["operating_margin_ttm"],
        "incremental_op_margin_yoy": (q["operating_income_ttm"].diff(4) / q["revenue_ttm"].diff(4)).where(q["revenue_ttm"].diff(4) > 0),
        "ocf_to_net_income_ttm": (q["ocf_ttm"] / q["net_income_ttm"]).where(q["net_income_ttm"] > 0),
        "fcf_margin_ttm": q["fcf_margin_ttm"],
        "capex_to_revenue_ttm": q["capex_to_revenue_ttm"],
        "capex_to_ocf_ttm": q["capex_to_ocf_ttm"],
        "capex_to_da": (q["capex_ttm"] / q["d_and_a_ttm"]).where(q["d_and_a_ttm"] > 0),
        "oci_rev_per_capex_lag4": q["incremental_oci_per_capex_lag4"],
    })
    return t


def analyse(q: pd.DataFrame, consensus: dict | None = None, break_label: str = "FY26 Q1") -> RPOAnalysis:
    q = q.copy()
    g = growth_matrix(q)
    brk = q.index[q["label"] == break_label]
    post = g.loc[brk[0]:] if len(brk) else g
    lead_lag = lead_lag_table(q)
    gr = granger_table(q)
    conv = forward_conversion(q)
    sc = scorecard(q)
    cov = coverage_metrics(q, consensus)
    a = RPOAnalysis(
        table=q[["label", "rpo", "rpo_yoy", "rpo_qoq", "rpo_to_ttm_revenue", "revenue", "revenue_yoy", "oci_revenue", "oci_growth",
                 "cloud_revenue", "cloud_growth", "operating_income", "ocf", "capex", "fcf", "eps_diluted"]],
        growth_corr_spearman=correlation_table(g, "spearman"),
        growth_corr_pearson=correlation_table(g, "pearson"),
        growth_corr_post_break=post.corr(method="spearman", min_periods=4),
        lead_lag=lead_lag,
        granger=gr,
        conversion=conv,
        scorecard=sc,
        coverage=cov,
    )
    a.findings = _findings(q, a)
    return a


def _findings(q: pd.DataFrame, a: RPOAnalysis) -> list[str]:
    last = q.iloc[-1]
    f = []
    f.append(f"RPO is ${last['rpo']/1e3:,.0f}bn, {last['rpo_to_ttm_revenue']:.1f}x trailing-twelve-month revenue "
             f"(vs {q['rpo_to_ttm_revenue'].iloc[-6]:.1f}x six quarters earlier); at the current quarterly run-rate it equals "
             f"{a.coverage['years_of_revenue_at_run_rate']:.1f} years of revenue.")
    f.append(f"Revenue growth has accelerated to {last['revenue_yoy']:.0%} YoY and OCI growth to {last['oci_growth']:.0%}"
             if pd.notna(last.get("oci_growth")) else f"Revenue growth is {last['revenue_yoy']:.0%} YoY.")
    f.append(f"Operating income grew {last['operating_income_yoy']:.0%} YoY and operating cash flow {last['ocf_yoy']:.0%} YoY, "
             f"but trailing FCF margin is {last['fcf_margin_ttm']:.0%} because capex is {last['capex_to_ocf_ttm']:.1f}x operating cash flow.")
    ll = a.lead_lag
    if ll["corr"].notna().any():
        best = ll["corr"].idxmax()
        f.append(f"Lead-lag: the highest rank-correlation between RPO growth and later revenue growth is at a lag of {best} quarter(s) "
                 f"(rho={ll.loc[best,'corr']:.2f}, 95% CI {ll.loc[best,'ci_low']:.2f} to {ll.loc[best,'ci_high']:.2f}, n={int(ll.loc[best,'n'])}, "
                 f"n_eff={ll.loc[best,'n_eff']:.0f}) - {'the interval excludes zero' if ll.loc[best,'ci_low']>0 else 'the interval includes zero, so the lag is not statistically distinguishable'}.")
    c4 = a.conversion["fwd4q_revenue_pct_of_rpo"].dropna()
    if len(c4):
        f.append(f"Historically the next four quarters of revenue equalled {c4.iloc[-1]:.0%} of RPO at {c4.index[-1].date()} "
                 f"(median {c4.median():.0%} over {len(c4)} observations); the ratio mechanically collapses when RPO jumps, so it measures "
                 f"backlog size relative to revenue, not conversion speed.")
    if "disclosed_12m_conversion_median" in a.coverage and "consensus_fy0_revenue_pct_of_rpo" in a.coverage:
        f.append(f"Oracle's own XBRL disclosure put the share of RPO recognised within 12 months at ~{a.coverage['disclosed_12m_conversion_median']:.0%} "
                 f"(last disclosed {a.coverage['disclosed_12m_conversion_last_date']}). Today, even if EVERY dollar of consensus current-year revenue came from RPO, "
                 f"12-month conversion could be at most {a.coverage['consensus_fy0_revenue_pct_of_rpo']:.0%} - the backlog is far longer-dated than the legacy book.")
    if "consensus_two_year_revenue_pct_of_rpo" in a.coverage:
        f.append(f"Consensus revenue for the next two fiscal years sums to {a.coverage['consensus_two_year_revenue_pct_of_rpo']:.0%} of current RPO "
                 f"(a forecast, not data): the market is not pricing conversion of most of the backlog within two years.")
    return f
