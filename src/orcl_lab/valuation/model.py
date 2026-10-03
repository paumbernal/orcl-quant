"""Operating model, DCF and multiple-based valuation.

Structure of the projection (all USD in $ millions, shares in millions)
------------------------------------------------------------------------
t = 0 is the latest reported balance sheet (end of FY27 Q1). Period 0 is the *stub* - the remaining 3 quarters of
FY27 (fraction 0.75 of a year) - followed by full fiscal years FY28 ... FY36. Net debt, net PP&E and working
capital are taken from the real balance sheet at t = 0, so nothing already reported is counted twice.

    revenue     OCI + SaaS + non-cloud segments.  FY27/FY28 are anchored to analyst consensus (low / mean / high by
                scenario); OCI is the residual so the segments always sum to the consensus total. FY29-31: OCI growth's
                excess over the terminal rate decays geometrically (persistence phi estimated from Oracle's own history);
                FY32-36: linear fade to the terminal growth rate.
    EBITDA      revenue x margin; margin moves linearly from today's level to a percentile of its own 10-year history.
    capex       stub: run-rate of the last four quarters. Afterwards the capex needed to hold net PP&E / revenue on a path
                from today's level to a terminal anchor (hyperscaler peers) - i.e. growth must be paid for in capital.
    D&A         depreciation = rate x (opening net PP&E + half of capex) + run-off of acquired-intangible amortisation.
    EBIT, tax   EBIT = EBITDA - D&A; tax rate converges to a long-run rate.   NOPAT = EBIT x (1 - t).
    NWC         ratio to revenue moves from today's (inflated by customer prepayments) to a historical percentile.
    UFCF        NOPAT + D&A - capex - change in NWC.      (Stock-based comp is NOT added back: it is a real cost.)
    financing   net debt rolls forward: ND_t = ND_(t-1) - UFCF + after-tax interest + dividends. No new share issuance is
                modelled (future raises at fair value would not change value per share; raises below fair value would hurt it).

Terminal value uses the value-driver form: growth g must be funded by reinvesting g / RONIC of NOPAT, where the return on
new investment RONIC = WACC + retention x (ROIC_T - WACC) (excess returns fade, they do not vanish and they do not last forever):
    FCF_(T+1) = NOPAT_T x (1+g) x (1 - g / RONIC)                  TV = FCF_(T+1) / (WACC - g)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .inputs import Anchors


@dataclass(frozen=True)
class Assumptions:
    name: str
    # --- revenue ---
    cons_rev_fy1: float
    cons_rev_fy2: float
    saas_g1: float
    other_g1: float
    phi: float
    g_term: float
    growth_shift: float = 0.0            # additive shift to OCI growth from FY29 on (sensitivity / reverse DCF)
    # --- margins, assets, working capital, tax ---
    margin_target: float = 0.40
    margin_shift: float = 0.0
    margin_years: int = 5
    k_target: float = 1.0
    k_years: int = 6
    dep_rate: float = 0.10
    maint_floor: float = 0.6
    ronic_retention: float = 0.5
    stub_capex: float = 0.0
    nwc_target: float = -0.12
    nwc_years: int = 3
    tax0: float = 0.13
    tax_lr: float = 0.18
    tax_years: int = 5
    # --- financing ---
    r_eff: float = 0.044
    r_marg: float = 0.068
    r_years: int = 5
    r_cash: float = 0.04              # yield earned on net cash if net debt turns negative
    # --- discounting ---
    wacc: float = 0.10
    ke: float = 0.11
    kd: float = 0.068
    beta: float = 1.15
    # --- multiples (trailing, from own history) ---
    pe: float = 25.0
    ev_ebitda: float = 15.0
    ev_revenue: float = 6.0
    explicit_years: int = 10

    def with_(self, **kw) -> "Assumptions":
        return replace(self, **kw)


def _frac_stub(A: Anchors) -> float:
    return (4 - A.elapsed_quarters) / 4.0


def project(A: Anchors, S: Assumptions) -> pd.DataFrame:
    """Annual projection table (index = fiscal-year label, first row is the FY27 stub)."""
    N = S.explicit_years
    stub = _frac_stub(A)
    fy0 = A.base_fy + 1
    labels = [f"FY{str(fy0 + i)[-2:]}" + ("E (Q2-Q4)" if i == 0 else "E") for i in range(N)]
    frac = np.array([stub] + [1.0] * (N - 1))

    # ---------------- revenue by segment ----------------
    s0 = A.seg_base
    saas, other, oci = np.zeros(N), np.zeros(N), np.zeros(N)
    totals = {0: S.cons_rev_fy1, 1: S.cons_rev_fy2}
    saas[0], other[0] = s0["saas"] * (1 + S.saas_g1), s0["other"] * (1 + S.other_g1)
    oci[0] = totals[0] - saas[0] - other[0]
    saas[1], other[1] = saas[0] * (1 + S.saas_g1), other[0] * (1 + S.other_g1)
    oci[1] = totals[1] - saas[1] - other[1]
    g_oci = np.zeros(N)
    g_oci[0], g_oci[1] = oci[0] / s0["oci"] - 1, oci[1] / oci[0] - 1
    g_unshifted = np.zeros(N)
    g_unshifted[1] = g_oci[1]
    for i in range(2, N):
        if i < 5:                                         # FY29-FY31: geometric decay of the excess over g_term
            g_unshifted[i] = S.g_term + S.phi * (g_unshifted[i - 1] - S.g_term)
        else:                                             # FY32-FY36: linear fade to g_term
            g31 = g_unshifted[4]
            g_unshifted[i] = g31 + (S.g_term - g31) * ((i - 4) / (N - 1 - 4))
        g_oci[i] = max(g_unshifted[i] + S.growth_shift, -0.5)
        oci[i] = oci[i - 1] * (1 + g_oci[i])
    g_saas1, g_oth1 = S.saas_g1, S.other_g1
    for i in range(2, N):
        w = (i - 1) / (N - 2)
        saas[i] = saas[i - 1] * (1 + g_saas1 + (S.g_term - g_saas1) * w)
        other[i] = other[i - 1] * (1 + g_oth1 + (S.g_term - g_oth1) * w)
    rev_full = oci + saas + other                          # full-fiscal-year revenue
    rev_flow = rev_full.copy()
    rev_flow[0] = rev_full[0] - A.cur_q_revenue            # stub = FY27 total less the quarter(s) already reported
    total_growth = np.r_[rev_full[0] / A.revenue_base - 1, rev_full[1:] / rev_full[:-1] - 1]

    # ---------------- EBITDA ----------------
    idx = np.arange(1, N + 1)
    margin = A.margin_now + (S.margin_target + S.margin_shift - A.margin_now) * np.minimum(idx / S.margin_years, 1.0)
    ebitda = rev_flow * margin

    # ---------------- capex, PP&E, depreciation ----------------
    capex, dep, ppe = np.zeros(N), np.zeros(N), np.zeros(N)
    r = S.dep_rate
    capex[0] = S.stub_capex
    dep[0] = r * (A.ppe_now + 0.5 * capex[0]) * stub
    ppe[0] = A.ppe_now + capex[0] - dep[0]
    k_start = ppe[0] / rev_full[0]
    k_path = np.zeros(N)
    k_path[0] = k_start
    for i in range(1, N):
        k_path[i] = k_start + (S.k_target - k_start) * min(i / S.k_years, 1.0)
        target = k_path[i] * rev_full[i]
        c = (target - (1 - r) * ppe[i - 1]) / (1 - 0.5 * r)
        c_floor = S.maint_floor * r * ppe[i - 1] / (1 - 0.5 * S.maint_floor * r)
        capex[i] = max(c, c_floor)
        dep[i] = r * (ppe[i - 1] + 0.5 * capex[i])
        ppe[i] = ppe[i - 1] + capex[i] - dep[i]
    end_year = stub + np.arange(N) * 1.0 - np.r_[0, np.zeros(N - 1)]       # years from t=0 to the end of each period
    end_year = np.cumsum(frac)
    mid_year = end_year - frac / 2
    amort = A.amort_ttm * np.clip(1 - mid_year / 5.0, 0, 1) * frac
    da = dep + amort
    ebit = ebitda - da

    # ---------------- tax, NOPAT, working capital, FCF ----------------
    tax = S.tax0 + (S.tax_lr - S.tax0) * np.minimum(idx / S.tax_years, 1.0)
    nopat = ebit * (1 - tax)
    nwc_ratio_now = A.nwc_now / A.ttm["revenue_ttm"]
    nwc_ratio = nwc_ratio_now + (S.nwc_target - nwc_ratio_now) * np.minimum(idx / S.nwc_years, 1.0)
    nwc = nwc_ratio * rev_full
    d_nwc = np.r_[nwc[0] - A.nwc_now, np.diff(nwc)]
    ufcf = nopat + da - capex - d_nwc

    # ---------------- financing, EPS, leverage ----------------
    rate = S.r_eff + (S.r_marg - S.r_eff) * np.minimum(idx / S.r_years, 1.0)
    nd = np.zeros(N)
    interest, dividends = np.zeros(N), A.dps * A.shares_m * frac
    prev = A.net_debt
    for i in range(N):
        interest[i] = (rate[i] if prev >= 0 else S.r_cash) * prev * frac[i]
        nd[i] = prev - ufcf[i] + interest[i] * (1 - tax[i]) + dividends[i]
        prev = nd[i]
    net_income = (ebit - interest) * (1 - tax)
    eps = net_income / A.shares_m
    ebitda_annual = ebitda / frac
    out = pd.DataFrame({
        "fraction_of_year": frac, "end_year": end_year, "mid_year": mid_year,
        "revenue_fy": rev_full, "revenue": rev_flow, "revenue_growth": total_growth,
        "oci_revenue": oci, "oci_growth": g_oci, "saas_revenue": saas, "other_revenue": other,
        "ebitda_margin": margin, "ebitda": ebitda, "dep": dep, "amort": amort, "d_and_a": da, "ebit": ebit,
        "operating_margin": ebit / rev_flow, "tax_rate": tax, "nopat": nopat,
        "capex": capex, "capex_to_revenue": capex / rev_flow, "ppe_end": ppe, "ppe_to_revenue": ppe / rev_full,
        "nwc": nwc, "d_nwc": d_nwc, "ufcf": ufcf, "ufcf_margin": ufcf / rev_flow,
        "interest": interest, "dividends": dividends, "net_debt_end": nd, "net_income": net_income, "eps": eps,
        "net_debt_to_ebitda": nd / ebitda_annual,
        "interest_cover": np.where(interest > 0, ebit / np.where(interest > 0, interest, 1.0), np.nan),
    }, index=labels)
    out.attrs["assumptions"] = S.name
    return out


# ------------------------------------------------------------------------------------------ DCF
@dataclass
class DCFResult:
    price: float
    enterprise_value: float
    equity_value: float
    pv_explicit: float
    pv_terminal: float
    terminal_value: float
    terminal_share_of_ev: float
    terminal_fcf: float
    terminal_roic: float
    implied_exit_ev_ebitda: float
    implied_exit_ev_revenue: float
    wacc: float
    g: float
    cumulative_fcf: float
    min_cumulative_fcf: float


def value_dcf(A: Anchors, S: Assumptions, proj: pd.DataFrame | None = None, wacc: float | None = None, g: float | None = None) -> DCFResult:
    P = proj if proj is not None else project(A, S)
    w = S.wacc if wacc is None else wacc
    gg = S.g_term if g is None else g
    if w <= gg + 0.005:
        raise ValueError(f"WACC {w:.3%} must exceed terminal growth {gg:.3%} by at least 0.5pp")
    df = (1 + w) ** -P["mid_year"].to_numpy()
    pv_explicit = float((P["ufcf"].to_numpy() * df).sum())
    last = P.iloc[-1]
    ic_T = last["ppe_end"] + last["nwc"]
    nopat_run = last["nopat"] / last["fraction_of_year"]                       # last period is a full year, so identical
    roic_T = nopat_run / ic_T if ic_T > 0 else np.nan
    # New investment earns WACC plus a retained share of today's excess return (RONIC); growth needs g / RONIC of NOPAT reinvested.
    ronic = w + S.ronic_retention * max((roic_T if np.isfinite(roic_T) else w) - w, 0.0)
    nopat_T1 = nopat_run * (1 + gg)
    fcf_T1 = nopat_T1 * (1 - gg / ronic)
    tv = fcf_T1 / (w - gg)
    t_tv = last["end_year"] - 0.5                                              # Gordon on mid-year flows
    pv_tv = float(tv * (1 + w) ** -t_tv)
    ev = pv_explicit + pv_tv
    eq = ev - A.net_debt - A.preferred - A.minority
    cum = P["ufcf"].cumsum()
    return DCFResult(
        price=eq / A.shares_m, enterprise_value=ev, equity_value=eq, pv_explicit=pv_explicit, pv_terminal=pv_tv,
        terminal_value=tv, terminal_share_of_ev=pv_tv / ev if ev > 0 else np.nan, terminal_fcf=fcf_T1,
        terminal_roic=float(roic_T) if np.isfinite(roic_T) else np.nan,
        implied_exit_ev_ebitda=float(tv / last["ebitda"]), implied_exit_ev_revenue=float(tv / last["revenue"]),
        wacc=w, g=gg, cumulative_fcf=float(cum.iloc[-1]), min_cumulative_fcf=float(cum.min()),
    )


def exit_multiple_value(A: Anchors, S: Assumptions, multiple: float, exit_offset: int = 4, proj: pd.DataFrame | None = None) -> float:
    """Price per share if the business is sold at `multiple` x EBITDA at the end of fiscal year index `exit_offset`."""
    P = proj if proj is not None else project(A, S)
    k = exit_offset
    df = (1 + S.wacc) ** -P["mid_year"].to_numpy()[: k + 1]
    pv = float((P["ufcf"].to_numpy()[: k + 1] * df).sum())
    ebitda_exit = P["ebitda"].iloc[k] / P["fraction_of_year"].iloc[k]
    pv_exit = multiple * ebitda_exit * (1 + S.wacc) ** -P["end_year"].iloc[k]
    return (pv + pv_exit - A.net_debt - A.preferred - A.minority) / A.shares_m


# ------------------------------------------------------------------------------------------ multiples
@dataclass
class MultipleValuation:
    forward_pe: float
    ev_ebitda: float
    ev_revenue: float
    ntm: dict[str, float] = field(default_factory=dict)


def ntm_metrics(A: Anchors, P: pd.DataFrame) -> dict[str, float]:
    """Next-twelve-months from t=0 = the stub (9 months) + one quarter of FY28."""
    stub = float(P["fraction_of_year"].iloc[0])
    w28 = 1.0 - stub
    ntm = {k: float(P[k].iloc[0] + w28 * P[k].iloc[1]) for k in ("revenue", "ebitda", "net_income", "eps", "dividends", "ufcf")}
    nd0, nd1, nd2 = A.net_debt, float(P["net_debt_end"].iloc[0]), float(P["net_debt_end"].iloc[1])
    ntm["net_debt_horizon"] = nd1 + w28 * (nd2 - nd1)
    return ntm


def value_multiples(A: Anchors, S: Assumptions, proj: pd.DataFrame | None = None) -> MultipleValuation:
    """12-month-ahead value = (trailing multiple x NTM metric) bridged to equity, then discounted at the cost of equity.

    A trailing multiple applied to the next-twelve-months metric is exactly the price one year from now (when NTM has
    become trailing). Net debt at the horizon reflects the cash burn in between - capex-driven debt reduces equity value.
    """
    P = proj if proj is not None else project(A, S)
    n = ntm_metrics(A, P)
    div = A.dps
    pe_px = S.pe * n["eps"]
    ps = lambda ev: (ev - n["net_debt_horizon"] - A.preferred - A.minority) / A.shares_m
    disc = lambda px: (px + div) / (1 + S.ke)
    mv = MultipleValuation(
        forward_pe=disc(pe_px) if n["eps"] > 0 else 0.0,
        ev_ebitda=disc(ps(S.ev_ebitda * n["ebitda"])),
        ev_revenue=disc(ps(S.ev_revenue * n["revenue"])),
        ntm=n,
    )
    return mv
