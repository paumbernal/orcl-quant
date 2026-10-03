"""Scenario construction, valuation runs, sensitivity tables and the reverse DCF.

Scenario definitions (what each case means, in words - the numbers come from `inputs.py`):

    BEAR  analyst-LOW revenue; OCI growth reverts quickly (pre-AI persistence); terminal EBITDA margin at the 10th
          percentile of own history; net PP&E/revenue never improves (capex stays elevated); higher depreciation rate
          and beta; multiples at the 25th percentile (valuation contracts).
    BASE  analyst-MEAN revenue; mid-regime persistence; median margin; capital intensity converges to the most
          intensive hyperscaler; median multiples.
    BULL  analyst-HIGH revenue; persistent OCI growth (full-sample AI-regime persistence); margin holds at today's
          record level; capital intensity converges to the hyperscaler median; lower beta; multiples at the 75th pct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from ..config import Config
from ..data.store import Store
from .inputs import SCEN, Anchors, build_anchors
from .model import (Assumptions, DCFResult, MultipleValuation, exit_multiple_value, project, value_dcf, value_multiples)

METHODS = ("dcf", "forward_pe", "ev_ebitda", "ev_revenue")
METHOD_LABELS = {"dcf": "DCF", "forward_pe": "Forward P/E", "ev_ebitda": "EV/EBITDA", "ev_revenue": "EV/Revenue"}


@dataclass
class ScenarioResult:
    name: str
    assumptions: Assumptions
    projection: pd.DataFrame
    dcf: DCFResult
    multiples: MultipleValuation
    method_prices: dict[str, float]
    blended: float
    diagnostics: dict[str, float]


@dataclass
class ValuationRun:
    anchors: Anchors
    scenarios: dict[str, ScenarioResult]
    summary: pd.DataFrame
    prob_weighted: float
    upside_prob_weighted: float
    reverse: dict
    sensitivities: dict[str, pd.DataFrame]
    derivations: pd.DataFrame
    peer_multiples: pd.DataFrame
    current_multiples: pd.DataFrame


def cost_of_capital(A: Anchors, cfg: Config, scenario: str) -> tuple[float, float, float, float]:
    """(beta, cost of equity, pre-tax cost of debt, WACC) for a scenario."""
    v = cfg.valuation
    betas = sorted(A.betas.values())
    beta = {"bear": betas[-1], "base": float(np.median(betas)), "bull": betas[0]}[scenario]
    ke = A.rf10 + beta * v.equity_risk_premium
    kd = A.rf10 + v.credit_spread
    E = A.price * A.shares_m
    D = A.gross_debt + A.preferred
    wD = D / (D + E)
    wacc = (1 - wD) * ke + wD * kd * (1 - v.long_run_tax_rate)
    return beta, ke, kd, wacc


def build_assumptions(A: Anchors, cfg: Config) -> dict[str, Assumptions]:
    v = cfg.valuation
    out = {}
    for s in SCEN:
        beta, ke, kd, wacc = cost_of_capital(A, cfg, s)
        low_high = {"bear": "low", "base": "avg", "bull": "high"}[s]
        mult = A.mult_stats
        out[s] = Assumptions(
            name=s,
            cons_rev_fy1=A.cons["fy1"][f"rev_{low_high}"], cons_rev_fy2=A.cons["fy2"][f"rev_{low_high}"],
            saas_g1=A.seg_growth_stage1["saas"], other_g1=A.seg_growth_stage1["other"],
            phi=A.phi[s], g_term=float(v.terminal_growth[s]),
            margin_target=A.margin_targets[s], margin_years=int(v.margin_convergence_years),
            k_target=A.k_targets[s], k_years=int(v.capital_intensity_convergence_years),
            dep_rate=A.dep_rate * float(v.depreciation_rate_multiplier[s]), maint_floor=float(v.maintenance_capex_floor),
            ronic_retention=float(v.terminal_excess_return_retention), r_cash=A.rf3m,
            stub_capex=A.avg_q_capex_4 * (4 - A.elapsed_quarters),
            nwc_target=A.nwc_targets[s], tax0=A.tax_now, tax_lr=A.tax_lr, tax_years=int(v.tax_normalisation_years),
            r_eff=A.eff_interest_rate, r_marg=A.rf10 + float(v.credit_spread),
            wacc=wacc, ke=ke, kd=kd, beta=beta,
            pe=float(mult.loc["pe_trailing", s]), ev_ebitda=float(mult.loc["ev_ebitda", s]), ev_revenue=float(mult.loc["ev_revenue", s]),
            explicit_years=int(v.explicit_years),
        )
    return out


def _diagnostics(A: Anchors, S: Assumptions, P: pd.DataFrame, cfg: Config) -> dict[str, float]:
    stress = cfg.valuation.stress
    cum = P["ufcf"].cumsum()
    neg_years = int((P["ufcf"] < 0).sum())
    first_pos = next((i for i, x in enumerate(P["ufcf"]) if x > 0), None)
    lev = P["net_debt_to_ebitda"]
    breach = lev[lev > stress.net_debt_to_ebitda_max]
    cloud = P["oci_revenue"] + P["saas_revenue"]                        # full-fiscal-year cloud revenue
    cloud_future_to_fy31 = float(cloud.iloc[0] - A.cur_q_cloud + cloud.iloc[1:5].sum())   # not-yet-reported cloud revenue, FY27 Q2 - FY31
    rpo = A.rpo
    return {
        "negative_fcf_years": neg_years,
        "first_positive_fcf_year": P.index[first_pos] if first_pos is not None else "none",
        "max_cumulative_funding_gap": float(-cum.min()) if cum.min() < 0 else 0.0,
        "peak_net_debt": float(P["net_debt_end"].max()),
        "peak_net_debt_to_ebitda": float(lev.max()),
        "first_year_leverage_breach": breach.index[0] if len(breach) else "none",
        "min_interest_cover": float(P["interest_cover"].min(skipna=True)),
        "interest_cover_breach": bool((P["interest_cover"].dropna() < stress.interest_cover_min).any()),
        "cum_cloud_revenue_to_fy31_pct_of_rpo": cloud_future_to_fy31 / rpo,
        "fy31_revenue": float(P["revenue_fy"].iloc[4]),
        "fy36_revenue": float(P["revenue_fy"].iloc[-1]),
        "revenue_cagr_to_fy31": float((P["revenue_fy"].iloc[4] / A.revenue_base) ** (1 / 5) - 1),
        "terminal_ebitda_margin": float(P["ebitda_margin"].iloc[-1]),
        "terminal_operating_margin": float(P["operating_margin"].iloc[-1]),
        "capex_fy28": float(P["capex"].iloc[1]),
        "peak_capex": float(P["capex"].max()),
    }


def blend(method_prices: dict[str, float], weights: dict[str, float]) -> float:
    return float(sum(method_prices[m] * weights[m] for m in METHODS))


def run_scenario(A: Anchors, S: Assumptions, cfg: Config) -> ScenarioResult:
    P = project(A, S)
    d = value_dcf(A, S, P)
    m = value_multiples(A, S, P)
    mp = {"dcf": d.price, "forward_pe": m.forward_pe, "ev_ebitda": m.ev_ebitda, "ev_revenue": m.ev_revenue}
    return ScenarioResult(S.name, S, P, d, m, mp, blend(mp, cfg.valuation.method_weights.to_dict()), _diagnostics(A, S, P, cfg))


# --------------------------------------------------------------------------- sensitivities
def sens_wacc_growth(A: Anchors, S: Assumptions, cfg: Config) -> pd.DataFrame:
    st = cfg.valuation.sensitivity
    P = project(A, S)
    waccs = [S.wacc + k * st.wacc_step for k in (-2, -1, 0, 1, 2)]
    gs = [S.g_term + k * st.g_step for k in (-2, -1, 0, 1, 2)]
    tab = pd.DataFrame(index=[f"{w:.1%}" for w in waccs], columns=[f"{g:.1%}" for g in gs], dtype=float)
    for w in waccs:
        for g in gs:
            try:
                tab.loc[f"{w:.1%}", f"{g:.1%}"] = value_dcf(A, S, P, wacc=w, g=g).price
            except ValueError:
                tab.loc[f"{w:.1%}", f"{g:.1%}"] = np.nan
    tab.index.name, tab.columns.name = "WACC", "Terminal growth"
    return tab


def sens_growth_margin(A: Anchors, S: Assumptions, cfg: Config) -> pd.DataFrame:
    st = cfg.valuation.sensitivity
    tab = pd.DataFrame(index=[f"{x:+.0%}" for x in st.growth_shift_pp], columns=[f"{x:+.0%}" for x in st.margin_shift_pp], dtype=float)
    for gs in st.growth_shift_pp:
        for ms in st.margin_shift_pp:
            S2 = S.with_(growth_shift=float(gs), margin_shift=float(ms))
            tab.loc[f"{gs:+.0%}", f"{ms:+.0%}"] = value_dcf(A, S2).price
    tab.index.name, tab.columns.name = "OCI growth shift (FY29+, pp/yr)", "Terminal EBITDA-margin shift (pp)"
    return tab


def sens_growth_multiple(A: Anchors, S: Assumptions, cfg: Config) -> pd.DataFrame:
    st = cfg.valuation.sensitivity
    mults = list(st.exit_multiples)
    tab = pd.DataFrame(index=[f"{x:+.0%}" for x in st.growth_shift_pp], columns=[f"{m:.0f}x" for m in mults], dtype=float)
    for gs in st.growth_shift_pp:
        S2 = S.with_(growth_shift=float(gs))
        P = project(A, S2)
        for m in mults:
            tab.loc[f"{gs:+.0%}", f"{m:.0f}x"] = exit_multiple_value(A, S2, m, int(st.exit_year_offset), P)
    tab.index.name, tab.columns.name = "OCI growth shift (FY29+, pp/yr)", "Exit EV/EBITDA at end of FY31"
    return tab


# --------------------------------------------------------------------------- reverse DCF
def reverse_dcf(A: Anchors, S: Assumptions) -> dict:
    """What does the current share price imply? Solve for the single parameter that sets DCF value = market price."""
    out: dict = {"price": A.price}
    base_P = project(A, S)

    def f_growth(x):
        return value_dcf(A, S.with_(growth_shift=x)).price - A.price

    def f_margin(x):
        return value_dcf(A, S.with_(margin_shift=x)).price - A.price

    for key, fn, lo, hi in (("growth_shift", f_growth, -0.45, 0.45), ("margin_shift", f_margin, -0.30, 0.30)):
        try:
            if fn(lo) * fn(hi) < 0:
                out[key] = float(optimize.brentq(fn, lo, hi, xtol=1e-6))
            else:
                out[key] = np.nan
        except Exception:
            out[key] = np.nan
    if not np.isnan(out["growth_shift"]):
        Pi = project(A, S.with_(growth_shift=out["growth_shift"]))
        out["implied_fy31_revenue"] = float(Pi["revenue_fy"].iloc[4])
        out["implied_fy36_revenue"] = float(Pi["revenue_fy"].iloc[-1])
        out["implied_revenue_cagr_to_fy31"] = float((Pi["revenue_fy"].iloc[4] / A.revenue_base) ** (1 / 5) - 1)
    out["base_fy31_revenue"] = float(base_P["revenue_fy"].iloc[4])
    out["base_fy36_revenue"] = float(base_P["revenue_fy"].iloc[-1])
    out["base_revenue_cagr_to_fy31"] = float((base_P["revenue_fy"].iloc[4] / A.revenue_base) ** (1 / 5) - 1)
    out["base_terminal_ebitda_margin"] = float(base_P["ebitda_margin"].iloc[-1])
    if not np.isnan(out["margin_shift"]):
        out["implied_terminal_ebitda_margin"] = float(base_P["ebitda_margin"].iloc[-1] + out["margin_shift"])
    return out


# --------------------------------------------------------------------------- orchestration
def peer_multiples_table(A: Anchors, snap: dict, ticker: str) -> pd.DataFrame:
    rows = {ticker: snap.get("info", {})}
    rows.update(snap.get("peers", {}))
    keep = {"forwardPE": "Forward P/E", "trailingPE": "Trailing P/E", "enterpriseToEbitda": "EV/EBITDA",
            "enterpriseToRevenue": "EV/Revenue", "beta": "Beta", "marketCap": "Market cap ($bn)"}
    df = pd.DataFrame({k: {nice: (v.get(raw) if v else None) for raw, nice in keep.items()} for k, v in rows.items()}).T
    df["Market cap ($bn)"] = pd.to_numeric(df["Market cap ($bn)"], errors="coerce") / 1e9
    return df.apply(pd.to_numeric, errors="coerce")


def run_valuation(cfg: Config, store: Store | None = None) -> ValuationRun:
    st = store or Store(cfg)
    A = build_anchors(st, cfg)
    assumptions = build_assumptions(A, cfg)
    scen = {s: run_scenario(A, assumptions[s], cfg) for s in SCEN}
    probs = cfg.valuation.scenario_probabilities.to_dict()

    rows = {}
    for s in SCEN:
        r = scen[s]
        rows[s] = {**{METHOD_LABELS[m]: r.method_prices[m] for m in METHODS}, "Blended": r.blended,
                   "Upside vs price": r.blended / A.price - 1.0}
    summary = pd.DataFrame(rows).T
    pw = float(sum(scen[s].blended * probs[s] for s in SCEN))
    base = scen["base"]
    run = ValuationRun(
        anchors=A, scenarios=scen, summary=summary, prob_weighted=pw, upside_prob_weighted=pw / A.price - 1.0,
        reverse=reverse_dcf(A, base.assumptions),
        sensitivities={
            "wacc_vs_terminal_growth": sens_wacc_growth(A, base.assumptions, cfg),
            "growth_vs_margin": sens_growth_margin(A, base.assumptions, cfg),
            "growth_vs_exit_multiple": sens_growth_multiple(A, base.assumptions, cfg),
        },
        derivations=A.log_frame(),
        peer_multiples=peer_multiples_table(A, st.consensus(), cfg.project.ticker),
        current_multiples=A.mult_stats[["current", "bear", "base", "bull", "current_percentile_rank"]],
    )
    return run


def save_valuation(run: ValuationRun, cfg: Config) -> None:
    d = cfg.models_dir
    run.summary.to_csv(d / "valuation_summary.csv")
    run.derivations.to_csv(d / "valuation_assumption_log.csv", index=False)
    for k, t in run.sensitivities.items():
        t.to_csv(d / f"sensitivity_{k}.csv")
    for s, r in run.scenarios.items():
        r.projection.to_csv(d / f"projection_{s}.csv")
