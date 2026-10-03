"""Valuation inputs: every anchor the scenarios need, derived from data, with a derivation log.

Basis labels used in the log (so a reader can always see what an assumption rests on):

    DATA       computed from reported historical figures (SEC XBRL, earnings releases, prices)
    CONSENSUS  taken from the dated analyst-consensus snapshot (a forecast, not data)
    SCENARIO   an explicit analyst assumption for the scenario (stated, not estimated)
    COMPUTED   follows arithmetically from the above
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..analytics import returns as R
from ..config import Config
from ..data.peer_fundamentals import latest_hyperscaler_capital_intensity
from ..data.store import Store
from .multiples import multiple_stats, multiples_history

SCEN = ("bear", "base", "bull")


@dataclass
class Derivation:
    scenario: str
    driver: str
    value: str
    basis: str
    explanation: str


@dataclass
class Anchors:
    # ---- market / balance sheet (as of the latest reported quarter) ----
    as_of: pd.Timestamp
    price: float
    price_date: pd.Timestamp
    shares_m: float
    net_debt: float
    gross_debt: float
    cash: float
    preferred: float
    minority: float
    ppe_now: float
    nwc_now: float
    dps: float
    # ---- base year (last completed fiscal year) ----
    base_fy_label: str
    base_fy: int
    seg_base: dict[str, float]
    revenue_base: float
    ppe_base_fy: float
    # ---- current fiscal year (partially reported) ----
    cur_fy_label: str
    elapsed_quarters: int
    cur_q_revenue: float
    cur_q_capex: float
    cur_q_cloud: float
    rpo: float
    avg_q_capex_4: float
    ttm: dict[str, float]
    # ---- consensus ----
    cons: dict[str, dict[str, float]]
    # ---- data-derived scenario anchors ----
    seg_growth_stage1: dict[str, float]
    margin_now: float
    margin_targets: dict[str, float]
    nwc_targets: dict[str, float]
    dep_rate: float
    amort_ttm: float
    k_targets: dict[str, float]
    phi: dict[str, float]
    tax_now: float
    tax_lr: float
    eff_interest_rate: float
    # ---- cost of capital ----
    rf10: float
    rf3m: float
    betas: dict[str, float]
    mult_stats: pd.DataFrame
    mult_hist: pd.DataFrame
    peers_capital_intensity: pd.Series
    log: list[Derivation] = field(default_factory=list)

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame([d.__dict__ for d in self.log])


def _ar1(s: pd.Series) -> tuple[float, float, int]:
    s = s.dropna()
    y, X = s.iloc[1:].to_numpy(), sm.add_constant(s.iloc[:-1].to_numpy())
    res = sm.OLS(y, X).fit()
    return float(res.params[1]), float(res.bse[1]), len(y)


def _consensus(snap: dict, base_revenue: float) -> dict[str, dict[str, float]]:
    """Map Yahoo's '0y' / '+1y' to fiscal years and sanity-check the mapping against reported revenue."""
    rev, eps = snap.get("revenue_estimate", {}), snap.get("earnings_estimate", {})
    out: dict[str, dict[str, float]] = {}
    ya = rev.get("0y", {}).get("yearAgoRevenue")
    if ya is None or abs(ya / 1e6 - base_revenue) / base_revenue > 0.01:
        raise ValueError(f"consensus '0y' year-ago revenue ({ya}) does not match last reported fiscal year "
                         f"({base_revenue} $m) - fiscal-year mapping of the snapshot is unreliable")
    for tag, key in (("fy1", "0y"), ("fy2", "+1y")):
        r, e = rev.get(key, {}), eps.get(key, {})
        out[tag] = {
            "rev_avg": r["avg"] / 1e6, "rev_low": r["low"] / 1e6, "rev_high": r["high"] / 1e6,
            "eps_avg": e.get("avg", np.nan), "eps_low": e.get("low", np.nan), "eps_high": e.get("high", np.nan),
            "n_analysts": e.get("numberOfAnalysts", np.nan),
        }
    return out


def build_anchors(st: Store, cfg: Config) -> Anchors:
    v = cfg.valuation
    q = st.fundamentals()
    last = q.iloc[-1]
    px_close = st.closes()[cfg.project.ticker]
    snap = st.consensus()
    rf = st.risk_free()
    log: list[Derivation] = []

    def L(scn, driver, value, basis, expl):
        log.append(Derivation(scn, driver, value, basis, expl))

    # ---------------- balance sheet at the latest reported quarter ----------------
    as_of = q.index[-1]
    gross_debt = float(last["total_debt_incl_leases"])
    cash = float(last["cash_and_securities"])
    net_debt = gross_debt - cash
    pref, minority = float(last["preferred_stock"]), float(last["minority_interest"])
    dilution = max(float(last["diluted_shares"] - last["basic_shares"]), 0.0)
    shares = float(last["shares_outstanding_cover"]) + dilution
    L("all", "Share count (m)", f"{shares:,.0f}", "DATA",
      f"10-Q cover-page shares outstanding ({last['shares_outstanding_cover']:,.0f}m) + dilution from the latest quarter "
      f"(diluted - basic weighted average = {dilution:,.0f}m).")
    L("all", "Net debt ($m)", f"{net_debt:,.0f}", "DATA",
      f"Debt {last['total_debt']:,.0f} + finance leases {last['finance_lease_liability']:,.0f} - cash & securities {cash:,.0f} "
      f"at {as_of.date()}. Operating-lease liabilities are NOT added (their cost is already inside EBITDA).")
    L("all", "Preferred / minority ($m)", f"{pref:,.0f} / {minority:,.0f}", "DATA", "Deducted from enterprise value at book value.")
    price = float(px_close.iloc[-1])
    div_ttm = float(last["dividends_paid_ttm"])
    dps = div_ttm / float(q["diluted_shares"].tail(4).mean())
    L("all", "Annual dividend per share ($)", f"{dps:.2f}", "DATA", "Trailing-12-month dividends paid / average diluted shares; held flat.")

    # ---------------- base year ----------------
    base_rows = q[q["fiscal_quarter"] == 4]
    base_end = base_rows.index[-1]
    base_fy = int(base_rows["fiscal_year"].iloc[-1])
    fy_q = q[q["fiscal_year"] == base_fy]
    if len(fy_q) != 4 or fy_q["oci_revenue"].isna().any():
        raise ValueError("base fiscal year lacks the OCI/SaaS revenue split - cannot build the segment model "
                         "(set SEC_USER_AGENT and re-run the build, or add kpi_overrides.csv rows)")
    seg_base = {"oci": float(fy_q["oci_revenue"].sum()), "saas": float(fy_q["saas_revenue"].sum())}
    revenue_base = float(fy_q["revenue"].sum())
    seg_base["other"] = revenue_base - seg_base["oci"] - seg_base["saas"]
    ppe_base_fy = float(q.loc[base_end, "ppe_net"])

    # ---------------- current fiscal year (partially reported) ----------------
    cur_fy = int(last["fiscal_year"])
    elapsed = int(last["fiscal_quarter"])
    cur_rows = q[q["fiscal_year"] == cur_fy]
    ttm = {k: float(last[k]) for k in ("revenue_ttm", "ebitda_ttm", "operating_income_ttm", "capex_ttm", "ocf_ttm", "fcf_ttm",
                                         "net_income_ttm", "interest_expense_ttm", "d_and_a_ttm", "eps_diluted_ttm", "sbc_ttm")}
    avg_q_capex = float(q["capex"].tail(4).mean())

    cons = _consensus(snap, revenue_base)

    # ---------------- segment growth anchors (stage 1) ----------------
    saas_g = float(q["saas_growth"].tail(4).mean())
    other_series = (q["revenue"] - q["cloud_revenue"])
    other_g = float((other_series / other_series.shift(4) - 1).tail(4).mean())
    L("all", "SaaS growth, FY27-28", f"{saas_g:.1%}", "DATA", "Mean of the last four quarters' reported SaaS YoY growth.")
    L("all", "Non-cloud (licence, hardware, services) growth, FY27-28", f"{other_g:.1%}", "DATA",
      "Mean YoY growth of (total revenue - cloud revenue) over the last four quarters.")

    # ---------------- EBITDA margin (percentiles of own 10-year TTM history) ----------------
    m_hist = (q["ebitda_ttm"] / q["revenue_ttm"]).dropna().tail(v.margin_history_years * 4)
    margin_now = float(m_hist.iloc[-1])
    mt = {s: float(m_hist.quantile(v.margin_percentiles[s])) for s in SCEN}
    mt["bull"] = max(mt["bull"], margin_now)      # a bull case must at least hold today's (record) margin
    for s in SCEN:
        floor_note = " Floored at today's margin so the bull case never assumes compression." if (s == "bull" and mt[s] == margin_now) else ""
        L(s, "Terminal EBITDA margin", f"{mt[s]:.1%}", "DATA",
          f"{int(v.margin_percentiles[s]*100)}th percentile of Oracle's own TTM EBITDA margin over {len(m_hist)//4} years "
          f"(range {m_hist.min():.1%}-{m_hist.max():.1%}; current {margin_now:.1%}). Reached linearly over {v.margin_convergence_years} years.{floor_note}")

    # ---------------- working capital (percentiles of NWC / revenue) ----------------
    nwc_ratio = (q["nwc_proxy"] / q["revenue_ttm"]).dropna().tail(v.margin_history_years * 4)
    nwc_t = {"bear": float(nwc_ratio.quantile(0.75)), "base": float(nwc_ratio.quantile(0.50)), "bull": float(nwc_ratio.quantile(0.25))}
    nwc_now = float(last["nwc_proxy"])
    for s in SCEN:
        L(s, "Net working capital / revenue", f"{nwc_t[s]:.1%}", "DATA",
          f"{ {'bear':'75th','base':'50th','bull':'25th'}[s]} percentile of (receivables - payables - contract liabilities)/TTM revenue over "
          f"{len(nwc_ratio)//4} years; today {nwc_now/ttm['revenue_ttm']:.1%} (customer prepayments spiked). Moves there over 3 years: a "
          f"prepayment unwind is a cash OUTFLOW.")

    # ---------------- depreciation & intangibles ----------------
    dep_ttm = float(q["depreciation"].tail(4).sum())
    avg_ppe = float((q["ppe_net"].iloc[-1] + q["ppe_net"].iloc[-5]) / 2)
    dep_rate = dep_ttm / avg_ppe
    amort_ttm = float(q["amortization_intangibles_is"].tail(4).sum())
    L("all", "Depreciation rate (of net PP&E)", f"{dep_rate:.1%}", "DATA",
      f"TTM depreciation {dep_ttm:,.0f} / average net PP&E {avg_ppe:,.0f}. Scenario multipliers (bear x{v.depreciation_rate_multiplier.bear}, "
      f"bull x{v.depreciation_rate_multiplier.bull}) reflect asset-life uncertainty (SCENARIO). Note: construction-in-progress does not depreciate yet, so the "
      f"in-service rate is likely higher than this blended figure.")
    L("all", "Amortisation of acquired intangibles", f"{amort_ttm:,.0f} $m/yr, run off over 5 yrs", "DATA", "TTM amount, declining linearly to zero.")

    # ---------------- capital intensity (terminal net PP&E / revenue) ----------------
    peers = pd.read_csv(cfg.processed_dir / "peer_fundamentals.csv", parse_dates=["fiscal_year_end"])
    pk = latest_hyperscaler_capital_intensity(peers)
    k_orcl = ppe_base_fy / revenue_base
    adverse, favourable = float(max(k_orcl, pk.max())), float(pk.median())
    pos = float(v.scenario_position)
    k_t = {"bear": adverse - pos * (adverse - favourable), "base": float(pk.max()), "bull": favourable + pos * (adverse - favourable)}
    for s, expl in (("bear", f"{pos:.0%} of the way from Oracle's own FY{str(base_fy)[-2:]} level ({k_orcl:.2f}x, adverse end) toward the hyperscaler median ({favourable:.2f}x)"),
                    ("base", f"the most capital-intensive hyperscaler peer ({pk.idxmax()} {pk.max():.2f}x)"),
                    ("bull", f"{pos:.0%} of the way from the hyperscaler median ({favourable:.2f}x; MSFT {pk.get('MSFT', np.nan):.2f}x, GOOGL {pk.get('GOOGL', np.nan):.2f}x, AMZN {pk.get('AMZN', np.nan):.2f}x) toward Oracle's own level ({k_orcl:.2f}x)")):
        L(s, "Terminal net PP&E / revenue", f"{k_t[s]:.2f}x", "DATA",
          f"Set to {expl}; converges over {v.capital_intensity_convergence_years} years. Capex is then whatever is needed to hold PP&E at that ratio.")

    # ---------------- growth persistence (AR(1) of YoY revenue growth) ----------------
    g_all = q["revenue_yoy"].dropna()
    brk = q.index[q["rpo_yoy"] > 1.0]
    brk_date = brk[0] if len(brk) else g_all.index[-1]
    phi_all, se_all, n_all = _ar1(g_all)
    phi_pre, se_pre, n_pre = _ar1(g_all[g_all.index < brk_date])
    lo_phi, hi_phi = float(np.clip(phi_pre ** 4, 0.05, 0.95)), float(np.clip(phi_all ** 4, 0.05, 0.95))
    pos_phi = float(v.scenario_position)
    phi_a = {"bear": lo_phi + pos_phi * (hi_phi - lo_phi), "base": 0.5 * (lo_phi + hi_phi), "bull": hi_phi - pos_phi * (hi_phi - lo_phi)}
    for s in SCEN:
        expl = {"bear": f"{pos_phi:.0%} of the way from the pre-regime-break estimate ({lo_phi:.2f}: AR(1) {phi_pre:.2f}, n={n_pre}, growth reverts quickly) toward the full-sample estimate",
                "bull": f"{pos_phi:.0%} of the way from the full-sample estimate ({hi_phi:.2f}: AR(1) {phi_all:.2f}, se {se_all:.2f}, n={n_all}, growth is persistent in the AI regime) toward the pre-break estimate",
                "base": "midpoint of the two regimes' annualised AR(1) estimates"}[s]
        L(s, "OCI growth persistence (annual, FY29-31)", f"{phi_a[s]:.2f}", "DATA",
          f"Excess growth over the terminal rate decays by this factor each year ({expl}). Regime break = first quarter RPO grew >100% YoY ({brk_date.date()}).")

    # ---------------- tax ----------------
    tax_hist = q["effective_tax_rate_ttm"].dropna().tail(40)
    tax_now = float(np.clip(tax_hist.iloc[-1], v.tax_rate_floor, v.tax_rate_cap))
    tax_lr = float(v.long_run_tax_rate)
    L("all", "Tax rate", f"{tax_now:.1%} -> {tax_lr:.1%}", "DATA / SCENARIO",
      f"Starts at the TTM effective rate (DATA, bounded {v.tax_rate_floor:.0%}-{v.tax_rate_cap:.0%}) and normalises over {v.tax_normalisation_years} years to a "
      f"long-run {tax_lr:.0%} (SCENARIO: Oracle's 10y median effective rate was {tax_hist.median():.1%}; statutory US federal is 21%).")

    # ---------------- rates & beta ----------------
    rf10, rf3m = float(rf["rf_10y"].iloc[-1]), float(rf["rf_3m"].iloc[-1])
    rets = R.simple_returns(st.prices())
    rfd = R.daily_rf(rf["rf_3m"])
    betas = {}
    for name, n in (("2y", 504), ("5y", 1260), ("full", len(rets))):
        sub = rets.tail(n)
        betas[name] = float(R.capm(sub[cfg.project.ticker], sub[cfg.data.benchmarks.sp500], rfd).beta)
    blume = {k: 0.67 * b + 0.33 for k, b in betas.items()}
    L("all", "Beta (Blume-adjusted)", ", ".join(f"{k}: {x:.2f}" for k, x in blume.items()), "DATA",
      "OLS beta of daily excess returns vs the S&P 500 (SPY) over 2y / 5y / full history, adjusted 0.67*raw + 0.33. "
      "Bear uses the highest, base the median, bull the lowest.")
    interest_ttm = ttm["interest_expense_ttm"]
    avg_gross = float((q["total_debt_incl_leases"].iloc[-1] + q["total_debt_incl_leases"].iloc[-5]) / 2)
    eff_rate = interest_ttm / avg_gross
    L("all", "Interest rate on existing net debt", f"{eff_rate:.1%} -> {rf10 + v.credit_spread:.1%}", "DATA",
      f"TTM interest expense / average gross debt, rising to the marginal cost (10y Treasury {rf10:.2%} + {v.credit_spread:.1%} spread [SCENARIO]) as debt refinances over 5 years.")

    # ---------------- multiples (own history) ----------------
    hist = multiples_history(st.closes()[cfg.project.ticker], q)
    mstats = multiple_stats(hist, v.multiple_history_years, v.multiple_percentiles.to_dict())
    for s in SCEN:
        L(s, "Valuation multiples (P/E, EV/EBITDA, EV/Revenue)",
          " / ".join(f"{mstats.loc[c, s]:.1f}x" for c in ("pe_trailing", "ev_ebitda", "ev_revenue")), "DATA",
          f"{int(v.multiple_percentiles[s]*100)}th percentile of Oracle's own trailing multiples over {v.multiple_history_years} years "
          f"(today {mstats.loc['pe_trailing','current']:.1f}x / {mstats.loc['ev_ebitda','current']:.1f}x / {mstats.loc['ev_revenue','current']:.1f}x).")

    for tag, lab in (("fy1", f"FY{str(cur_fy)[-2:]}"), ("fy2", f"FY{str(cur_fy + 1)[-2:]}")):
        c = cons[tag]
        L("bear", f"Revenue {lab} ($m)", f"{c['rev_low']:,.0f}", "CONSENSUS", f"Lowest analyst estimate in the snapshot ({snap['retrieved_at_utc'][:10]}).")
        L("base", f"Revenue {lab} ($m)", f"{c['rev_avg']:,.0f}", "CONSENSUS", f"Mean of {int(c['n_analysts'])} analysts.")
        L("bull", f"Revenue {lab} ($m)", f"{c['rev_high']:,.0f}", "CONSENSUS", "Highest analyst estimate in the snapshot.")
    for s in SCEN:
        L(s, "Terminal growth", f"{v.terminal_growth[s]:.1%}", "SCENARIO",
          f"Stated scenario assumption; must stay below the 10-year Treasury yield ({rf10:.2%}) and long-run nominal GDP growth.")
        L(s, "Scenario probability", f"{v.scenario_probabilities[s]:.0%}", "SCENARIO", "Subjective weight used only for the probability-weighted value.")
    L("all", "Equity risk premium", f"{v.equity_risk_premium:.1%}", "SCENARIO", "Stated assumption (typical long-run range 4-6%).")

    return Anchors(
        as_of=as_of, price=price, price_date=px_close.index[-1], shares_m=shares, net_debt=net_debt, gross_debt=gross_debt, cash=cash,
        preferred=pref, minority=minority, ppe_now=float(last["ppe_net"]), nwc_now=nwc_now, dps=dps,
        base_fy_label=f"FY{str(base_fy)[-2:]}", base_fy=base_fy, seg_base=seg_base, revenue_base=revenue_base, ppe_base_fy=ppe_base_fy,
        cur_fy_label=f"FY{str(cur_fy)[-2:]}", elapsed_quarters=elapsed, cur_q_revenue=float(cur_rows["revenue"].sum()),
        cur_q_capex=float(cur_rows["capex"].sum()), cur_q_cloud=float(cur_rows["cloud_revenue"].sum()),
        rpo=float(last["rpo"]), avg_q_capex_4=avg_q_capex, ttm=ttm, cons=cons,
        seg_growth_stage1={"saas": saas_g, "other": other_g}, margin_now=margin_now, margin_targets=mt, nwc_targets=nwc_t,
        dep_rate=dep_rate, amort_ttm=amort_ttm, k_targets=k_t, phi=phi_a, tax_now=tax_now, tax_lr=tax_lr, eff_interest_rate=eff_rate,
        rf10=rf10, rf3m=rf3m, betas=blume, mult_stats=mstats, mult_hist=hist, peers_capital_intensity=pk, log=log,
    )
