"""The nine dashboard pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import chart, df_show, fmt_pct, fmt_usd, get_bundle, get_cfg, header

from orcl_lab.models import score as SC
from orcl_lab.reporting.report import build_sections
from orcl_lab.risk import montecarlo as MC
from orcl_lab.valuation.model import project, value_dcf
from orcl_lab.viz import theme as T


# =============================================================================================== 1
def page_summary():
    b, cfg = get_bundle(), get_cfg()
    header("Executive summary", "ORCL Quantitative Investment & AI Valuation Lab - *can Oracle's AI/cloud growth justify its valuation and rapidly increasing capital requirements?*",
           ["HISTORICAL", "CONSENSUS", "MODEL"])
    v, sc = b.val, b.val.scenarios
    score = float(b.score["score_live"].dropna().iloc[-1])
    c = st.columns(4)
    c[0].metric("Price", fmt_usd(b.meta["price"], 2), help=f"Close {b.meta['price_date']}")
    c[1].metric("Base-case value", fmt_usd(sc["base"].blended), f"{sc['base'].blended / b.meta['price'] - 1:+.0%}")
    c[2].metric("Probability-weighted", fmt_usd(v.prob_weighted), f"{v.upside_prob_weighted:+.0%}")
    c[3].metric("Model conclusion", b.conclusion["label"], f"vote total {b.conclusion['total']:+d}", delta_color="off")
    c = st.columns(4)
    c[0].metric("Bear case", fmt_usd(sc["bear"].blended), f"{sc['bear'].blended / b.meta['price'] - 1:+.0%}")
    c[1].metric("Bull case", fmt_usd(sc["bull"].blended), f"{sc['bull'].blended / b.meta['price'] - 1:+.0%}")
    c[2].metric("Investment score", f"{score:.0f} / 100", SC.regime_label(score, cfg).title(), delta_color="off")
    c[3].metric("Signal (experimental)", str(b.signal.frame["state"].dropna().iloc[-1]).replace("_", " ").title())

    st.subheader("Model-driven conclusion")
    votes = b.conclusion["votes"].copy()
    votes["value"] = votes["value"].map(lambda x: f"{x:+d}")
    votes.columns = ["Vote", "Score", "Evidence"]
    df_show(votes, hide_index=True)
    st.caption(b.conclusion["rule"] + " The conclusion is the output of this fixed rule applied to model outputs - not a subjective view.")

    chart("football")
    st.subheader("Research report")
    st.caption("Generated automatically from the analysis bundle (`orcl-lab report` writes the same text to `reports/orcl_research_report.md`).")
    for i, (title, body) in enumerate(build_sections(b, cfg)):
        if title in ("Summary",):
            st.markdown(body)
        else:
            with st.expander(title, expanded=(i == 1)):
                st.markdown(body)

    st.subheader("Data provenance")
    man = get_cfg().path("data", "manifest.json")
    import json
    entries = json.loads(man.read_text(encoding="utf-8"))
    rows = [{"dataset": k, "kind": e["kind"], "source": e["source"], "retrieved (UTC)": e["retrieved_at_utc"], "rows": e.get("rows"),
             "first": e.get("first_date"), "last": e.get("last_date")} for k, e in sorted(entries.items())]
    df_show(pd.DataFrame(rows), hide_index=True)
    st.caption("Every dataset is recorded with its source and retrieval time in `data/manifest.json`. Kinds: HISTORICAL (reported/traded), CONSENSUS (third-party forecast snapshot), "
               "MODEL (this repository's outputs). Assumptions live in `config.yaml` and the derivation log on the Valuation page.")


# =============================================================================================== 2
def page_price():
    b = get_bundle()
    header("Price & technical analysis", "Trend, momentum, volatility, beta and how ORCL compares with the market, the Nasdaq 100, semiconductors and AI/cloud peers.", ["HISTORICAL"])
    start = st.select_slider("Chart start date (technical charts)", options=["2012-01-01", "2015-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01", "2025-01-01"], value="2018-01-01")
    t = b.trend
    c = st.columns(5)
    c[0].metric("Close", fmt_usd(t["close"], 2))
    c[1].metric("vs 200-day average", "Above" if t["above_200dma"] else "Below")
    c[2].metric("RSI (14)", f"{t['rsi']:.0f}", t["rsi_read"], delta_color="off")
    c[3].metric("52-week high distance", fmt_pct(t["pct_from_52w_high"]))
    c[4].metric("Relative volume", f"{t['rel_volume']:.2f}x")
    st.subheader("Performance and risk statistics")
    perf = b.perf[["CAGR", "Ann. volatility", "Sharpe", "Sortino", "Max drawdown", "Beta (vs S&P 500)", "Alpha (ann.)", "Total return", "Current drawdown"]].copy()
    perf.index = [T.lab(i) for i in perf.index]
    df_show(perf.style.format({"CAGR": "{:.1%}", "Ann. volatility": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max drawdown": "{:.1%}", "Beta (vs S&P 500)": "{:.2f}",
                               "Alpha (ann.)": "{:.1%}", "Total return": "{:,.0%}", "Current drawdown": "{:.1%}"}))
    cap = b.capm
    st.caption(f"CAPM vs S&P 500 (daily excess returns, Newey-West errors): beta {cap.beta:.2f} (t={cap.beta_tstat:.0f}), alpha {cap.alpha_annual:.1%} p.a. (t={cap.alpha_tstat:.1f}), R2 {cap.r_squared:.0%}. "
               f"Sharpe/Sortino use the 3-month T-bill; benchmarks are total-return ETF proxies (SPY, QQQ, SMH). Data: {b.sources['prices']}.")
    for k in ("price_ma", "rsi_macd", "atr_relvol", "rebased", "drawdowns", "rolling_vol", "rolling_beta", "capm_scatter", "corr_heatmap", "rolling_corr", "divergence", "calendar_returns"):
        chart(k, start)
    st.subheader("Divergence episodes versus the AI / cloud peer basket")
    ep = b.divergence.episodes
    if not ep.empty:
        e = ep.copy()
        e["start"], e["end"] = e["start"].dt.date, e["end"].dt.date
        df_show(e.tail(15).style.format({"peak_zscore": "{:.1f}", "peak_relative_return": "{:+.0%}"}), hide_index=True)
    st.subheader("Trailing returns")
    pr = b.period_returns.copy()
    pr.index = [T.lab(i) for i in pr.index]
    df_show(pr.style.format("{:+.1%}"))


# =============================================================================================== 3
def page_fundamentals():
    b = get_bundle()
    header("Fundamentals", "Revenue, margins, cash flow, capital expenditure and balance sheet since FY15 - reported figures from SEC filings and Oracle's earnings releases.", ["HISTORICAL"])
    q = b.q
    last = q.iloc[-1]
    c = st.columns(5)
    c[0].metric("Revenue (TTM)", f"${last['revenue_ttm'] / 1e3:,.1f}bn", f"{last['revenue_ttm_yoy']:+.0%} YoY")
    c[1].metric("Operating margin (TTM)", fmt_pct(last["operating_margin_ttm"], 1))
    c[2].metric("FCF (TTM)", f"${last['fcf_ttm'] / 1e3:,.1f}bn", fmt_pct(last["fcf_margin_ttm"]) + " margin", delta_color="off")
    c[3].metric("Net debt / EBITDA", f"{last['net_debt_to_ebitda_ttm']:.1f}x")
    c[4].metric("Capex / OCF (TTM)", f"{last['capex_to_ocf_ttm']:.2f}x")
    for k in ("rev_growth", "margins", "cashflow_capex", "capex_intensity", "leverage", "eps_shares"):
        chart(k)
    st.subheader("Data tables")
    view = st.radio("Frequency", ["Quarterly", "Annual"], horizontal=True)
    if view == "Quarterly":
        cols = ["label", "avail_date", "revenue", "revenue_yoy", "gross_margin_proxy", "operating_income", "operating_margin", "net_income", "eps_diluted", "ocf", "capex", "fcf", "fcf_margin",
                "total_debt_incl_leases", "cash_and_securities", "net_debt", "interest_expense", "diluted_shares", "rpo", "cloud_revenue", "oci_revenue", "oci_growth", "capex_yoy",
                "revenue_yoy", "eps_diluted_yoy", "fcf_margin_ttm", "capex_to_ocf_ttm", "debt_to_ebitda_ttm", "net_debt_to_ebitda_ttm", "interest_coverage_ttm", "rpo_to_ttm_revenue", "rpo_yoy"]
        cols = list(dict.fromkeys([c for c in cols if c in q.columns]))
        df_show(q[cols].sort_index(ascending=False))
        st.download_button("Download quarterly fundamentals (CSV)", q.to_csv().encode(), "orcl_fundamentals_quarterly.csv")
        st.caption("USD amounts in $ millions, shares in millions. `avail_date` is the first session whose close reflects the release (point-in-time safe). Gross margin is a proxy: Oracle does not report gross profit. "
                   "Derived quarters: Q4 = fiscal year - 9 months. Where two filings report a period the first-filed value is kept.")
    else:
        df_show(b.a.sort_index(ascending=False))
        st.download_button("Download annual fundamentals (CSV)", b.a.to_csv().encode(), "orcl_fundamentals_annual.csv")
    st.caption(f"Source: {b.sources['fundamentals']}.")


# =============================================================================================== 4
def page_rpo():
    b = get_bundle()
    r = b.rpo
    header("AI / RPO analysis", "Is the contracted backlog turning into revenue, profit and cash - and how fast? The RPO -> Revenue -> Cash-flow conversion framework.", ["HISTORICAL", "CONSENSUS", "MODEL"])
    st.info("**Key findings (computed from the data)**\n\n" + "\n".join(f"- {f}" for f in r.findings))
    for k in ("rpo", "cloud_mix", "growth_chain", "funnel", "lead_lag", "capital_efficiency"):
        chart(k)
    tabs = st.tabs(["Growth correlations (Spearman)", "Growth correlations (Pearson)", "Lead-lag detail", "Cash conversion", "Forward revenue / RPO", "Granger test"])
    with tabs[0]:
        df_show(r.growth_corr_spearman.style.format("{:.2f}"))
        st.caption("Rank correlations of YoY growth rates (minimum 8 paired observations). Pearson is dominated by the FY26 Q1 RPO step-change; see the tab and the post-break matrix below.")
        st.markdown("**Since the regime break (FY26 Q1)** - only a handful of observations:")
        df_show(r.growth_corr_post_break.style.format("{:.2f}"))
    with tabs[1]:
        df_show(r.growth_corr_pearson.style.format("{:.2f}"))
    with tabs[2]:
        df_show(r.lead_lag.style.format({"corr": "{:.2f}", "n_eff": "{:.1f}", "ci_low": "{:.2f}", "ci_high": "{:.2f}", "p_adj": "{:.3f}"}))
    with tabs[3]:
        df_show(b.cash_conv.tail(12).style.format("{:.2f}"))
        st.caption("Incremental operating margin = change in TTM operating income / change in TTM revenue (YoY). OCI revenue per capex dollar uses capex lagged four quarters.")
    with tabs[4]:
        cv = r.conversion.dropna(subset=["fwd4q_revenue"]).copy()
        df_show(cv.style.format({"rpo": "{:,.0f}", "fwd4q_revenue": "{:,.0f}", "fwd8q_revenue": "{:,.0f}", "fwd4q_revenue_pct_of_rpo": "{:.0%}", "fwd8q_revenue_pct_of_rpo": "{:.0%}", "rpo_pct_next_12m_disclosed": "{:.0%}"}))
        st.caption("Revenue recognised in the following 4 / 8 quarters as a share of RPO at each date. This is a lower bound on conversion (total revenue includes non-RPO revenue; some RPO converts later). `rpo_pct_next_12m_disclosed` is Oracle's own XBRL disclosure (FY19-FY21 only).")
    with tabs[5]:
        df_show(r.granger.style.format({"F": "{:.2f}", "p_value": "{:.3f}"}))
        st.caption("H0: past RPO growth does not help predict revenue growth beyond its own past (first differences). Precedence is not causation.")
    st.subheader("Limitations of this analysis")
    for t in r.limitations:
        st.markdown(f"- {t}")


# =============================================================================================== 5
def page_valuation():
    b, cfg = get_bundle(), get_cfg()
    v = b.val
    A = v.anchors
    header("Valuation", "Forward P/E, EV/EBITDA, EV/Revenue and a discounted-cash-flow model, run under bear / base / bull assumptions that are each traced to data, consensus or a stated scenario assumption.",
           ["HISTORICAL", "CONSENSUS", "ASSUMPTION", "MODEL"])
    st.dataframe(v.summary.style.format({"DCF": "${:,.0f}", "Forward P/E": "${:,.0f}", "EV/EBITDA": "${:,.0f}", "EV/Revenue": "${:,.0f}", "Blended": "${:,.0f}", "Upside vs price": "{:+.0%}"}))
    st.caption(f"Price ${A.price:,.2f}. Probability-weighted blended value ${v.prob_weighted:,.0f} ({v.upside_prob_weighted:+.0%}) with weights {dict(cfg.valuation.scenario_probabilities.to_dict())}. "
               f"Method weights: {dict(cfg.valuation.method_weights.to_dict())}.")
    chart("football")
    chart("dcf_bridge")
    chart("reverse_dcf")
    st.subheader("Sensitivity tables (base-case DCF)")
    for k in ("sens_wacc_g", "sens_growth_margin", "sens_growth_multiple"):
        chart(k)
    chart("multiples_hist")
    chart("peer_multiples")

    st.subheader("Interactive DCF explorer")
    st.caption("Change the drivers and watch the DCF value per share move. Everything else stays at the selected scenario's values.")
    s = st.radio("Start from scenario", ["bear", "base", "bull"], index=1, horizontal=True)
    S0 = v.scenarios[s].assumptions
    c1, c2, c3 = st.columns(3)
    wacc = c1.slider("WACC", 6.0, 14.0, float(round(S0.wacc * 100, 2)), 0.1, format="%.1f%%") / 100
    g = c1.slider("Terminal growth", 0.0, 4.5, float(S0.g_term * 100), 0.1, format="%.1f%%") / 100
    gs = c2.slider("Shift to OCI growth from FY29 (pp / year)", -20.0, 20.0, 0.0, 0.5) / 100
    ms = c2.slider("Shift to terminal EBITDA margin (pp)", -15.0, 15.0, 0.0, 0.5) / 100
    kt = c3.slider("Terminal net PP&E / revenue (x)", 0.4, 2.0, float(round(S0.k_target, 2)), 0.05)
    phi = c3.slider("OCI growth persistence (annual, FY29-31)", 0.1, 0.9, float(round(S0.phi, 2)), 0.01)
    try:
        S = S0.with_(wacc=wacc, g_term=g, growth_shift=gs, margin_shift=ms, k_target=kt, phi=phi)
        P = project(A, S)
        d = value_dcf(A, S, P)
        m = st.columns(4)
        m[0].metric("DCF value per share", fmt_usd(d.price), f"{d.price / A.price - 1:+.0%} vs price")
        m[1].metric("Enterprise value", f"${d.enterprise_value / 1e3:,.0f}bn")
        m[2].metric("Terminal value share of EV", f"{d.terminal_share_of_ev:.0%}")
        m[3].metric("Implied exit EV/EBITDA", f"{d.implied_exit_ev_ebitda:.1f}x")
        fig = go.Figure()
        x = [y.split("E")[0] + "E" for y in P.index]
        fig.add_trace(go.Bar(x=x, y=P["ufcf"] / 1e3, name="Unlevered FCF", marker_color=T.BLUE))
        fig.add_trace(go.Scatter(x=x, y=P["revenue_fy"] / 1e3, name="Revenue", line=dict(color=T.ORANGE, width=2.4)))
        T.style(fig, "Your scenario: revenue and unlevered free cash flow", "Fiscal year", "USD bn", height=380, hover="x")
        try:
            st.plotly_chart(fig, width="stretch")
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
        st.caption("What this shows: revenue (line) and unlevered FCF (bars) for the assumptions above. Source: MODEL OUTPUT (this repository). Data: FY27E-FY36E.")
    except ValueError as exc:
        st.error(str(exc))

    st.subheader("Assumption derivation log")
    st.caption("Every assumption, its value, and whether it is DATA (computed from reported figures), CONSENSUS (analyst snapshot), or SCENARIO (stated analyst assumption).")
    log = v.derivations
    basis = st.multiselect("Filter by basis", sorted(log["basis"].unique()), default=sorted(log["basis"].unique()))
    scn = st.multiselect("Filter by scenario", sorted(log["scenario"].unique()), default=sorted(log["scenario"].unique()))
    df_show(log[log["basis"].isin(basis) & log["scenario"].isin(scn)], hide_index=True)


# =============================================================================================== 6
def page_factor():
    b = get_bundle()
    fm = b.factor
    header("Quantitative factor model", "What explains ORCL's returns, how stable is it, and does it predict anything out of sample? Correlation, OLS, rolling regression, regularised walk-forward prediction and an earnings event study.",
           ["HISTORICAL", "MODEL"])
    st.info("**Key findings (computed from the data)**\n\n" + "\n".join(f"- {f}" for f in fm.findings))
    st.warning("**What this page does NOT claim:** causality. " + " ".join(fm.limitations[:1]) + " See the limitations at the bottom of the page.")
    st.subheader("A. Daily factor attribution")
    c = fm.daily.full.coefs.copy()
    df_show(c.style.format({"coef": "{:.3f}", "std_err": "{:.3f}", "t": "{:.1f}", "p_value": "{:.3f}"}))
    st.caption(f"OLS of ORCL excess returns on market, tech-tilt, semis/AI and hyperscaler factors, Newey-West (HAC) errors. R2 {fm.daily.full.r2:.1%}, adj. R2 {fm.daily.full.adj_r2:.1%}, n={fm.daily.full.n:,}, "
               f"Durbin-Watson {fm.daily.full.durbin_watson:.2f}, Breusch-Pagan p={fm.daily.full.breusch_pagan_p:.2f}, condition number {fm.daily.full.condition_number:.0f}. VIFs: " +
               ", ".join(f"{k} {x:.2f}" for k, x in fm.daily.full.vif.items()) + ".")
    chart("rolling_loadings")
    with st.expander("Factor loadings by calendar year"):
        df_show(fm.daily.by_year.style.format("{:.2f}"))
    st.subheader("B. Earnings-window regressions (explanatory, small N)")
    chart("spec_r2")
    df_show(fm.window_specs.style.format({"r2": "{:.1%}", "adj_r2": "{:.1%}", "loo_r2": "{:.1%}", "max_vif": "{:.1f}", "condition_number": "{:.0f}"}))
    sel = st.selectbox("Show coefficients for specification", list(fm.window_fits))
    df_show(fm.window_fits[sel].coefs.style.format({"coef": "{:.3f}", "std_err": "{:.3f}", "t": "{:.2f}", "p_value": "{:.3f}"}))
    st.caption("HC3 robust standard errors. S5 (adds OCI growth) is not estimated: only 17 windows are available.")
    st.subheader("C. Correlation, multicollinearity and walk-forward prediction")
    for k in ("feature_corr", "feature_collinearity", "wf_scatter", "feature_importance"):
        chart(k)
    df_show(fm.walk_forward.importance.style.format("{:.5f}"))
    df_show(fm.walk_forward.folds, hide_index=True)
    st.subheader("D. Earnings event study")
    ev = b.event
    chart("event_hist")
    chart("event_path")
    chart("event_scatter")
    df_show(ev.summary.style.format("{:.4f}"))
    st.markdown("**Robustness of the 1-day abnormal return to the benchmark choice**")
    df_show(ev.robustness.style.format("{:+.2%}"))
    st.markdown("**Cross-sectional regressions of the reaction on news (HC3)**")
    cs = ev.cross_section.copy()
    df_show(cs.style.format({c: "{:.3f}" for c in cs.columns if cs[c].dtype.kind == "f"}), hide_index=True)
    st.markdown("**Rank correlations with permutation p-values**")
    df_show(ev.correlations.style.format({"spearman_rho": "{:.2f}", "p_value_permutation": "{:.3f}"}), hide_index=True)
    with st.expander("All events"):
        t = ev.table.copy()
        df_show(t[["reaction_date", "car_1d", "car_3d", "car_5d", "car_20d", "eps_surprise_pct", "revenue_yoy", "revenue_accel", "oci_growth", "rpo_yoy", "guid_rev_mid"]].sort_index(ascending=False))
    for n in ev.notes:
        st.caption("- " + n)
    st.subheader("Limitations")
    for t in fm.limitations:
        st.markdown(f"- {t}")


# =============================================================================================== 7
def page_risk():
    b, cfg = get_bundle(), get_cfg()
    rk = b.risk
    header("Risk & Monte Carlo", "Historical and parametric VaR, expected shortfall, drawdowns, and 20,000-path Monte Carlo under four return models - including why a normal distribution understates ORCL's tail risk.", ["HISTORICAL", "MODEL"])
    r = rk["returns"]
    from orcl_lab.analytics import returns as R
    rf = b.px.index
    c = st.columns(5)
    c[0].metric("Annualised volatility", fmt_pct(R.annualised_vol(r)))
    c[1].metric("Max drawdown", fmt_pct(rk["dd_stats"].max_drawdown), f"peak {rk['dd_stats'].peak_date.date()}", delta_color="off")
    c[2].metric("Downside deviation", fmt_pct(rk["downside_dev"]))
    c[3].metric("Beta (S&P 500)", f"{b.capm.beta:.2f}")
    c[4].metric("Excess kurtosis", f"{rk['tail']['excess_kurtosis']:.0f}")
    st.subheader("Value-at-Risk and Expected Shortfall")
    vt = rk["var_table"].copy()
    vt["horizon_days"] = vt["horizon_days"].astype(int)
    df_show(vt.style.format({c: "{:.2%}" for c in vt.columns if c not in ("horizon_days", "confidence")} | {"confidence": "{:.0%}"}), hide_index=True)
    st.caption("Positive numbers are losses as a share of the position. Cornish-Fisher VaR is omitted (NaN) when the sample skew/kurtosis make the expansion non-monotonic - it is for ORCL. Historical multi-day VaR uses overlapping compounded returns.")
    for k in ("return_hist", "qq_plot", "var_methods", "var_backtest"):
        chart(k)
    bt = pd.DataFrame([{"method": m, "confidence": lv, "days": x[0].n, "exceptions": x[0].exceptions, "expected": round(x[0].expected, 1), "Kupiec p": x[0].kupiec_p,
                        "Christoffersen p": x[0].christoffersen_p, "verdict": x[0].verdict} for (m, lv), x in rk["var_bt"].items()])
    df_show(bt.style.format({"Kupiec p": "{:.4f}", "Christoffersen p": "{:.4f}", "confidence": "{:.0%}"}), hide_index=True)
    st.caption("Out-of-sample VaR backtest: VaR for day t is estimated from the 252 days ending t-1. Kupiec tests the breach frequency; Christoffersen tests whether breaches cluster.")
    st.subheader("Monte Carlo")
    st.caption(f"Settings: {rk['mc_settings']['n_sims']:,} simulations, {rk['mc_settings']['horizon_days']} trading days, drift mode `{rk['mc_settings']['drift_mode']}` "
               f"({rk['mc_settings']['drift_annual']:.1%} p.a.), seed {rk['mc_settings']['seed']}.")
    sm = rk["mc_summary"].copy()
    sm.index = ["Normal (GBM)", "Student-t", "GARCH(1,1)-t", "Block bootstrap"]
    df_show(sm.style.format({c: "{:.1%}" for c in sm.columns if "value" not in c} | {c: "{:.1f}" for c in sm.columns if "value" in c}))
    for k in ("mc_fan", "mc_terminal", "mc_mdd", "tail_freq"):
        chart(k)
    df_show(rk["mc_tails"].style.format("{:.4%}"))
    st.markdown("**Drift sensitivity (GARCH-t)** - the choice of drift matters more than the choice of distribution for the *expected* return:")
    ad = pd.DataFrame(rk["mc_alt_drift"]).T[["expected_return", "median_return", "p05_return", "p95_return", "prob_loss", "expected_shortfall_5pct"]]
    df_show(ad.style.format("{:.1%}"))
    with st.expander("Run your own simulation"):
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Simulations", 10_000, 100_000, 20_000, 5_000)
        hz = c2.number_input("Horizon (trading days)", 21, 1260, 252, 21)
        mode = c3.selectbox("Drift", ["capm", "historical", "zero"])
        mdl = c4.selectbox("Model", list(MC.MODELS), index=2)
        if st.button("Run simulation"):
            drift = MC.drift_annual(mode, r, rk["mc_settings"]["rf"], rk["mc_settings"]["beta"], rk["mc_settings"]["erp"])
            res = MC.simulate(r, mdl, int(n), int(hz), drift, rk["mc_settings"]["seed"], cfg.risk.monte_carlo.block_size)
            s = res.summary(100.0)
            df_show(pd.DataFrame(s, index=["value"]).drop(columns=["model"]).T.style.format("{:.3f}"))
            fig = go.Figure(go.Histogram(x=res.terminal / 100 - 1, nbinsx=80, marker_color=T.BLUE))
            T.style(fig, f"{mdl}: simulated {int(hz)}-day return", "Return", "Simulations", height=340, legend=False, hover="x", x_tickformat=".0%")
            try:
                st.plotly_chart(fig, width="stretch")
            except TypeError:
                st.plotly_chart(fig, use_container_width=True)
            st.caption("What this shows: the distribution of simulated returns for your settings. Source: MODEL OUTPUT on Yahoo Finance price history.")


# =============================================================================================== 8
def page_scenarios():
    b, cfg = get_bundle(), get_cfg()
    v = b.val
    header("Scenario analysis", "BEAR / BASE / BULL: what has to be true in each, and the implied ORCL share price.", ["CONSENSUS", "ASSUMPTION", "MODEL"])
    cols = st.columns(3)
    narr = {
        "bear": ("BEAR", "OCI growth falls substantially after the consensus-low path; revenue growth slows; capex stays elevated (asset turnover barely improves); FCF stays negative longer; the valuation multiple contracts."),
        "base": ("BASE", "Cloud growth stays strong along the consensus path then fades at a history-based persistence; RPO converts gradually; FCF recovers in the late-2020s; multiples normalise to their median."),
        "bull": ("BULL", "OCI growth stays extremely strong; RPO converts rapidly with new bookings on top; operating leverage keeps margins at record levels; capital intensity converges toward the hyperscaler median; multiples stay elevated."),
    }
    for col, s in zip(cols, ("bear", "base", "bull")):
        with col:
            r = v.scenarios[s]
            st.markdown(f"#### {narr[s][0]}")
            st.metric("Implied value per share", fmt_usd(r.blended), f"{r.blended / b.meta['price'] - 1:+.0%} vs price")
            st.caption(narr[s][1])
    st.subheader("Assumptions side by side")
    A = v.anchors
    rows = {}
    for s in ("bear", "base", "bull"):
        S = v.scenarios[s].assumptions
        d = v.scenarios[s].diagnostics
        P = v.scenarios[s].projection
        rows[s.title()] = {
            "Revenue FY27E ($bn)": S.cons_rev_fy1 / 1e3, "Revenue FY28E ($bn)": S.cons_rev_fy2 / 1e3, "OCI growth FY28E": P["oci_growth"].iloc[1], "OCI growth persistence (FY29-31)": S.phi,
            "Revenue FY31E ($bn)": d["fy31_revenue"] / 1e3, "Revenue CAGR FY26-FY31": d["revenue_cagr_to_fy31"], "Terminal EBITDA margin": S.margin_target,
            "Terminal GAAP operating margin": d["terminal_operating_margin"], "Terminal net PP&E / revenue (x)": S.k_target, "Depreciation rate": S.dep_rate,
            "Peak capex ($bn)": d["peak_capex"] / 1e3, "WACC": S.wacc, "Beta": S.beta, "Terminal growth": S.g_term, "P/E multiple (x)": S.pe, "EV/EBITDA multiple (x)": S.ev_ebitda,
            "EV/Revenue multiple (x)": S.ev_revenue, "Years of negative FCF": d["negative_fcf_years"], "First positive FCF year": d["first_positive_fcf_year"],
            "Peak net debt / EBITDA (x)": d["peak_net_debt_to_ebitda"], "Min interest cover (x)": d["min_interest_cover"], "Cum. cloud revenue to FY31 / today's RPO": d["cum_cloud_revenue_to_fy31_pct_of_rpo"],
            "Scenario probability": float(cfg.valuation.scenario_probabilities[s]),
        }
    tab = pd.DataFrame(rows)
    pct_rows = {"OCI growth FY28E", "OCI growth persistence (FY29-31)", "Revenue CAGR FY26-FY31", "Terminal EBITDA margin", "Terminal GAAP operating margin", "Depreciation rate",
                "WACC", "Terminal growth", "Scenario probability"}
    out = tab.copy().astype(object)
    for i in tab.index:
        for c in tab.columns:
            x = tab.loc[i, c]
            if not isinstance(x, (int, float, np.floating)):
                out.loc[i, c] = str(x)
            elif i == "OCI growth persistence (FY29-31)":
                out.loc[i, c] = f"{x:.2f}"
            elif i in pct_rows:
                out.loc[i, c] = f"{x:.1%}"
            elif i.startswith("Cum."):
                out.loc[i, c] = f"{x:.0%}"
            elif i.startswith("Years"):
                out.loc[i, c] = f"{int(x)}"
            else:
                out.loc[i, c] = f"{x:,.2f}"
    df_show(out)
    st.caption("Each assumption's derivation (DATA / CONSENSUS / SCENARIO) is in the log on the Valuation page. Scenario drivers sit at inter-quartile positions of their observed ranges, not extremes.")
    for k in ("scenario_prices", "scenario_revenue", "scenario_fcf", "scenario_margin_capex", "scenario_leverage", "scenario_vs_mc"):
        chart(k)
    st.subheader("Year-by-year projection")
    s = st.radio("Scenario", ["bear", "base", "bull"], index=1, horizontal=True, key="proj_s")
    P = v.scenarios[s].projection
    df_show(P[["revenue_fy", "revenue_growth", "oci_revenue", "oci_growth", "ebitda_margin", "operating_margin", "capex", "ppe_to_revenue", "ufcf", "net_debt_end", "net_debt_to_ebitda", "interest_cover", "eps"]]
            .style.format({"revenue_fy": "{:,.0f}", "revenue_growth": "{:.1%}", "oci_revenue": "{:,.0f}", "oci_growth": "{:.1%}", "ebitda_margin": "{:.1%}", "operating_margin": "{:.1%}", "capex": "{:,.0f}",
                           "ppe_to_revenue": "{:.2f}", "ufcf": "{:,.0f}", "net_debt_end": "{:,.0f}", "net_debt_to_ebitda": "{:.1f}", "interest_cover": "{:.1f}", "eps": "{:.2f}"}))
    st.caption("USD millions unless stated. FY27E covers the remaining nine months for flows (stub); revenue_fy is the full fiscal year. EPS is GAAP; net debt turns negative only because no buybacks or dividend growth are assumed.")


# =============================================================================================== 9
def page_score():
    b, cfg = get_bundle(), get_cfg()
    header("Investment score", "A transparent 0-100 score built from point-in-time percentile ranks, an experimental five-state signal, and an honest backtest with transaction costs and a train/test split.", ["HISTORICAL", "MODEL"])
    score = float(b.score["score_live"].dropna().iloc[-1])
    sig = str(b.signal.frame["state"].dropna().iloc[-1]).replace("_", " ")
    c = st.columns(4)
    c[0].metric("Investment score", f"{score:.1f} / 100", SC.regime_label(score, cfg).title(), delta_color="off")
    c[1].metric("Signal state (experimental)", sig)
    c[2].metric("Composite", f"{b.signal.frame['composite'].dropna().iloc[-1]:+.2f}")
    c[3].metric("Components live", f"{int(b.signal.frame['n_components'].iloc[-1])} / 6")
    chart("score_history")
    chart("score_components")

    st.subheader("Weights (configurable - and deliberately not optimised)")
    st.caption("Default weights from `config.yaml`. Move the sliders to see how the score changes - the point is robustness, not to find weights that look best.")
    base_w = cfg.score.weights.to_dict()
    cols = st.columns(6)
    w = {}
    for col, k in zip(cols, base_w):
        w[k] = col.slider(k.replace("_", " ").title(), 0, 100, int(round(base_w[k] * 100)), 5, key=f"w_{k}")
    tot = sum(w.values()) or 1
    wn = {k: x / tot for k, x in w.items()}
    sc = SC.component_scores(b.panel)
    live = SC.total_score(sc, wn)
    live = live["score"].where(live["weight_coverage"] >= 0.70)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b.score.index, y=b.score["score_live"], name="Configured weights", line=dict(color=T.INK, width=1.6)))
    fig.add_trace(go.Scatter(x=live.index, y=live, name="Your weights", line=dict(color=T.ORANGE, width=1.6)))
    T.style(fig, "Score under configured vs your weights", "Date", "Score (0-100)", height=360)
    fig.update_yaxes(range=[0, 100])
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"What this shows: the same percentile-rank components combined with your weights (normalised to 100%). Latest: configured {score:.1f}, yours {live.dropna().iloc[-1]:.1f}. Source: MODEL OUTPUT. Data: {live.dropna().index.min().date()} to {live.dropna().index.max().date()}.")
    st.divider()
    chart("score_buckets")
    st.markdown("**Score regimes (episodes)**")
    ep = b.buckets.episodes
    if len(ep):
        e = ep.copy()
        e["start"], e["end"] = pd.to_datetime(e["start"]).dt.date, pd.to_datetime(e["end"]).dt.date
        df_show(e.tail(15).style.format({c: "{:+.0%}" for c in e.columns if "return" in c or "fwd" in c}), hide_index=True)
    chart("weight_robust")
    st.subheader("Why the scoring methodology could fail")
    for t in SC.WHY_IT_MIGHT_FAIL:
        st.markdown(f"- {t}")
    st.subheader("Experimental five-state signal and honest backtest")
    st.warning("Experimental research model - **not** a guaranteed or recommended trading strategy. Thresholds were calibrated on the training sample only; execution is at the next open with transaction costs; long-only.")
    chart("signal_timeline")
    chart("backtest_nav")
    chart("backtest_dd")
    for name, t in b.backtest.by_period.items():
        st.markdown(f"**{name}**")
        df_show(t[["CAGR", "Ann. volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Total return", "Avg exposure", "Annual turnover"]].style.format(
            {"CAGR": "{:.1%}", "Ann. volatility": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max drawdown": "{:.1%}", "Calmar": "{:.2f}", "Total return": "{:+.0%}", "Avg exposure": "{:.0%}", "Annual turnover": "{:.1f}"}, na_rep=""))
    st.markdown("**Signal state diagnostics**")
    df_show(b.backtest.state_table.style.format({"share": "{:.0%}", "mean_next_day_ret_bp": "{:.1f}", "ann_vol": "{:.0%}"}))
    chart("cost_sensitivity")
    pl = b.backtest.placebo
    st.caption(f"Placebo test: re-ordering the strategy's own exposure decisions at random ({pl['n']} permutations) achieves a Sharpe at least as high as the strategy's ({pl['observed_sharpe']:.2f}) "
               f"{pl['p_value_one_sided']:.0%} of the time (placebo mean {pl['placebo_mean']:.2f}, 95th percentile {pl['placebo_p95']:.2f}). "
               + " ".join(f"Sharpe difference vs exposure-matched mix, {k}: {x[0]:+.2f} (95% bootstrap CI {x[1]:+.2f} to {x[2]:+.2f})." for k, x in b.backtest.sharpe_diff.items()))
    for n in b.backtest.notes:
        st.caption("- " + n)
