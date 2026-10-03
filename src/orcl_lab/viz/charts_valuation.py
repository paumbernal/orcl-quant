"""Page 5 (valuation) and page 8 (scenario analysis)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..valuation.model import value_dcf
from ..valuation.scenarios import METHOD_LABELS, METHODS
from . import theme as T
from .theme import SCENARIO, Chart

MODEL = "MODEL OUTPUT (this repository) built on SEC XBRL fundamentals, Oracle earnings releases and a Yahoo Finance consensus snapshot"


def _heat(tab: pd.DataFrame, title: str, price: float, key: str, expl: str, base_pos: tuple[int, int], src: str, rng: str, page: int = 5) -> Chart:
    z = tab.values.astype(float)
    text = [[("" if np.isnan(v) else f"${v:,.0f}") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(tab.columns), y=list(tab.index), text=text, texttemplate="%{text}", zmid=price,
                               colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]], colorbar=dict(title="USD / share"),
                               hovertemplate=f"{tab.index.name}: %{{y}}<br>{tab.columns.name}: %{{x}}<br>Value: %{{text}}<extra></extra>"))
    i, j = base_pos
    fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color=T.INK, width=3))
    T.style(fig, title, tab.columns.name, tab.index.name, height=400, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return Chart(key, page, fig, title, expl, src, rng)


def build(b) -> list[Chart]:
    out: list[Chart] = []
    v = b.val
    A = v.anchors
    scen = v.scenarios
    price = A.price
    tkr = b.meta["ticker"]
    src = MODEL
    asof = f"valuation date {A.price_date.date()}, balance sheet {A.as_of.date()}"
    pt = b.consensus.get("price_targets", {})

    # ---------------------------------------------------------------- football field
    fig = go.Figure()
    order = list(METHODS) + ["blended"]
    labels = [METHOD_LABELS[m] for m in METHODS] + ["Blended (weights in config)"]
    for m, lab in zip(order, labels):
        vals = {s: (scen[s].method_prices[m] if m != "blended" else scen[s].blended) for s in scen}
        lo, hi = min(vals.values()), max(vals.values())
        fig.add_trace(go.Bar(y=[lab], x=[hi - lo], base=[lo], orientation="h", marker_color="rgba(42,120,214,0.30)", showlegend=False, hoverinfo="skip"))
        for s in ("bear", "base", "bull"):
            fig.add_trace(go.Scatter(y=[lab], x=[vals[s]], mode="markers", marker=dict(size=12, color=SCENARIO[s], line=dict(color=T.SURFACE, width=2)),
                                     name=s.title(), legendgroup=s, showlegend=(m == "dcf"), hovertemplate=f"{s.title()} {lab}: $%{{x:,.0f}}<extra></extra>"))
    fig.add_vline(x=price, line=dict(color=T.INK, width=2), annotation_text=f"Price ${price:,.0f}", annotation_position="top")
    if pt:
        for k, dash in (("low", "dot"), ("mean", "dash"), ("high", "dot")):
            if pt.get(k):
                fig.add_vline(x=pt[k], line=dict(color=T.MUTED, width=1, dash=dash), annotation_text=f"Analyst {k} ${pt[k]:,.0f}",
                              annotation_position="top left" if k == "low" else "top right")
    T.style(fig, f"Valuation football field: value per share by method and scenario", "Implied value per share (USD)", None, height=470, hover="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    out.append(Chart("football", 5, fig, "Valuation football field: value per share by method and scenario",
                     f"Each row is a valuation method; the shaded span runs from the bear to the bull value with the three scenario values marked. The black line is the current price (${price:,.2f}); "
                     f"dashed grey lines are sell-side price targets (low / mean / high). Base-case blended value is ${scen['base'].blended:,.0f} ({scen['base'].blended/price-1:+.0%}); "
                     f"probability-weighted ${v.prob_weighted:,.0f} ({v.upside_prob_weighted:+.0%}). Multiple-based methods apply Oracle's own historical (capital-light-era) multiples and so sit above the DCF in the bear case.",
                     src, asof))

    # ---------------------------------------------------------------- DCF bridge (base)
    d = scen["base"].dcf
    meas = ["relative", "relative", "total", "relative", "relative", "total"]
    xs = ["PV of FY27-36 free cash flow", "PV of terminal value", "Enterprise value", "Net debt (incl. finance leases)", "Preferred stock & minorities", "Equity value"]
    ys = [d.pv_explicit / 1e3, d.pv_terminal / 1e3, 0, -A.net_debt / 1e3, -(A.preferred + A.minority) / 1e3, 0]
    fig = go.Figure(go.Waterfall(x=xs, y=ys, measure=meas, increasing=dict(marker=dict(color=T.BLUE)), decreasing=dict(marker=dict(color=T.ORANGE)),
                                 totals=dict(marker=dict(color=T.INK2)), connector=dict(line=dict(color=T.AXIS)),
                                 text=[f"${abs(y):,.0f}bn" if y else "" for y in ys], textposition="outside"))
    T.style(fig, f"Base-case DCF bridge: enterprise value to equity value (= ${d.price:,.0f} per share)", None, "USD bn", height=430, legend=False, hover="x")
    out.append(Chart("dcf_bridge", 5, fig, "Base-case DCF bridge",
                     f"Present value of the explicit cash flows (negative early on as capex outruns cash generation, positive later) plus the terminal value gives enterprise value; "
                     f"subtracting net debt and preferred/minority claims leaves equity. Terminal value is {d.terminal_share_of_ev:.0%} of EV (implied exit multiple {d.implied_exit_ev_ebitda:.1f}x EBITDA) - "
                     f"the valuation depends overwhelmingly on what happens after FY36 and on the cost of capital ({scen['base'].assumptions.wacc:.1%} WACC).",
                     src, asof))

    # ---------------------------------------------------------------- sensitivity heatmaps
    st = v.sensitivities
    t1 = st["wacc_vs_terminal_growth"]
    out.append(_heat(t1, "Sensitivity: WACC vs terminal growth (base-case DCF value per share)", price, "sens_wacc_g",
                     "Value per share as the discount rate and long-run growth change, other base assumptions fixed (black box = base). Red cells are below the current price, blue above. "
                     "Each +0.5pp on WACC removes roughly the same value as ~1pp of terminal growth adds - a typical Gordon-model trade-off.", (2, 2), src, asof))
    t2 = st["growth_vs_margin"]
    out.append(_heat(t2, "Sensitivity: OCI growth vs terminal EBITDA margin (base-case DCF value per share)", price, "sens_growth_margin",
                     "Rows shift OCI growth in every year from FY29 by the stated percentage points (the FY27-28 consensus anchor is unchanged); columns shift the terminal EBITDA margin. "
                     "The table shows how much operating performance must be delivered for the stock to be worth more (blue) or less (red) than today's price.", (2, 2), src, asof))
    t3 = st["growth_vs_exit_multiple"]
    out.append(_heat(t3, "Sensitivity: OCI growth vs exit EV/EBITDA multiple at end-FY31 (value per share)", price, "sens_growth_multiple",
                     "An exit-multiple view of the same question: sell the business at the end of FY31 at the stated EV/EBITDA, add the cash flows until then, subtract net debt. "
                     "Oracle's own 10-year median EV/EBITDA is shown in the multiples chart; the base-case DCF implies the exit multiple in the bridge note.", (2, 2), src, asof))

    # ---------------------------------------------------------------- multiples history
    mh = A.mult_hist[A.mult_hist.index >= A.mult_hist.index[-1] - pd.DateOffset(years=10)]
    ms = A.mult_stats
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=("Trailing P/E (GAAP)", "EV / EBITDA (TTM)", "EV / Revenue (TTM)"))
    for i, (col, color) in enumerate((("pe_trailing", T.BLUE), ("ev_ebitda", T.VIOLET), ("ev_revenue", T.AQUA)), start=1):
        s = mh[col].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name=col, line=dict(color=color, width=1.8), showlegend=False), i, 1)
        for lvl, dash in (("bear", "dot"), ("base", "dash"), ("bull", "dot")):
            fig.add_hline(y=ms.loc[col, lvl], line=dict(color=T.MUTED, width=1, dash=dash), row=i, col=1,
                          annotation_text=f"{lvl} = {ms.loc[col, lvl]:.1f}x", annotation_position="top left")
    T.style(fig, "Oracle's own valuation multiples (point-in-time) and the scenario anchors", None, None, height=640, legend=False)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    for i in (1, 2, 3):
        fig.update_yaxes(title_text="x", row=i, col=1)
    out.append(Chart("multiples_hist", 5, fig, "Oracle's own valuation multiples (point-in-time)",
                     f"Multiples computed each day from the price then and only the fundamentals public then. Dotted/dashed lines are the 25th/50th/75th percentiles used for bear/base/bull. "
                     f"Today: P/E {ms.loc['pe_trailing','current']:.1f}x ({ms.loc['pe_trailing','current_percentile_rank']:.0%} percentile of the last 10 years), "
                     f"EV/EBITDA {ms.loc['ev_ebitda','current']:.1f}x ({ms.loc['ev_ebitda','current_percentile_rank']:.0%}), EV/Revenue {ms.loc['ev_revenue','current']:.1f}x ({ms.loc['ev_revenue','current_percentile_rank']:.0%}). "
                     f"Caveat: those history points were earned when capex ran at a median {b.q['capex_to_revenue_ttm'].loc[mh.index[0]:].median():.0%} of revenue (now {b.q['capex_to_revenue_ttm'].iloc[-1]:.0%}) - applying them to a capital-heavy business flatters it.", b.sources["fundamentals"] + "; " + b.sources["prices"], T.span(mh.index)))

    # ---------------------------------------------------------------- peer multiples
    pm = v.peer_multiples
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Forward P/E", "EV / EBITDA", "EV / Revenue"), horizontal_spacing=0.08)
    for i, col in enumerate(("Forward P/E", "EV/EBITDA", "EV/Revenue"), start=1):
        s = pm[col].dropna()
        fig.add_trace(go.Bar(x=list(s.index), y=s.values, marker_color=[T.ENTITY.get(t, T.MUTED) for t in s.index], showlegend=False,
                             text=[f"{x:.1f}x" for x in s.values], textposition="outside"), 1, i)
    T.style(fig, "Relative valuation: ORCL vs AI / cloud peers (current)", None, "Multiple (x)", height=400, legend=False, hover="x")
    out.append(Chart("peer_multiples", 5, fig, "Relative valuation: ORCL vs AI / cloud peers",
                     "Current multiples from Yahoo Finance for ORCL and the peers (forward P/E uses next-fiscal-year consensus EPS). Peers differ enormously in margin structure and growth - "
                     "cheap on one multiple and expensive on another is the norm - so this is context for the DCF, not a target. ORCL's EV multiples include a large debt load that "
                     "equity-based multiples (P/E) do not.", "Yahoo Finance (current snapshot) - CONSENSUS / MARKET DATA, " + b.meta["consensus_date"], b.meta["consensus_date"]))

    # ---------------------------------------------------------------- reverse DCF curve
    base = scen["base"]
    shifts = np.arange(-0.30, 0.2001, 0.025)
    vals = [value_dcf(A, base.assumptions.with_(growth_shift=float(s))).price for s in shifts]
    rv = v.reverse
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shifts, y=vals, name="Base-case DCF value", line=dict(color=T.BLUE, width=2.6)))
    fig.add_hline(y=price, line=dict(color=T.INK, width=2), annotation_text=f"Price ${price:,.0f}", annotation_position="top left")
    if not np.isnan(rv.get("growth_shift", np.nan)):
        fig.add_trace(go.Scatter(x=[rv["growth_shift"]], y=[price], mode="markers", name="Market-implied", marker=dict(size=14, color=T.ORANGE, line=dict(color=T.SURFACE, width=2))))
    T.style(fig, "Reverse DCF: what OCI growth does the current price imply?", "Shift to OCI growth from FY29 on (percentage points per year)", "DCF value per share (USD)", height=420, hover="x", x_tickformat=".0%")
    gs = rv.get("growth_shift", np.nan)
    txt = (f"Holding every other base-case assumption fixed, today's price is consistent with OCI growth running {gs*100:+.1f}pp per year versus the base path from FY29 - "
           f"FY31 revenue of ${rv.get('implied_fy31_revenue', np.nan)/1e3:,.0f}bn ({rv.get('implied_revenue_cagr_to_fy31', np.nan):.0%} CAGR from FY26) against ${rv['base_fy31_revenue']/1e3:,.0f}bn ({rv['base_revenue_cagr_to_fy31']:.0%}) in the base case. "
           f"Alternatively, with base growth, the price implies a terminal EBITDA margin of {rv.get('implied_terminal_ebitda_margin', np.nan):.1%} versus {rv['base_terminal_ebitda_margin']:.1%}."
           if not np.isnan(gs) else "The current price lies outside the range of the growth shifts tested.")
    out.append(Chart("reverse_dcf", 5, fig, "Reverse DCF: what OCI growth does the current price imply?", txt, src, asof))

    # ================================================================ PAGE 8 : scenarios

    # scenario prices by method
    fig = go.Figure()
    for s in ("bear", "base", "bull"):
        ys = [scen[s].method_prices[m] for m in METHODS] + [scen[s].blended]
        fig.add_trace(go.Bar(x=[METHOD_LABELS[m] for m in METHODS] + ["Blended"], y=ys, name=s.title(), marker_color=SCENARIO[s],
                             text=[f"${y:,.0f}" for y in ys], textposition="outside"))
    fig.add_hline(y=price, line=dict(color=T.INK, width=2), annotation_text=f"Price ${price:,.0f}")
    T.style(fig, "Implied ORCL share price by scenario and method", "Valuation method", "Implied value per share (USD)", height=450, hover="x")
    fig.update_layout(barmode="group")
    out.append(Chart("scenario_prices", 8, fig, "Implied ORCL share price by scenario and method",
                     f"Bear / base / bull blended values are ${scen['bear'].blended:,.0f} / ${scen['base'].blended:,.0f} / ${scen['bull'].blended:,.0f} against a price of ${price:,.0f}. "
                     f"Scenario definitions: bear = analyst-low revenue, fast growth fade, lower margin, high capital intensity, high beta, contracting multiples; bull = analyst-high revenue, persistent growth, "
                     "record margins held, hyperscaler-median asset turnover, low beta, expanding multiples. All drivers sit at inter-quartile (not extreme) positions of their observed ranges.", src, asof))

    # scenario revenue paths
    ann = b.a[["label", "revenue"]].tail(5)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ann["label"]), y=ann["revenue"] / 1e3, name="Reported", line=dict(color=T.INK, width=2.6)))
    for s in ("bear", "base", "bull"):
        P = scen[s].projection
        xs = [ann["label"].iloc[-1]] + [y.split("E")[0] + "E" for y in P.index]
        ys = [ann["revenue"].iloc[-1] / 1e3] + list(P["revenue_fy"] / 1e3)
        fig.add_trace(go.Scatter(x=xs, y=ys, name=s.title(), line=dict(color=SCENARIO[s], width=2.4)))
    cons = A.cons
    fig.add_trace(go.Scatter(x=[f"FY{str(A.base_fy + 1)[-2:]}E", f"FY{str(A.base_fy + 2)[-2:]}E"], y=[cons["fy1"]["rev_avg"] / 1e3, cons["fy2"]["rev_avg"] / 1e3], mode="markers",
                             name="Consensus mean", marker=dict(size=11, color=T.SURFACE, line=dict(color=T.INK, width=2.5), symbol="diamond")))
    T.style(fig, "Scenario revenue paths vs reported history", "Fiscal year", "Revenue (USD bn)", height=450, hover="x")
    P0 = scen["base"].projection
    out.append(Chart("scenario_revenue", 8, fig, "Scenario revenue paths vs reported history",
                     f"FY27-28 are anchored to analyst consensus (low / mean / high); beyond that OCI growth decays at a rate estimated from Oracle's own history and fades to the terminal rate. "
                     f"FY31 revenue: bear ${scen['bear'].diagnostics['fy31_revenue']/1e3:,.0f}bn, base ${scen['base'].diagnostics['fy31_revenue']/1e3:,.0f}bn, bull ${scen['bull'].diagnostics['fy31_revenue']/1e3:,.0f}bn "
                     f"(cumulative cloud revenue to FY31 = {scen['bear'].diagnostics['cum_cloud_revenue_to_fy31_pct_of_rpo']:.0%} / {scen['base'].diagnostics['cum_cloud_revenue_to_fy31_pct_of_rpo']:.0%} / "
                     f"{scen['bull'].diagnostics['cum_cloud_revenue_to_fy31_pct_of_rpo']:.0%} of today's RPO - above 100% means new bookings are required).", src, asof))

    # scenario FCF
    fig = go.Figure()
    for s in ("bear", "base", "bull"):
        P = scen[s].projection
        fig.add_trace(go.Bar(x=[y.split("E")[0] + "E" for y in P.index], y=P["ufcf"] / 1e3, name=s.title(), marker_color=SCENARIO[s]))
    T.style(fig, "Scenario unlevered free cash flow", "Fiscal year (FY27 = remaining nine months)", "Unlevered FCF (USD bn)", height=430, hover="x")
    fig.update_layout(barmode="group")
    fcfd = {s: scen[s].diagnostics for s in scen}
    out.append(Chart("scenario_fcf", 8, fig, "Scenario unlevered free cash flow",
                     f"Cash generated after capex and working capital, before financing. Free cash flow stays negative for {fcfd['bear']['negative_fcf_years']} / {fcfd['base']['negative_fcf_years']} / {fcfd['bull']['negative_fcf_years']} "
                     f"periods (bear / base / bull); the largest cumulative funding gap is ${fcfd['bear']['max_cumulative_funding_gap']/1e3:,.0f}bn / ${fcfd['base']['max_cumulative_funding_gap']/1e3:,.0f}bn / "
                     f"${fcfd['bull']['max_cumulative_funding_gap']/1e3:,.0f}bn, to be financed with debt, equity or customer prepayments.", src, asof))

    # scenario leverage
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, subplot_titles=("Net debt / EBITDA (x)", "Interest cover: EBIT / interest (x)"))
    for s in ("bear", "base", "bull"):
        P = scen[s].projection
        x = [y.split("E")[0] + "E" for y in P.index]
        fig.add_trace(go.Scatter(x=x, y=P["net_debt_to_ebitda"], name=s.title(), line=dict(color=SCENARIO[s], width=2.4), legendgroup=s), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=P["interest_cover"].clip(upper=15), name=s.title(), line=dict(color=SCENARIO[s], width=2.4), legendgroup=s, showlegend=False), 2, 1)
    fig.add_hline(y=4.0, line=dict(color=T.RED, width=1, dash="dot"), row=1, col=1, annotation_text="stress limit 4.0x")
    fig.add_hline(y=3.0, line=dict(color=T.RED, width=1, dash="dot"), row=2, col=1, annotation_text="stress floor 3.0x")
    T.style(fig, "Scenario leverage path: can Oracle fund the build-out?", None, None, height=560, hover="x")
    fig.update_xaxes(title_text="Fiscal year", row=2, col=1)
    out.append(Chart("scenario_leverage", 8, fig, "Scenario leverage path: can Oracle fund the build-out?",
                     f"Net debt rolls forward with free cash flow, after-tax interest and dividends (no new equity assumed). Peak net debt/EBITDA is {fcfd['bear']['peak_net_debt_to_ebitda']:.1f}x / {fcfd['base']['peak_net_debt_to_ebitda']:.1f}x / "
                     f"{fcfd['bull']['peak_net_debt_to_ebitda']:.1f}x; minimum interest cover {fcfd['bear']['min_interest_cover']:.1f}x / {fcfd['base']['min_interest_cover']:.1f}x / {fcfd['bull']['min_interest_cover']:.1f}x. "
                     "Dotted red lines are rating-agency-style rules of thumb, stated as assumptions in config.yaml; interest cover is capped at 15x for readability.", src, asof))

    # margins & capex per scenario
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, subplot_titles=("GAAP operating margin (EBIT / revenue)", "Capex / revenue"))
    for s in ("bear", "base", "bull"):
        P = scen[s].projection
        x = [y.split("E")[0] + "E" for y in P.index]
        fig.add_trace(go.Scatter(x=x, y=P["operating_margin"], name=s.title(), line=dict(color=SCENARIO[s], width=2.4), legendgroup=s), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=P["capex_to_revenue"], name=s.title(), line=dict(color=SCENARIO[s], width=2.4), legendgroup=s, showlegend=False), 2, 1)
    T.style(fig, "Scenario operating margin and capital intensity", None, None, height=520, hover="x")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title_text="Fiscal year", row=2, col=1)
    out.append(Chart("scenario_margin_capex", 8, fig, "Scenario operating margin and capital intensity",
                     "Operating margin after the depreciation that capex creates, and capex as a share of revenue. Capex is whatever is needed to take net PP&E/revenue from today's level toward a "
                     "terminal anchor (hyperscaler peers) while revenue grows - so slower revenue growth does not by itself make capex fall (the bear case 'capex remains elevated').", src, asof))

    # scenario vs Monte Carlo
    mc = b.risk["mc"]["garch_t"]
    ret = mc.terminal / 100.0 * price
    fig = go.Figure(go.Histogram(x=ret, nbinsx=80, marker_color="rgba(137,135,129,0.55)", name="Simulated price in 12 months (GARCH-t)"))
    for s in ("bear", "base", "bull"):
        fig.add_vline(x=scen[s].blended, line=dict(color=SCENARIO[s], width=3), annotation_text=f"{s.title()} value ${scen[s].blended:,.0f}", annotation_position="top")
    fig.add_vline(x=price, line=dict(color=T.INK, width=2, dash="dash"), annotation_text=f"Price ${price:,.0f}", annotation_position="bottom")
    T.style(fig, "Fundamental scenario values vs the 12-month market-price distribution", "Share price (USD)", "Simulations", height=430, hover="x")
    out.append(Chart("scenario_vs_mc", 8, fig, "Fundamental scenario values vs the 12-month market-price distribution",
                     "Grey: where a volatility-clustered Monte Carlo (20,000 paths, CAPM drift) says the price could be in 12 months, ignoring fundamentals. Coloured lines: what the fundamental scenarios say the "
                     "business is worth today. They answer different questions, but if the scenario values lie far outside the simulated range (or the price sits at an extreme) the two views disagree.",
                     MODEL + "; Monte Carlo on Yahoo Finance price history", asof))
    return out
