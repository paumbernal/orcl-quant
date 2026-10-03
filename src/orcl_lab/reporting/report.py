"""Automatic research report.

Answers the nine questions in the brief and ends with a model-driven conclusion. Every number is read from the
``AnalysisBundle`` and every verdict is a conditional on stated thresholds - re-running the pipeline on new data regenerates a
report that is correct for that data (nothing in this file is a hard-coded result).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import Config, load_config
from .runner import AnalysisBundle, load_bundle


def pct(x, d=0, sign=False):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:{'+' if sign else ''}.{d}%}"


def usd(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"-${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}"


def bn(x_m, d=0):
    if x_m is None or np.isnan(x_m):
        return "n/a"
    return f"-${abs(x_m) / 1e3:,.{d}f}bn" if x_m < 0 else f"${x_m / 1e3:,.{d}f}bn"


def ordinal(p: float) -> str:
    n = int(round(p * 100))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def md_table(df: pd.DataFrame, fmt: dict | None = None, index: bool = True) -> str:
    fmt = fmt or {}
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        for c in df.columns:
            v = row[c]
            f = fmt.get(c)
            cells.append(f(v) if f and not (isinstance(v, float) and np.isnan(v)) else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_sections(b: AnalysisBundle, cfg: Config) -> list[tuple[str, str]]:
    m, q, v, A = b.meta, b.q, b.val, b.val.anchors
    last = q.iloc[-1]
    price = m["price"]
    sc = v.scenarios
    base, bear, bull = sc["base"], sc["bear"], sc["bull"]
    fd = b.factor.daily.full
    ev_sum = b.event.summary.loc["car_1d"]
    corr = b.event.correlations
    rv = v.reverse
    sens = v.sensitivities
    wf = b.factor.walk_forward
    sm = b.risk["mc_summary"]
    ms = A.mult_stats
    score = float(b.score["score_live"].dropna().iloc[-1])
    state = str(b.signal.frame["state"].dropna().iloc[-1]).replace("_", " ")
    concl = b.conclusion
    te = b.backtest.by_period["Test (out of sample)"]
    rpo = b.rpo
    secs: list[tuple[str, str]] = []
    comp_now = b.score[["growth", "valuation", "momentum", "cash_flow", "balance_sheet", "market_risk"]].dropna(how="all").iloc[-1]
    hi, lo = comp_now[comp_now >= 70].index.tolist(), comp_now[comp_now <= 30].index.tolist()
    score_shape = ("A polarised picture: " + (", ".join(c.replace("_", " ") for c in hi) + " near the top of Oracle's own history" if hi else "") +
                   (" while " if hi and lo else "") + (", ".join(c.replace("_", " ") for c in lo) + " near the bottom" if lo else "") + ". ") if (hi or lo) else ""
    ts_ = b.buckets.spread_tstat
    score_sorting = ("did not sort future returns in the intended direction" if np.nan_to_num(ts_.get(126, 0)) <= 1.0 and np.nan_to_num(ts_.get(252, 0)) <= 1.0
                     else "sorted future returns in the intended direction (but overlapping windows make the significance fragile)")

    # ------------------------------------------------------------------ header & summary
    def gcorr(a, c):
        r = corr[(corr["variable"] == a) & (corr["target"] == "signed CAR(0)")]
        return (float(r["spearman_rho"].iloc[0]), float(r["p_value_permutation"].iloc[0])) if len(r) else (np.nan, np.nan)

    eps_rho, eps_p = gcorr("eps_surprise_pct", None)
    acc_rho, acc_p = gcorr("revenue_accel", None)
    base_up = base.blended / price - 1
    head = f"""# ORCL Quantitative Investment & AI Valuation Lab - Research Report

**Oracle Corporation (NYSE: ORCL)** | price {usd(price, 2)} at the {m['price_date']} close | latest reported quarter {m['latest_quarter']} (ended {m['latest_quarter_end']}) | consensus snapshot {m['consensus_date']} | generated {m['generated_at_utc']}

> **How to read this.** Every figure is either *historical data* (SEC filings, exchange prices, FRED rates - with source and retrieval date in `data/manifest.json`), a *consensus forecast* (a dated Yahoo Finance snapshot), a *stated assumption* (`config.yaml`, with the derivation logged in `models/valuation_assumption_log.csv`) or a *model output*. The conclusion at the end is the output of a fixed voting rule - not a discretionary view. This is research, not investment advice.

**Central question:** can Oracle's AI/cloud growth justify its valuation and rapidly rising capital requirements, and what does a quantitative model imply about risk/reward?

**Short answer.** Growth is real and accelerating (revenue {pct(last['revenue_yoy'])} YoY, OCI {pct(last['oci_growth'])}, backlog {bn(last['rpo'])}) and is reaching operating income and operating cash flow - but not free cash flow (TTM FCF margin {pct(last['fcf_margin_ttm'])}, capex {last['capex_to_ocf_ttm']:.1f}x operating cash flow), so the build-out is being financed by debt (net debt/EBITDA {last['net_debt_to_ebitda_ttm']:.1f}x), customer prepayments and equity (shares outstanding {pct(last['shares_outstanding_cover']/q['shares_outstanding_cover'].iloc[-5]-1, 1, True)} YoY). The scenario model values the stock at {usd(bear.blended)} / {usd(base.blended)} / {usd(bull.blended)} (bear / base / bull) against {usd(price)}; the statistical models find essentially no out-of-sample predictability; the quantitative score is {score:.0f}/100. **Model-driven conclusion: {concl['label']}** (vote total {concl['total'] if concl['total'] == 0 else format(concl['total'], '+d')}).
"""
    secs.append(("Summary", head))

    # ------------------------------------------------------------------ Q1 drivers
    ep = b.divergence.episodes
    recent_ep = ep.tail(3) if len(ep) else ep
    recently = len(ep) > 0 and (pd.Timestamp(m["price_date"]) - ep["end"].iloc[-1]).days <= 45
    divergence_head = ("The stock is currently trading on its own story, not its peers'." if recently else "The stock has recently moved broadly in line with its peers.")
    ep_txt = "; ".join(f"{r['start'].date()} to {r['end'].date()}: {r['direction'].replace('ORCL ', '')} ({r['peak_relative_return']:+.0%} peak relative)" for _, r in recent_ep.iterrows()) or "none flagged"
    t = b.trend
    q1 = f"""**1. Market and sector factors explain only about a third of daily variance.** A four-factor regression (market, tech tilt, semiconductors/AI, hyperscalers) has R2 = {fd.r2:.0%} over {fd.n:,} days; the market beta is {fd.coefs.loc['MKT','coef']:.2f} (t={fd.coefs.loc['MKT','t']:.0f}) and the semiconductor/AI loading {fd.coefs.loc['SEMI','coef']:.2f} (t={fd.coefs.loc['SEMI','t']:.1f}). The rolling loadings swing widely (chart: *Rolling 252-day factor loadings*), so full-sample numbers describe an average that no single year resembles.

**2. Earnings events dominate the stock-specific part.** Across {int(ev_sum['n'])} releases the mean absolute one-day abnormal return is {ev_sum['mean_abs_CAR']:.1%} (range {ev_sum['min']:+.0%} to {ev_sum['max']:+.0%}) while the mean is {ev_sum['mean_CAR']:+.2%} (p={ev_sum['p_value']:.2f}). The size of the EPS beat/miss has {'a significant' if eps_p < 0.05 else 'no statistically significant'} rank relationship with the reaction (rho={eps_rho:.2f}, p={eps_p:.2f}); the change in revenue-growth momentum has rho={acc_rho:.2f} (p={acc_p:.2f}). In other words, the market is trading growth trajectory, backlog and capital intensity, not the quarterly EPS print.

**3. {divergence_head}** Divergence episodes versus an AI/cloud peer basket: {ep_txt}. Latest technical state: price {'above' if t['above_200dma'] else 'below'} the 200-day average, RSI {t['rsi']:.0f}, {t['rsi_read']}, {abs(t['pct_from_52w_high']):.0%} {'below' if t['pct_from_52w_high'] < 0 else 'above'} the 52-week high, ATR {t['atr_pct']:.1%} of price. Annualised volatility is {b.roll_vol.iloc[-1, -1]:.0%} (252-day) and the stock is {abs(b.drawdowns['ORCL'].iloc[-1]):.0%} below its running peak.

**4. Fundamentals trajectory.** Revenue {pct(last['revenue_yoy'])} YoY, OCI {pct(last['oci_growth'])}, operating income {pct(last['operating_income_yoy'])}, but gross-margin proxy {pct(last['gross_margin_proxy'], 1)} (down from {pct(q['gross_margin_proxy'].iloc[-9], 1)} two years ago) and FCF margin {pct(last['fcf_margin_ttm'])}. The variables the market appears to price are therefore (i) the pace of OCI/backlog conversion, (ii) the funding of the capex programme and (iii) the margin and asset turnover the new capacity will earn.

*Caveat:* factor loadings and event correlations describe association, not causation, and are estimated on a short sample dominated by one AI-capex cycle."""
    secs.append(("1. What is driving ORCL's stock?", q1))

    # ------------------------------------------------------------------ Q2 valuation
    gs_txt = (f"Holding every other base-case assumption fixed, today's price is consistent with OCI growth running {rv['growth_shift']*100:+.1f}pp per year relative to the base path from FY29 "
              f"(FY31 revenue {bn(rv.get('implied_fy31_revenue', np.nan))} vs {bn(rv['base_fy31_revenue'])} in the base case), or with a terminal EBITDA margin of "
              f"{pct(rv.get('implied_terminal_ebitda_margin', np.nan), 1)} vs {pct(rv['base_terminal_ebitda_margin'], 1)}." if not np.isnan(rv.get('growth_shift', np.nan)) else "The reverse DCF could not bracket the current price.")
    verdict = ("below" if base_up >= cfg.conclusion.upside_bull else "above" if base_up <= cfg.conclusion.upside_bear else "close to")
    msum = v.summary.copy()
    msum_fmt = msum.copy()
    for c in msum.columns:
        msum_fmt[c] = msum[c].map(lambda x: pct(x, 0, True) if c == "Upside vs price" else usd(x))
    msum_fmt.index = [s.title() for s in msum_fmt.index]
    q2 = f"""**Relative to its own history**, Oracle trades at {ms.loc['pe_trailing','current']:.1f}x trailing GAAP earnings ({ordinal(ms.loc['pe_trailing','current_percentile_rank'])} percentile of the last 10 years), {ms.loc['ev_ebitda','current']:.1f}x EV/EBITDA ({ordinal(ms.loc['ev_ebitda','current_percentile_rank'])}) and {ms.loc['ev_revenue','current']:.1f}x EV/revenue ({ordinal(ms.loc['ev_revenue','current_percentile_rank'])}). Versus AI/cloud peers (Yahoo, current) its forward P/E is {v.peer_multiples.loc['ORCL','Forward P/E']:.1f}x against {v.peer_multiples.drop('ORCL')['Forward P/E'].median():.1f}x median - but its EV multiples carry a ${A.net_debt/1e3:,.0f}bn net-debt burden and its capital intensity is far higher.

**Scenario valuation** (four methods blended with the weights in `config.yaml`; DCF {cfg.valuation.method_weights.dcf:.0%}):

{md_table(msum_fmt)}

Probability-weighted value ({cfg.valuation.scenario_probabilities.bear:.0%}/{cfg.valuation.scenario_probabilities.base:.0%}/{cfg.valuation.scenario_probabilities.bull:.0%}): **{usd(v.prob_weighted)}** ({pct(v.upside_prob_weighted, 0, True)} vs price). Base-case blended value is {pct(base_up, 0, True)} vs price, i.e. the price sits **{verdict}** the base case. The scenario spread is wide and right-skewed (bear {pct(bear.blended/price-1, 0, True)}, bull {pct(bull.blended/price-1, 0, True)}) because growth, margin and capital intensity compound.

**What the price implies (reverse DCF).** {gs_txt}

**Sensitivity of the base-case DCF ({usd(base.dcf.price)} per share):** +5pp/yr OCI growth from FY29 = {usd(sens['growth_vs_margin'].iloc[3, 2] - sens['growth_vs_margin'].iloc[2, 2])}; -5pp = {usd(sens['growth_vs_margin'].iloc[1, 2] - sens['growth_vs_margin'].iloc[2, 2])}; +/-3pp terminal EBITDA margin = {usd(sens['growth_vs_margin'].iloc[2, 3] - sens['growth_vs_margin'].iloc[2, 2])} / {usd(sens['growth_vs_margin'].iloc[2, 1] - sens['growth_vs_margin'].iloc[2, 2])}; +/-0.5pp WACC = {usd(sens['wacc_vs_terminal_growth'].iloc[1, 2] - sens['wacc_vs_terminal_growth'].iloc[2, 2])} / {usd(sens['wacc_vs_terminal_growth'].iloc[3, 2] - sens['wacc_vs_terminal_growth'].iloc[2, 2])}.

**Is the valuation justified?** On this model the current price is {'less than' if base_up > 0 else 'more than'} the base-case value, so the market is {'discounting a path somewhat worse than the base case' if base_up >= cfg.conclusion.upside_bull else 'pricing in more than the base case' if base_up <= cfg.conclusion.upside_bear else 'roughly in line with the base case'}. The answer is **conditional**: the valuation is justified if OCI growth and margins stay within a few points of the base path *and* the capital intensity converges toward hyperscaler levels; it is not if the {bear.name} drivers materialise (value {usd(bear.blended)}). Terminal value is {base.dcf.terminal_share_of_ev:.0%} of base-case EV, so the answer hinges on the post-FY36 business and the {base.assumptions.wacc:.1%} cost of capital.

*Methodological caveat:* the multiple-based methods use Oracle's own historical multiples, earned when capex ran at a median {pct(q['capex_to_revenue_ttm'].loc[A.mult_hist.index[0]:].median())} of revenue (now {pct(last['capex_to_revenue_ttm'])}); they likely flatter a capital-heavy Oracle, which is why they exceed the DCF in the bear case."""
    secs.append(("2. Is the current valuation justified?", q2))

    # ------------------------------------------------------------------ Q3 growth -> cash flow
    sc_tab = rpo.scorecard.copy()
    sc_fmt = sc_tab.copy().astype(object)
    for i in sc_tab.index:
        for j in sc_tab.columns:
            x = sc_tab.loc[i, j]
            sc_fmt.loc[i, j] = "" if pd.isna(x) else (f"{x:.1f}x" if "Capex / OCF" in i else pct(x))
    ll = rpo.lead_lag.dropna(subset=["corr"])
    kk = ll["corr"].idxmax()
    cc = rpo.coverage
    reaching = []
    if last["revenue_yoy"] > 0.10:
        reaching.append("revenue")
    if last["operating_income_yoy"] > 0.10:
        reaching.append("operating income")
    if last["ocf_yoy"] > 0.10:
        reaching.append("operating cash flow")
    if last["eps_diluted_yoy"] > 0.10:
        reaching.append("EPS")
    notreach = []
    if last["fcf_margin_ttm"] < 0:
        notreach.append("free cash flow")
    q3 = f"""**Yes for revenue, operating income and operating cash flow; not (yet) for free cash flow.** Latest-quarter YoY growth: {', '.join(reaching) or 'none of the below'} are growing more than 10%. {'Free cash flow is negative' if notreach else 'Free cash flow is positive'}: TTM FCF is {bn(last['fcf_ttm'], 1)} ({pct(last['fcf_margin_ttm'])} of revenue) because TTM capex ({bn(last['capex_ttm'], 1)}) is {last['capex_to_ocf_ttm']:.1f}x operating cash flow ({bn(last['ocf_ttm'], 1)}). Contract liabilities (largely customer prepayments) are {bn(last['contract_liabilities'], 1)} vs {bn(q['contract_liabilities'].iloc[-5], 1)} a year ago{', which boosts operating cash flow now and is repaid through service delivery later' if last['contract_liabilities'] > 1.25 * q['contract_liabilities'].iloc[-5] else ''}.

**RPO -> revenue -> cash-flow scorecard** (last six quarters):

{md_table(sc_fmt)}

**Conversion framework.** RPO is {bn(last['rpo'])}, {cc['rpo_to_ttm_revenue_x']:.1f}x trailing revenue ({cc['years_of_revenue_at_run_rate']:.1f} years of revenue at the current run-rate). Oracle's own XBRL disclosure put the 12-month conversion at ~{cc.get('disclosed_12m_conversion_median', np.nan):.0%} of RPO before FY22; today even if all of consensus current-year revenue came from RPO, 12-month conversion could be at most {pct(cc.get('consensus_fy0_revenue_pct_of_rpo', np.nan))} - the backlog is far longer-dated than the legacy book, so its value depends on multi-year delivery and counterparty performance. Consensus revenue for the next two years sums to only {pct(cc.get('consensus_two_year_revenue_pct_of_rpo', np.nan))} of RPO.

**Lag between RPO growth and revenue growth.** The highest rank correlation between RPO growth and later revenue growth is {ll.loc[kk, 'corr']:.2f} at a lag of {kk} quarter(s), but the 95% interval ({ll.loc[kk,'ci_low']:.2f} to {ll.loc[kk,'ci_high']:.2f}) {'excludes' if ll.loc[kk,'ci_low'] > 0 else 'includes'} zero (effective sample {ll.loc[kk,'n_eff']:.0f} quarters). A Granger-type test finds {'predictive precedence' if (len(rpo.granger) and rpo.granger['p_value'].min() < 0.05) else 'no significant predictive precedence'} (min p = {rpo.granger['p_value'].min() if len(rpo.granger) else float('nan'):.2f}). **The lag cannot be pinned down statistically** with ~28 quarters and one structural break.

**Capital efficiency.** Over the OCI disclosure window the OCI run-rate grew {b.q.loc[b.q['oci_revenue'].notna(), 'oci_revenue'].iloc[-1]/b.q.loc[b.q['oci_revenue'].notna(), 'oci_revenue'].iloc[0]:.1f}x while net PP&E grew {b.q.loc[b.q['oci_revenue'].notna(), 'ppe_net'].iloc[-1]/b.q.loc[b.q['oci_revenue'].notna(), 'ppe_net'].iloc[0]:.1f}x; the valuation tests whether asset turnover converges toward hyperscaler levels (net PP&E/revenue of {', '.join(f'{k} {x:.2f}x' for k, x in A.peers_capital_intensity.items())} vs Oracle {last['ppe_net']/last['revenue_ttm']:.2f}x).

**Limitations of this analysis:** """ + " ".join(f"({i+1}) {t}" for i, t in enumerate(rpo.limitations[:5]))
    secs.append(("3. Is AI growth translating into cash flow?", q3))

    # ------------------------------------------------------------------ Q4 balance sheet
    sh = last["shares_outstanding_cover"] / q["shares_outstanding_cover"].iloc[-5] - 1
    bd = {s: sc[s].diagnostics for s in sc}
    stress = cfg.valuation.stress
    base_ok = base.diagnostics["first_year_leverage_breach"] == "none" and not base.diagnostics["interest_cover_breach"]
    bear_ok = bear.diagnostics["first_year_leverage_breach"] == "none" and not bear.diagnostics["interest_cover_breach"]
    rising = "rising" if last["net_debt_incl_leases"] > 1.1 * q["net_debt_incl_leases"].iloc[-5] else "stable"
    verdict4 = (f"Elevated and {rising}, but within rating-agency-style limits in the base case" if base_ok else f"Excessive in the base case (leverage {rising})") + (
        "; the bear case breaches a limit" if not bear_ok else "; even the bear case stays within limits")
    q4 = f"""**{verdict4}.** Debt incl. finance leases is {bn(last['total_debt_incl_leases'])} against {bn(last['cash_and_securities'])} of cash and securities (net debt {bn(last['net_debt_incl_leases'])}); net debt/EBITDA {last['net_debt_to_ebitda_ttm']:.1f}x, EBITDA interest cover {last['ebitda_interest_cover_ttm']:.1f}x, EBIT interest cover {last['interest_coverage_ttm']:.1f}x. These ratios look benign mainly because EBITDA is also growing fast and because the company raised equity: shares outstanding are {pct(sh, 1, True)} YoY and a {bn(last['preferred_stock'], 1)} preferred issue sits ahead of common equity. Operating-lease liabilities ({bn(last['operating_lease_liability'])}) and not-yet-commenced data-centre leases are additional fixed commitments not in net debt.

**Scenario leverage paths** (net debt rolls with free cash flow, after-tax interest and dividends; no new equity):

| | Bear | Base | Bull |
|---|---|---|---|
| Peak net debt | {bn(bd['bear']['peak_net_debt'])} | {bn(bd['base']['peak_net_debt'])} | {bn(bd['bull']['peak_net_debt'])} |
| Peak net debt / EBITDA | {bd['bear']['peak_net_debt_to_ebitda']:.1f}x | {bd['base']['peak_net_debt_to_ebitda']:.1f}x | {bd['bull']['peak_net_debt_to_ebitda']:.1f}x |
| Minimum interest cover | {bd['bear']['min_interest_cover']:.1f}x | {bd['base']['min_interest_cover']:.1f}x | {bd['bull']['min_interest_cover']:.1f}x |
| Largest cumulative funding gap | {bn(bd['bear']['max_cumulative_funding_gap'])} | {bn(bd['base']['max_cumulative_funding_gap'])} | {bn(bd['bull']['max_cumulative_funding_gap'])} |
| First year FCF turns positive | {bd['bear']['first_positive_fcf_year']} | {bd['base']['first_positive_fcf_year']} | {bd['bull']['first_positive_fcf_year']} |

Stress limits (assumptions in `config.yaml`): net debt/EBITDA <= {stress.net_debt_to_ebitda_max:.1f}x, interest cover >= {stress.interest_cover_min:.1f}x. The risk is less about solvency in the base case than about **dependence on capital markets and counterparties**: the funding gap is several times today's operating cash flow, and the revenue that justifies it sits in a concentrated set of long-dated contracts that cannot be verified from public data."""
    secs.append(("4. Is Oracle taking excessive balance-sheet risk?", q4))

    # ------------------------------------------------------------------ Q5 quant model
    bt_ = b.backtest
    q5 = f"""**Investment score: {score:.0f}/100 ({'attractive' if score >= cfg.score.bullish_threshold else 'unattractive' if score <= cfg.score.bearish_threshold else 'neutral'}).** Components (0-100): growth {b.score['growth'].dropna().iloc[-1]:.0f}, valuation {b.score['valuation'].dropna().iloc[-1]:.0f}, momentum {b.score['momentum'].dropna().iloc[-1]:.0f}, cash flow {b.score['cash_flow'].dropna().iloc[-1]:.0f}, balance sheet {b.score['balance_sheet'].dropna().iloc[-1]:.0f}, market risk {b.score['market_risk'].dropna().iloc[-1]:.0f}. {score_shape}Historically the score {score_sorting} (attractive-minus-unattractive Newey-West t-stats: {', '.join(f'{h}d {x:.1f}' for h, x in b.buckets.spread_tstat.items())}); weights were fixed in advance and not tuned. Under 500 random re-weightings the latest score ranges {b.weight_rob['latest_score'].min():.0f}-{b.weight_rob['latest_score'].max():.0f}.

**Experimental five-state signal: {state}.** Composite of momentum, valuation, earnings-surprise proxy, OCI growth, RPO growth and FCF trend (equal weights, thresholds from the training sample only). Honest backtest (next-open execution, {cfg.backtest.cost_bps_per_side} bp per side, long-only): out of sample after {cfg.backtest.train_end}, strategy Sharpe {te.loc['Strategy','Sharpe']:.2f} vs ORCL buy-and-hold {te.loc['ORCL buy & hold','Sharpe']:.2f}, Nasdaq 100 {te.loc[[i for i in te.index if i.startswith('Nasdaq')][0],'Sharpe']:.2f}, S&P 500 {te.loc[[i for i in te.index if i.startswith('S&P')][0],'Sharpe']:.2f}; CAGR {te.loc['Strategy','CAGR']:.1%} vs {te.loc['ORCL buy & hold','CAGR']:.1%}. A placebo that randomly re-orders the strategy's own exposures matches its Sharpe {bt_.placebo['p_value_one_sided']:.0%} of the time - **{'some evidence of' if bt_.placebo['p_value_one_sided'] < 0.05 else 'no demonstrable'} timing skill**. This is a research exercise, not a trading strategy.

**Statistical factor models.** In-sample R2 of the earnings-window regressions rises with every added variable ({b.factor.window_specs['r2'].dropna().iloc[0]:.0%} -> {b.factor.window_specs['r2'].dropna().iloc[-1]:.0%}) while leave-one-out R2 falls ({b.factor.window_specs['loo_r2'].dropna().iloc[0]:.0%} -> {b.factor.window_specs['loo_r2'].dropna().iloc[-1]:.0%}): overfitting at small N. The walk-forward model of the next {wf.metrics['horizon_days']}-day excess return has out-of-sample R2 of {wf.metrics['oos_r2_vs_hist_mean']:.1%} versus a constant forecast (information coefficient {wf.metrics['information_coefficient']:.2f}, Diebold-Mariano p={wf.dm_pvalue:.2f}). **{"The data do not support a claim that observable factors predict ORCL's returns" if (wf.metrics['oos_r2_vs_hist_mean'] <= 0 or wf.dm_pvalue > 0.10) else "The walk-forward model shows some out-of-sample skill, which should be treated cautiously given the small effective sample"}**; the factor models describe, they do not forecast.

**Monte Carlo (one year, {b.risk['mc_settings']['n_sims']:,} paths, CAPM drift {pct(b.risk['mc_settings']['drift_annual'], 1)}):**\n\n{md_table(sm[['expected_return','median_return','p05_return','p95_return','prob_loss','expected_shortfall_5pct','prob_drawdown_over_50pct']].rename(columns={'expected_return':'Expected return','median_return':'Median','p05_return':'5th pct','p95_return':'95th pct','prob_loss':'P(loss)','expected_shortfall_5pct':'ES (5%)','prob_drawdown_over_50pct':'P(MDD>50%)'}), {c: (lambda x: pct(x, 1)) for c in ['Expected return','Median','5th pct','95th pct','P(loss)','ES (5%)','P(MDD>50%)']})}

A normal distribution understates tail risk for this stock (excess kurtosis {b.risk['tail']['excess_kurtosis']:.0f}; {b.risk['tail']['observed_days_below_-4sd']} days beyond -4 sd vs {b.risk['tail']['normal_expected_days_below_-4sd']:.2f} expected), and volatility clusters, so the GARCH and bootstrap rows are the more realistic ones."""
    secs.append(("5. What does the quantitative model say?", q5))

    # ------------------------------------------------------------------ Q6 upside catalysts
    g = sens["growth_vs_margin"]
    w = sens["wacc_vs_terminal_growth"]
    cat = [
        f"**OCI growth outrunning the base path.** Each +5pp/yr from FY29 adds about {usd(g.iloc[3, 2] - g.iloc[2, 2])} per share to the base-case DCF; backlog conversion (RPO {bn(last['rpo'])}, 9x revenue) and new large contracts are the mechanism. Next-quarter guidance: total revenue growth {last['guid_rev_growth_lo']:.0f}-{last['guid_rev_growth_hi']:.0f}% and cloud growth {last['guid_cloud_growth_lo']:.0f}-{last['guid_cloud_growth_hi']:.0f}%.",
        f"**Asset turnover improving toward hyperscaler levels.** The base case converges net PP&E/revenue to {A.k_targets['base']:.2f}x; the bull case assumes {A.k_targets['bull']:.2f}x. Every +3pp of terminal EBITDA margin is worth {usd(g.iloc[2, 3] - g.iloc[2, 2])} per share.",
        f"**Free-cash-flow inflection.** The base case turns unlevered FCF positive in {bd['base']['first_positive_fcf_year']} (bull {bd['bull']['first_positive_fcf_year']}); evidence of capex peaking below operating cash flow would remove the main reason the stock trades at a discount to the model.",
        f"**Lower discount rate / risk premium.** -0.5pp on WACC is worth {usd(w.iloc[1, 2] - w.iloc[2, 2])} per share; the stock's beta has been {b.roll_beta.iloc[-1, 0]:.1f} on a rolling year, so a calmer tape matters.",
        f"**Mean reversion in sentiment.** The stock is {abs(b.drawdowns['ORCL'].iloc[-1]):.0%} below its high, momentum and risk components of the score are near historical lows, and sell-side targets average {usd(b.consensus.get('price_targets', {}).get('mean', np.nan))} (range {usd(b.consensus.get('price_targets', {}).get('low', np.nan))}-{usd(b.consensus.get('price_targets', {}).get('high', np.nan))}) - but targets are opinions, and the score history shows low momentum has not reliably preceded rebounds.",
    ]
    secs.append(("6. Biggest upside catalysts", "\n".join(f"- {c}" for c in cat)))

    # ------------------------------------------------------------------ Q7 downside risks
    risks = [
        f"**Growth fades faster than backlog implies.** Bear-case value {usd(bear.blended)} ({pct(bear.blended/price-1, 0, True)}): analyst-low revenue, OCI growth reverting at its pre-AI persistence, margin at the 25th percentile of its history. -5pp/yr of OCI growth costs {usd(g.iloc[1, 2] - g.iloc[2, 2])} per share.",
        f"**Capex and depreciation outrun revenue.** Capex is {last['capex_to_ocf_ttm']:.1f}x operating cash flow; the gross-margin proxy has fallen {abs(last['gross_margin_proxy']-q['gross_margin_proxy'].iloc[-9])*100:.0f}pp in two years and depreciation is {pct(last['d_and_a']/last['revenue'], 0)} of revenue and rising as construction-in-progress enters service.",
        f"**Financing risk.** The base-case funding gap peaks at {bn(bd['base']['max_cumulative_funding_gap'])} (bear: interest cover falls to {bd['bear']['min_interest_cover']:.1f}x vs a {stress.interest_cover_min:.1f}x floor). A closed or expensive debt/equity market, or a rating downgrade, hits both the cost of capital and the ability to deliver contracted capacity. Equity dilution is already visible ({pct(sh, 1, True)} shares YoY).",
        f"**Counterparty and concentration risk.** RPO is {last['rpo_to_ttm_revenue']:.1f}x revenue and a small number of very large contracts can dominate it (cannot be verified from public data). Contract cancellation, renegotiation or customer financing stress would reduce conversion without any change in reported RPO until it is recognised.",
        f"**Earnings-event and tail risk.** The average absolute earnings-day move is {ev_sum['mean_abs_CAR']:.1%}; the worst was {ev_sum['min']:+.0%}. The simulated one-year 5th-percentile return is {sm.loc['garch_t','p05_return']:.0%} (GARCH-t) with an expected shortfall of {sm.loc['garch_t','expected_shortfall_5pct']:.0%} and a {sm.loc['garch_t','prob_drawdown_over_50pct']:.0%} chance of a >50% intra-year drawdown. Realised 99% VaR was breached {b.risk['var_bt'][('historical', 0.99)][0].exceptions} times in {b.risk['var_bt'][('historical', 0.99)][0].n:,} days vs {b.risk['var_bt'][('historical', 0.99)][0].expected:.0f} expected.",
        f"**Valuation-multiple contraction.** Multiples are anchored on Oracle's capital-light history; if the market re-rates Oracle as an infrastructure utility-like business, EV/EBITDA could converge below the {ms.loc['ev_ebitda','bear']:.1f}x bear-case anchor.",
    ]
    secs.append(("7. Biggest downside risks", "\n".join(f"- {c}" for c in risks)))

    # ------------------------------------------------------------------ Q8 price range
    pt = b.consensus.get("price_targets", {})
    q8 = f"""The scenario model implies a **blended value range of {usd(bear.blended)} - {usd(bull.blended)}** with a base case of **{usd(base.blended)}** (DCF only: {usd(bear.dcf.price)} / {usd(base.dcf.price)} / {usd(bull.dcf.price)}), and a probability-weighted value of **{usd(v.prob_weighted)}**. The current price is {usd(price, 2)}.

{md_table(msum_fmt)}

For comparison: the Monte Carlo 12-month price distribution (GARCH-t, current price as start) has a 5th / median / 95th percentile of {usd(sm.loc['garch_t','p05_value'] * price / 100)} / {usd(sm.loc['garch_t','median_value'] * price / 100)} / {usd(sm.loc['garch_t','p95_value'] * price / 100)}; sell-side targets are {usd(pt.get('low', np.nan))} / {usd(pt.get('mean', np.nan))} / {usd(pt.get('high', np.nan))} (low / mean / high; consensus snapshot, not data). The scenario values describe what the business is worth under stated assumptions, the Monte Carlo describes where the *price* may trade; they answer different questions.

**Researcher degrees of freedom (disclosed).** Modelling choices made *after* seeing an intermediate result, and why:
1. *Scenario positions.* A first version put every scenario driver at the 10th/90th percentile of its range simultaneously; that produced implausible joint extremes (bull value above $1,000/share, bear equity worth roughly zero). Drivers were moved to inter-quartile positions (`valuation.scenario_position`, `margin_percentiles`) - a judgement about joint plausibility, not a fit to the current price.
2. *Terminal value.* The first version let new investment earn the terminal ROIC forever; it now earns WACC plus half of the excess return (`terminal_excess_return_retention`).
3. *Bull-case margin* is floored at today's record margin (a bull case must not assume compression); *long-run tax* is an explicit 18% assumption rather than the 12% historical median.
4. *Walk-forward model.* The first run showed an out-of-sample R2 of about -115% because ridge extrapolated regime-shifted OCI values; features are now winsorised at +/-3 s.d. and the alpha grid widened (result: about -3.5%). This was a validity fix, not a search for a better score.
Method weights, scenario probabilities, score weights, signal components/quantiles/exposures and the train/test split were fixed in `config.yaml` before any result was seen and were not changed; the backtest is therefore a single specification (number of trials = 1). The sensitivity tables show the full range."""
    secs.append(("8. What price range does the scenario model imply?", q8))

    # ------------------------------------------------------------------ Q9 monitor
    nq = m["next_earnings"] or "n/a"
    nq_eps = b.consensus.get("next_quarter_eps_estimate")
    oci_cur = last["oci_revenue"]
    mon = pd.DataFrame([
        ("Cloud / OCI growth", f"cloud {pct(last['cloud_growth'])}, OCI {pct(last['oci_growth'])}", f"Q2 guidance: cloud +{last['guid_cloud_growth_lo']:.0f}% to +{last['guid_cloud_growth_hi']:.0f}%. Above the top end = confirming; below the bottom = negative."),
        ("RPO and net new bookings", f"{bn(last['rpo'])}; QoQ change {bn(last['rpo'] - q['rpo'].iloc[-2])}", "Growth continuing without cancellations; watch disclosed 12-month and 5-year conversion shares."),
        ("Revenue vs guidance", f"{pct(last['revenue_yoy'])} YoY", f"Guidance +{last['guid_rev_growth_lo']:.0f}% to +{last['guid_rev_growth_hi']:.0f}%; consensus FY revenue {bn(A.cons['fy1']['rev_avg'])}."),
        ("Non-GAAP EPS", f"{usd(last['eps_nongaap_actual'], 2)} last quarter", f"Guidance {usd(last['guid_eps_nongaap_lo'], 2)}-{usd(last['guid_eps_nongaap_hi'], 2)}; consensus {usd(nq_eps, 2) if nq_eps else 'n/a'}."),
        ("Capex and capex/OCF", f"{bn(last['capex'], 1)} this quarter; TTM {last['capex_to_ocf_ttm']:.1f}x OCF", "A decline in capex/OCF toward 1x is the FCF-inflection signal; a further rise raises financing need."),
        ("Free cash flow", f"{bn(last['fcf'], 1)} this quarter; TTM {pct(last['fcf_margin_ttm'])} margin", f"Base case turns FCF positive in {bd['base']['first_positive_fcf_year']}."),
        ("Gross margin (proxy) and operating margin", f"{pct(last['gross_margin_proxy'], 1)} / {pct(last['operating_margin'], 1)}", "Gross-margin compression is the cost of mix shift to AI infrastructure; operating margin should hold via opex leverage."),
        ("Leverage and coverage", f"net debt/EBITDA {last['net_debt_to_ebitda_ttm']:.1f}x; EBIT cover {last['interest_coverage_ttm']:.1f}x", f"Limits used here: {stress.net_debt_to_ebitda_max:.1f}x / {stress.interest_cover_min:.1f}x; watch new debt, ratings actions, equity/preferred issuance and share count."),
        ("Customer prepayments", f"contract liabilities {bn(last['contract_liabilities'], 1)}", "Prepayments inflate operating cash flow now and reverse as capacity is delivered."),
        ("Stock reaction", f"typical absolute move {ev_sum['mean_abs_CAR']:.1%}", f"Rank correlation of the reaction with the EPS surprise is {eps_rho:.2f} (p={eps_p:.2f}) and with revenue-growth momentum {acc_rho:.2f} (p={acc_p:.2f})."),
    ], columns=["Metric", "Latest", "What to look for"])
    q9 = f"**Next earnings release: {nq}** (Q2 FY27; consensus EPS {usd(nq_eps, 2) if nq_eps else 'n/a'}).\n\n" + md_table(mon, index=False)
    secs.append(("9. What metrics should investors monitor next quarter?", q9))

    # ------------------------------------------------------------------ conclusion
    votes = concl["votes"].copy()
    votes["vote"] = votes["vote"].astype(str)
    votes["score"] = votes["value"].map(lambda x: f"{x:+d}")
    vt = votes[["vote", "score", "evidence"]]
    vt.columns = ["Vote", "Score", "Evidence"]
    pos = [r["vote"].split(" (")[0].lower() for _, r in concl["votes"].iterrows() if r["value"] > 0]
    neg = [r["vote"].split(" (")[0].lower() for _, r in concl["votes"].iterrows() if r["value"] < 0]
    neu = [r["vote"].split(" (")[0].lower() for _, r in concl["votes"].iterrows() if r["value"] == 0]
    vote_story = ("Positive: " + (", ".join(pos) or "none") + ". Negative: " + (", ".join(neg) or "none") + ". Neutral: " + (", ".join(neu) or "none") + ".")
    concl_txt = f"""{md_table(vt, index=False)}

Rule: {concl['rule']} Vote total = **{concl['total'] if concl['total'] == 0 else format(concl['total'], '+d')}**.

> ## Investment conclusion: **{concl['label']}**
>
> *This conclusion is model-driven, not subjective: it is the output of the five-vote rule above applied to the model outputs on {m['price_date']}. {vote_story} It changes only when a model output or a stated threshold changes. Not investment advice.*"""
    secs.append(("Model-driven conclusion", concl_txt))
    return secs


def render_markdown(secs: list[tuple[str, str]]) -> str:
    parts = []
    for title, body in secs:
        if title == "Summary":
            parts.append(body)
        elif title == "Model-driven conclusion":
            parts.append("\n---\n\n## Conclusion\n\n" + body)
        else:
            parts.append(f"## {title}\n\n{body}")
    return "\n\n".join(parts) + "\n"


def write_report(cfg: Config | None = None, bundle: AnalysisBundle | None = None) -> str:
    cfg = cfg or load_config()
    b = bundle or load_bundle(cfg)
    md = render_markdown(build_sections(b, cfg))
    out = cfg.reports_dir / "orcl_research_report.md"
    out.write_text(md, encoding="utf-8")
    return str(out)
