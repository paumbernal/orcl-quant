"""Page 6 (factor model + event study), page 7 (risk & Monte Carlo), page 9 (score, signal, backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from . import theme as T
from .theme import SCENARIO, STATE_COLORS, Chart

MODEL_SRC = "MODEL OUTPUT (this repository)"


def _sl(df, start):
    return df if start is None else df[df.index >= pd.Timestamp(start)]


def build(b, start: str | None = "2018-01-01") -> list[Chart]:
    out: list[Chart] = []
    tkr = b.meta["ticker"]
    prices_src = b.sources["prices"]
    fm = b.factor
    ev = b.event
    esrc = f"{b.sources['events']}; {prices_src}"

    # =============================================================== PAGE 6 : factor model
    roll = _sl(fm.daily.rolling.dropna(), start)
    fig = go.Figure()
    for c, color in zip(("MKT", "TECH", "SEMI", "CLOUD"), (T.BLUE, T.ORANGE, T.VIOLET, T.AQUA)):
        fig.add_trace(go.Scatter(x=roll.index, y=roll[c], name={"MKT": "Market (S&P 500)", "TECH": "Tech tilt (Nasdaq 100 - S&P 500)", "SEMI": "Semis/AI (SMH - Nasdaq 100)", "CLOUD": "Hyperscalers (MSFT/AMZN/GOOGL - Nasdaq 100)"}[c],
                                 line=dict(color=color, width=1.8)))
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    T.style(fig, f"Rolling 252-day factor loadings of {tkr} excess returns", "Date", "Loading (regression coefficient)", height=430)
    f0 = fm.daily.full
    out.append(Chart("rolling_loadings", 6, fig, f"Rolling 252-day factor loadings of {tkr} excess returns",
                     f"Four-factor regression of daily excess returns re-estimated on a rolling year. Full-sample R2 is {f0.r2:.0%} (market-only {fm.daily.capm_only.r2:.0%}); "
                     f"the {1-f0.r2:.0%} remainder is stock-specific. Over the window shown the semis/AI loading ranged {roll['SEMI'].min():.2f} to {roll['SEMI'].max():.2f} and the tech-tilt loading {roll['TECH'].min():.2f} to {roll['TECH'].max():.2f}: "
                     "exposures that move this much are regime-specific rather than structural.",
                     prices_src, T.span(roll.index)))

    sp = fm.window_specs.dropna(subset=["r2"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[s.split(":")[0] for s in sp.index], y=sp["r2"], name="In-sample R2", marker_color=T.BLUE))
    fig.add_trace(go.Bar(x=[s.split(":")[0] for s in sp.index], y=sp["loo_r2"], name="Leave-one-out R2 (out-of-sample proxy)", marker_color=T.ORANGE))
    fig.add_hline(y=0, line=dict(color=T.INK2, width=1))
    T.style(fig, "Overfitting at small N: in-sample vs leave-one-out R2 as variables are added", "Specification (S1 market only ... S4 + RPO and FCF trend)", "R2", height=420, y_tickformat=".0%", hover="x")
    fig.update_layout(barmode="group")
    out.append(Chart("spec_r2", 6, fig, "Overfitting at small N: in-sample vs leave-one-out R2",
                     "Earnings-window regressions of ORCL's excess return on market factors and then on progressively more fundamental variables (S1 to S4). The blue bars rise with every added variable; "
                     f"the orange bars (each point predicted from a model fitted without it) {'fall' if sp['loo_r2'].iloc[-1] < sp['loo_r2'].iloc[0] else 'do not fall'} as variables are added (S1 {sp['loo_r2'].iloc[0]:.0%} -> {sp.index[-1].split(':')[0]} {sp['loo_r2'].iloc[-1]:.0%}), "
                     f"{'which is the signature of overfitting' if sp['loo_r2'].iloc[-1] < sp['loo_r2'].iloc[0] else 'so the extra variables are adding some real information'}. Sample sizes: " +
                     ", ".join(f"{s.split(':')[0]}: n={int(r['n'])}" for s, r in sp.iterrows()) + ".", esrc, f"{fm.window_dataset.index.min().date()} to {fm.window_dataset.index.max().date()} earnings windows"))

    wf = fm.walk_forward
    P = wf.predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P["ridge"], y=P["y"], mode="markers", marker=dict(color=T.BLUE, size=7, opacity=0.6), name="Monthly sample"))
    lim = [min(P.min().min(), -0.3), max(P.max().max(), 0.3)]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=T.MUTED, dash="dash"), name="Perfect forecast (45 degrees)"))
    T.style(fig, f"Walk-forward predictions vs realised {wf.metrics['horizon_days']}-day excess returns (out of sample)", "Predicted excess return", "Realised excess return", height=430, hover="closest")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    m = wf.metrics
    out.append(Chart("wf_scatter", 6, fig, "Walk-forward predictions vs realised returns (out of sample)",
                     f"Each point is a month: the model (ridge regression, refit yearly on past data only, with an embargo so no training label overlaps the test window) predicts the next {m['horizon_days']}-day excess return. "
                     f"Out-of-sample R2 versus a constant historical-mean forecast = {m['oos_r2_vs_hist_mean']:.1%}; information coefficient {m['information_coefficient']:.2f}; "
                     f"Diebold-Mariano p = {wf.dm_pvalue:.2f}. Points on the dashed line would be perfect forecasts; a cloud with no slope means no predictive power.", esrc, f"{m['first_test']} to {m['last_test']} (n={m['n_oos']})"))

    imp = wf.importance.head(10).sort_values("permutation_importance_oos")
    fig = go.Figure(go.Bar(x=imp["permutation_importance_oos"], y=imp.index, orientation="h", marker_color=T.BLUE))
    T.style(fig, "Feature importance (out-of-sample permutation, increase in squared error)", "Increase in test MSE when the feature is shuffled", None, height=400, legend=False, hover="y")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    out.append(Chart("feature_importance", 6, fig, "Feature importance (out-of-sample permutation)",
                     "How much worse the walk-forward predictions get when each feature is randomly shuffled within the test fold. Importance is relative and, with a model that does not beat a constant forecast, "
                     "it ranks inputs by how much the (weak) model leans on them - not by how much they truly drive returns. Never read it as causal.", esrc, f"{m['first_test']} to {m['last_test']}"))

    ct = fm.corr_target.sort_values("spearman_rho")
    fig = go.Figure(go.Bar(x=ct["spearman_rho"], y=ct.index, orientation="h", marker_color=[T.BLUE if p < 0.10 else T.AXIS for p in ct["p_value_adj"]]))
    fig.add_vline(x=0, line=dict(color=T.INK2, width=1))
    T.style(fig, f"Rank correlation of each feature with the next {m['horizon_days']}-day excess return", "Spearman correlation (blue: adj. p < 0.10)", None, height=500, legend=False, hover="y")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    out.append(Chart("feature_corr", 6, fig, "Correlation of each feature with forward returns",
                     "Simple correlations, monthly samples, point-in-time features. P-values are adjusted for overlapping 63-day windows (effective sample ~1/3 of observations). "
                     "Multiple correlations are screened, so a couple will look 'significant' by chance; features with short histories (OCI, RPO) have very few effective observations.", esrc, "monthly samples, 2010-2026"))

    cf = fm.corr_features
    fig = go.Figure(go.Heatmap(z=cf.values, x=list(cf.columns), y=list(cf.index), zmin=-1, zmax=1, colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                               colorbar=dict(title="Spearman"), hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    T.style(fig, "Multicollinearity: correlation between predictive features", None, None, height=560, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickangle=-45)
    out.append(Chart("feature_collinearity", 6, fig, "Multicollinearity: correlation between predictive features",
                     "Growth, valuation and momentum measures overlap heavily (dark red / blue blocks). When regressors move together, individual coefficients become unstable and "
                     "interpretable only as a group - one reason this project reports predictive accuracy and permutation importance rather than coefficient 't-stats' as evidence.", esrc, "monthly samples"))

    # ---- event study
    et = ev.table
    fig = go.Figure(go.Histogram(x=et["car_1d"], nbinsx=30, marker_color=T.BLUE, name="1-day abnormal return"))
    fig.add_vline(x=0, line=dict(color=T.INK, width=1))
    s1 = ev.summary.loc["car_1d"]
    T.style(fig, f"{tkr} 1-day abnormal return on the earnings reaction day", "Abnormal return (market-model, S&P 500)", "Number of earnings events", height=400, legend=False, hover="x", x_tickformat=".0%")
    out.append(Chart("event_hist", 6, fig, f"{tkr} 1-day abnormal return on the earnings reaction day",
                     f"{int(s1['n'])} earnings releases. Mean abnormal return {s1['mean_CAR']:+.2%} (t={s1['t_stat']:.2f}, p={s1['p_value']:.2f}); mean absolute move {s1['mean_abs_CAR']:.1%}; range {s1['min']:+.0%} to {s1['max']:+.0%}. "
                     + (" The mean is statistically indistinguishable from zero while the typical move is large: earnings are a volatility event, not a directional one." if s1["p_value"] > 0.05 else " The mean abnormal return is statistically different from zero."), esrc, f"{et['reaction_date'].min().date()} to {et['reaction_date'].max().date()}"))

    pth = ev.paths
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pth.index, y=pth["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pth.index, y=pth["p25"], fill="tonexty", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name="Inter-quartile range"))
    fig.add_trace(go.Scatter(x=pth.index, y=pth["mean_CAR"], name="Mean CAR", line=dict(color=T.BLUE, width=2.6)))
    fig.add_trace(go.Scatter(x=pth.index, y=pth["median_CAR"], name="Median CAR", line=dict(color=T.ORANGE, width=2, dash="dash")))
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    T.style(fig, "Average cumulative abnormal return around earnings (day 0 = first session after the release)", "Trading days relative to the reaction day", "Cumulative abnormal return", height=430, y_tickformat=".0%")
    out.append(Chart("event_path", 6, fig, "Average cumulative abnormal return around earnings",
                     f"Mean CAR at day 0 / +4 / +19: {pth['mean_CAR'].loc[0]:+.1%} / {pth['mean_CAR'].loc[4]:+.1%} / {pth['mean_CAR'].loc[19]:+.1%}; mean CAR over the five days before the release: {pth['mean_CAR'].loc[-5]:+.1%} at day -5. "
                     "The band shows how wide the typical reaction is. Windows requested in the brief: 1 day (day 0), 3 days (0-2), 5 days (0-4) and 20 days (0-19).", esrc, f"{len(et)} events"))

    d = et[["eps_surprise_pct", "revenue_accel", "car_1d"]].dropna()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("EPS surprise vs consensus", "Revenue-growth acceleration (YoY, pp change)"), horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=et["eps_surprise_pct"], y=et["car_1d"], mode="markers", marker=dict(color=T.BLUE, size=8, opacity=0.7), text=et.index.strftime("%b %Y"),
                             hovertemplate="%{text}<br>surprise %{x:.1f}%<br>reaction %{y:.1%}<extra></extra>", showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=et["revenue_accel"], y=et["car_1d"], mode="markers", marker=dict(color=T.ORANGE, size=8, opacity=0.7), text=et.index.strftime("%b %Y"),
                             hovertemplate="%{text}<br>acceleration %{x:.1%}<br>reaction %{y:.1%}<extra></extra>", showlegend=False), 1, 2)
    T.style(fig, "What explains the size and sign of the reaction?", None, "1-day abnormal return", height=430, legend=False, hover="closest")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(title_text="EPS surprise (%)", row=1, col=1)
    fig.update_xaxes(title_text="Change in YoY revenue growth", tickformat=".0%", row=1, col=2)
    co = ev.correlations
    def rho(var, tgt="signed CAR(0)"):
        r = co[(co["variable"] == var) & (co["target"] == tgt)]
        return (r["spearman_rho"].iloc[0], r["p_value_permutation"].iloc[0], int(r["n"].iloc[0])) if len(r) else (np.nan, np.nan, 0)
    r1, r2_ = rho("eps_surprise_pct"), rho("revenue_accel")
    out.append(Chart("event_scatter", 6, fig, "What explains the size and sign of the reaction?",
                     f"Left: rank correlation between the EPS surprise (reported minus consensus) and the reaction is rho={r1[0]:.2f} (permutation p={r1[1]:.2f}, n={r1[2]}) - "
                     f"{'statistically significant' if r1[1] < 0.05 else 'not statistically significant'}. "
                     f"Right: the change in revenue-growth momentum has rho={r2_[0]:.2f} (p={r2_[1]:.2f}, n={r2_[2]}) - {'statistically significant' if r2_[1] < 0.05 else 'not statistically significant'}. This is a correlation in ~60 events, not a causal estimate; "
                     "revenue surprise and guidance-vs-consensus are not available from free sources, so these are proxies.", esrc, f"{len(et)} events"))

    # =============================================================== PAGE 7 : risk
    rk = b.risk
    r = rk["returns"]
    mu, sd = r.mean(), r.std(ddof=1)
    xs = np.linspace(r.min(), r.max(), 400)
    nu, loc, scale = stats.t.fit(r)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=140, histnorm="probability density", marker_color="rgba(42,120,214,0.45)", name="Observed daily returns"))
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, sd), name="Normal (same mean, s.d.)", line=dict(color=T.ORANGE, width=2.4)))
    fig.add_trace(go.Scatter(x=xs, y=stats.t.pdf(xs, nu, loc, scale), name=f"Student-t (fitted, {nu:.1f} d.o.f.)", line=dict(color=T.AQUA, width=2.4)))
    T.style(fig, f"{tkr} daily return distribution vs normal and Student-t (log density)", "Daily return", "Probability density (log scale)", height=450, hover="x", x_tickformat=".0%")
    fig.update_yaxes(type="log", range=[-3, 2])
    td = rk["tail"]
    out.append(Chart("return_hist", 7, fig, f"{tkr} daily return distribution vs normal and Student-t",
                     f"On a log scale the tails separate: excess kurtosis is {td['excess_kurtosis']:.0f} and skew {td['skew']:+.1f}; there were {td['observed_days_below_-4sd']} days worse than -4 standard deviations where a normal distribution "
                     f"expects {td['normal_expected_days_below_-4sd']:.2f}. {'A normal model therefore understates tail risk by orders of magnitude; the fitted Student-t tracks the tails far better.' if td['excess_kurtosis'] > 3 else 'Tails are close to normal.'}",
                     prices_src, T.span(r.index)))

    qq = stats.probplot(r, dist="norm")
    th, ob = qq[0]
    sel = np.linspace(0, len(th) - 1, 1500).astype(int)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=th[sel], y=ob[sel], mode="markers", marker=dict(color=T.BLUE, size=5), name="Observed quantiles"))
    fig.add_trace(go.Scatter(x=[th.min(), th.max()], y=[qq[1][1] + qq[1][0] * th.min(), qq[1][1] + qq[1][0] * th.max()], line=dict(color=T.ORANGE, width=2), name="Normal reference line"))
    T.style(fig, "Normal QQ plot of daily returns", "Theoretical normal quantile (standard deviations)", "Observed daily return", height=420, hover="closest")
    fig.update_yaxes(tickformat=".0%")
    out.append(Chart("qq_plot", 7, fig, "Normal QQ plot of daily returns",
                     "If returns were normal the dots would lie on the line. The S-shape - dots peeling away at both ends - shows both tails are far heavier than normal. "
                     f"Jarque-Bera test of normality: p = {td['jarque_bera_p']:.1e}.", prices_src, T.span(r.index)))

    vt = rk["var_table"]
    row1 = vt[(vt["horizon_days"] == 1)]
    methods = ["Historical VaR", "Parametric VaR (normal)", "Parametric VaR (Student-t)", "Historical ES", "Parametric ES (normal)", "Parametric ES (Student-t)"]
    fig = go.Figure()
    cols = [T.BLUE, T.ORANGE, T.AQUA, T.VIOLET, T.YELLOW, T.MAGENTA]
    for mth, color in zip(methods, cols):
        fig.add_trace(go.Bar(x=[f"{int(c*100)}% 1-day" for c in row1["confidence"]], y=row1[mth], name=mth, marker_color=color, text=[f"{x:.1%}" for x in row1[mth]], textposition="outside"))
    T.style(fig, "Value-at-Risk and Expected Shortfall by method (1-day horizon)", "Confidence level", "Loss as % of position", height=450, y_tickformat=".0%", hover="x")
    fig.update_layout(barmode="group")
    h99 = row1[row1["confidence"] == 0.99].iloc[0]
    out.append(Chart("var_methods", 7, fig, "Value-at-Risk and Expected Shortfall by method (1-day)",
                     f"VaR is the loss exceeded on 1% (or 5%) of days; Expected Shortfall is the average loss on those worst days. At 99%, the normal-model ES ({h99['Parametric ES (normal)']:.1%}) is "
                     f"{abs(1 - h99['Parametric ES (normal)']/h99['Historical ES']):.0%} {'below' if h99['Parametric ES (normal)'] < h99['Historical ES'] else 'above'} the historical ES ({h99['Historical ES']:.1%}). "
                     f"A model that understates ES is dangerous for risk limits. The Cornish-Fisher estimate is omitted when the skew/kurtosis make its expansion non-monotonic (it is for ORCL). 10-day figures are in the table below.",
                     prices_src, T.span(r.index)))

    bt, frame = rk["var_bt"][("historical", 0.99)]
    fr = _sl(frame, start)
    exc = fr[fr["hit"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr.index, y=fr["ret"], mode="lines", line=dict(color=T.AXIS, width=1), name="Daily return"))
    fig.add_trace(go.Scatter(x=fr.index, y=-fr["var"], line=dict(color=T.ORANGE, width=2), name="-VaR (99%, 252-day historical, known the day before)"))
    fig.add_trace(go.Scatter(x=exc.index, y=exc["ret"], mode="markers", marker=dict(color=T.RED, size=6), name="VaR breaches"))
    T.style(fig, "VaR backtest: 99% one-day historical VaR vs realised returns", "Date", "Daily return", height=430, y_tickformat=".0%", hover="x")
    bn = rk["var_bt"][("normal", 0.99)][0]
    out.append(Chart("var_backtest", 7, fig, "VaR backtest: 99% one-day historical VaR vs realised returns",
                     f"A 99% VaR should be breached about 1% of days. Over {bt.n:,} days the historical model was breached {bt.exceptions} times (expected {bt.expected:.0f}; Kupiec p={bt.kupiec_p:.4f}, verdict: {bt.verdict}); "
                     f"the normal model {bn.exceptions} times (verdict: {bn.verdict}). Breaches cluster in volatile spells, so a trailing-window VaR reacts too slowly.", prices_src, T.span(fr.index)))

    mcs = rk["mc"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Normal (GBM)", "Student-t", "GARCH(1,1)-t", "Block bootstrap"), shared_yaxes=True, horizontal_spacing=0.06, vertical_spacing=0.14)
    for (model, pos) in (("normal", (1, 1)), ("student_t", (1, 2)), ("garch_t", (2, 1)), ("bootstrap", (2, 2))):
        qp = mcs[model].quantile_paths
        x = qp.index
        for lo, hi, a in (("p05", "p95", 0.12), ("p25", "p75", 0.25)):
            fig.add_trace(go.Scatter(x=x, y=qp[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"), *pos)
            fig.add_trace(go.Scatter(x=x, y=qp[lo], fill="tonexty", fillcolor=f"rgba(42,120,214,{a})", line=dict(width=0), showlegend=False, hoverinfo="skip"), *pos)
        fig.add_trace(go.Scatter(x=x, y=qp["p50"], line=dict(color=T.BLUE, width=2), showlegend=False), *pos)
        fig.add_hline(y=100, line=dict(color=T.AXIS, width=1, dash="dot"), row=pos[0], col=pos[1])
    T.style(fig, "Monte Carlo fan charts: 20,000 simulated one-year paths per model (start = 100)", None, None, height=640, legend=False, hover="x")
    fig.update_yaxes(title_text="Value (start = 100)", col=1)
    fig.update_xaxes(title_text="Trading days", row=2)
    st_ = rk["mc_settings"]
    names_ = {"normal": "Normal", "student_t": "Student-t", "garch_t": "GARCH-t", "bootstrap": "Bootstrap"}
    out.append(Chart("mc_fan", 7, fig, "Monte Carlo fan charts: simulated one-year paths",
                     f"{st_['n_sims']:,} paths per model, {st_['horizon_days']} trading days, drift = {st_['drift_mode']} ({st_['drift_annual']:.1%} p.a. = 3m T-bill {st_['rf']:.1%} + beta {st_['beta']:.2f} x ERP {st_['erp']:.0%}); seed {st_['seed']}. "
                     "Shaded bands: 5-95% (light) and 25-75% (dark), line = median. 5th-95th percentile width at the horizon: " +
                     ", ".join(f"{names_[m]} {mcs[m].quantile_paths['p95'].iloc[-1] - mcs[m].quantile_paths['p05'].iloc[-1]:.0f}" for m in mcs) +
                     f". The GARCH model starts from the current conditional volatility ({mcs['garch_t'].params['current_annualised_vol']:.0%} annualised vs {mcs['garch_t'].params['long_run_annualised_vol']:.0%} long-run).",
                     prices_src + "; simulation parameters fitted on full price history", T.span(r.index)))

    fig = go.Figure()
    names = {"normal": "Normal", "student_t": "Student-t", "garch_t": "GARCH(1,1)-t", "bootstrap": "Block bootstrap"}
    colors = {"normal": T.ORANGE, "student_t": T.AQUA, "garch_t": T.BLUE, "bootstrap": T.VIOLET}
    for m_, res in mcs.items():
        rets_ = res.terminal / 100.0 - 1.0
        h, edges = np.histogram(rets_, bins=np.linspace(-0.9, 2.5, 120), density=True)
        fig.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=h, name=names[m_], line=dict(color=colors[m_], width=2.2)))
    fig.add_vline(x=0, line=dict(color=T.INK, width=1))
    T.style(fig, "Distribution of one-year returns by model", "One-year return", "Probability density", height=430, hover="x", x_tickformat=".0%")
    sm = rk["mc_summary"]
    out.append(Chart("mc_terminal", 7, fig, "Distribution of one-year returns by model",
                     f"Same drift, different shape. Probability of loss: " + ", ".join(f"{names[m]} {sm.loc[m,'prob_loss']:.0%}" for m in sm.index) +
                     f". 5th-percentile return: " + ", ".join(f"{names[m]} {sm.loc[m,'p05_return']:.0%}" for m in sm.index) + ". Fat-tailed models put more mass in the extreme left tail than the normal.", prices_src, f"{st_['n_sims']:,} simulations per model"))

    fig = go.Figure()
    for m_, res in mcs.items():
        h, edges = np.histogram(res.max_drawdown, bins=np.linspace(-1, 0, 60), density=True)
        fig.add_trace(go.Scatter(x=(edges[:-1] + edges[1:]) / 2, y=h, name=names[m_], line=dict(color=colors[m_], width=2.2)))
    T.style(fig, "Distribution of the worst peak-to-trough drawdown within a simulated year", "Maximum drawdown in the year", "Probability density", height=410, hover="x", x_tickformat=".0%")
    out.append(Chart("mc_mdd", 7, fig, "Distribution of simulated maximum drawdowns",
                     "Probability that the worst drawdown within the year exceeds 50%: " + ", ".join(f"{names[m]} {sm.loc[m,'prob_drawdown_over_50pct']:.1%}" for m in sm.index) +
                     ". Volatility clustering (GARCH) makes deep drawdowns several times more likely than an i.i.d. normal model implies.", prices_src, f"{st_['n_sims']:,} simulations per model"))

    tc = rk["mc_tails"]
    fig = go.Figure()
    pal = {"Historical (observed)": T.INK, "Normal (theory)": T.ORANGE, "normal": T.MUTED, "student_t": T.AQUA, "garch_t": T.BLUE, "bootstrap": T.VIOLET}
    for idx in tc.index:
        fig.add_trace(go.Bar(x=list(tc.columns), y=tc.loc[idx], name=names.get(idx, idx), marker_color=pal.get(idx, T.MUTED)))
    T.style(fig, "Frequency of large down days: history vs each model", "Daily move below mean by at least k standard deviations", "Share of days (log scale)", height=430, hover="x")
    fig.update_yaxes(type="log", tickformat=".3%")
    fig.update_layout(barmode="group")
    out.append(Chart("tail_freq", 7, fig, "Frequency of large down days: history vs each model",
                     f"How often a day falls k standard deviations below average. A normal distribution gives {tc.loc['Normal (theory)'].iloc[2]:.4%} for -4 sd; history shows {tc.loc['Historical (observed)'].iloc[2]:.3%} "
                     f"({tc.loc['Historical (observed)'].iloc[2] / tc.loc['Normal (theory)'].iloc[2]:.0f}x). The fat-tailed models (Student-t, GARCH, bootstrap) come far closer to the observed frequency than the normal model.", prices_src, T.span(r.index)))

    # =============================================================== PAGE 9 : score, signal, backtest
    sc = _sl(b.score, "2013-01-01")
    live = sc["score_live"]
    close = _sl(b.panel["close"], "2013-01-01")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=close.index, y=close, name=f"{tkr} price", line=dict(color=T.BLUE, width=1.8)), 1, 1)
    fig.add_trace(go.Scatter(x=live.index, y=live, name="Investment score", line=dict(color=T.INK, width=1.8)), 2, 1)
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(27,175,122,0.14)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=40, fillcolor="rgba(227,73,72,0.12)", line_width=0, row=2, col=1)
    T.style(fig, f"Quantitative investment score (0-100) through time vs {tkr} price", None, None, height=560)
    fig.update_yaxes(title_text="Adj. close (USD, log)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Score (0-100)", range=[0, 100], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    last = live.dropna().iloc[-1]
    out.append(Chart("score_history", 9, fig, "Quantitative investment score through time",
                     f"Weighted score of growth, valuation, momentum, cash flow, balance sheet and market-risk percentile ranks, each measured against Oracle's own history using only data available at the time. "
                     f"Green band = 'attractive' (>=60), red = 'unattractive' (<=40). Latest: {last:.1f}. The score is a gauge of how Oracle looks relative to its own past, not a forecast.",
                     b.sources["fundamentals"] + "; " + prices_src, T.span(live.dropna().index)))

    comps = ["growth", "valuation", "momentum", "cash_flow", "balance_sheet", "market_risk"]
    cur = sc[comps].dropna(how="all").iloc[-1]
    cfg_w = {"growth": .25, "valuation": .20, "momentum": .20, "cash_flow": .15, "balance_sheet": .10, "market_risk": .10}
    try:
        from ..config import load_config
        cfg_w = load_config().score.weights.to_dict()
    except Exception:
        pass
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Current component scores (0-100)", "Component scores through time"), horizontal_spacing=0.1)
    fig.add_trace(go.Bar(x=cur.values, y=[f"{c.replace('_', ' ').title()} ({cfg_w[c]:.0%})" for c in comps], orientation="h", marker_color=T.BLUE, showlegend=False,
                         text=[f"{v:.0f}" for v in cur.values], textposition="outside"), 1, 1)
    recent = sc[comps].dropna(how="all")
    recent = recent[recent.index >= recent.index[-1] - pd.DateOffset(years=4)]
    for c, color in zip(comps, (T.BLUE, T.ORANGE, T.AQUA, T.VIOLET, T.YELLOW, T.RED)):
        fig.add_trace(go.Scatter(x=recent.index, y=recent[c], name=c.replace("_", " ").title(), line=dict(color=color, width=1.8)), 1, 2)
    T.style(fig, "What is driving the score?", None, None, height=440, hover="x")
    fig.update_xaxes(range=[0, 110], row=1, col=1, title_text="Score")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=2)
    out.append(Chart("score_components", 9, fig, "What is driving the score?",
                     "Left: current score for each component with its weight. Right: the same components over the last four years. Read the breakdown, not just the total - "
                     "offsetting components (very strong growth, very weak momentum and risk) can average out to a mid-table total that hides a polarised picture.", b.sources["fundamentals"] + "; " + prices_src, T.span(recent.index)))

    bk = b.buckets.table
    fig = go.Figure()
    for h, color in ((126, T.BLUE), (252, T.ORANGE)):
        fig.add_trace(go.Bar(x=list(bk.index), y=bk[f"mean_fwd_{h}d"], name=f"Mean forward {h}-day return", marker_color=color,
                             text=[f"{x:.0%}" for x in bk[f"mean_fwd_{h}d"]], textposition="outside"))
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    T.style(fig, "Did the score historically sort future returns? Mean forward return by score bucket", "Score regime on the date", "Mean forward return", height=430, y_tickformat=".0%", hover="x")
    fig.update_layout(barmode="group")
    ts = b.buckets.spread_tstat
    out.append(Chart("score_buckets", 9, fig, "Did the score historically sort future returns?",
                     f"Average return over the following 126 / 252 trading days, grouped by the score on the starting date. The weights were fixed in advance and never tuned to these results. "
                     f"Attractive-minus-unattractive spread t-statistics (Newey-West, overlapping windows): 126d {ts.get(126, float('nan')):.1f}, 252d {ts.get(252, float('nan')):.1f}. "
                     "Overlapping windows mean far fewer independent observations than days. "
                     + ("The attractive regime did NOT outperform the unattractive regime historically - the score is a description of Oracle versus its own past, not a timing tool."
                        if (np.nan_to_num(ts.get(126, 0)) <= 1.0 and np.nan_to_num(ts.get(252, 0)) <= 1.0) else
                        "The attractive regime outperformed on these windows, but overlapping observations make the apparent significance fragile."), b.sources["fundamentals"] + "; " + prices_src, T.span(live.dropna().index)))

    wr = b.weight_rob
    fig = go.Figure(go.Histogram(x=wr["latest_score"], nbinsx=30, marker_color=T.BLUE, name="Latest score"))
    fig.add_vline(x=last, line=dict(color=T.ORANGE, width=3), annotation_text=f"configured weights: {last:.1f}")
    T.style(fig, "Weight robustness: latest score under 500 random perturbations of the weights", "Latest investment score", "Number of weight draws", height=380, legend=False, hover="x")
    out.append(Chart("weight_robust", 9, fig, "Weight robustness",
                     f"Weights drawn from a Dirichlet distribution centred on the configured weights. The latest score ranges {wr['latest_score'].min():.0f}-{wr['latest_score'].max():.0f} (median {wr['latest_score'].median():.0f}) "
                     f"and the historical series stays {wr['corr_with_base_history'].min():.2f}-{wr['corr_with_base_history'].max():.2f} correlated with the baseline: the conclusion does not hinge on the exact weights.", MODEL_SRC, "weights perturbed; history 2013-2026"))

    sg = b.signal.frame
    s2 = _sl(sg, "2013-06-01").dropna(subset=["composite"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=close.index, y=close, name=f"{tkr} price", line=dict(color=T.BLUE, width=1.6), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=s2.index, y=s2["composite"], name="Composite", line=dict(color=T.INK, width=1.6), showlegend=False), 2, 1)
    thr = b.signal.thresholds
    for k, c in (("strong_bearish", T.RED), ("bearish", "#f2a09f"), ("bullish", "#8fd3b8"), ("strong_bullish", T.AQUA)):
        fig.add_hline(y=thr[k], line=dict(color=c, width=1, dash="dot"), row=2, col=1)
    fig.add_vline(x=pd.Timestamp(b.signal.train_end), line=dict(color=T.MUTED, width=1.5, dash="dash"))
    T.style(fig, "Experimental five-state signal: composite and thresholds (calibrated on the training sample only)", None, None, height=520)
    fig.update_yaxes(title_text="Adj. close (log)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Composite (-1 to +1)", row=2, col=1)
    fig.update_xaxes(title_text="Date (dashed line = end of training sample)", row=2, col=1)
    out.append(Chart("signal_timeline", 9, fig, "Experimental five-state signal: composite and thresholds",
                     f"Equal-weighted average of six point-in-time percentile ranks (momentum, valuation, earnings-surprise proxy, OCI growth, RPO growth, FCF trend). Dotted lines are the 10/30/70/90th percentiles of the composite "
                     f"over the training sample (to {b.signal.train_end.date()}) and define the five states. Latest state: {str(sg['state'].dropna().iloc[-1]).replace('_', ' ')}. Experimental - not a recommendation.",
                     b.sources["fundamentals"] + "; " + prices_src, T.span(s2.index)))

    bt_ = b.backtest
    nav = bt_.nav
    fig = go.Figure()
    palette = [T.BLUE, T.ORANGE, T.MUTED, T.AQUA]
    for i, c in enumerate(nav.columns):
        if c == "Strategy":
            fig.add_trace(go.Scatter(x=nav.index, y=nav[c], name="Strategy (net of costs)", line=dict(color=T.INK, width=3)))
        else:
            fig.add_trace(go.Scatter(x=nav.index, y=nav[c], name=c, line=dict(color=palette[(i - 1) % 4], width=1.6)))
    fig.add_vline(x=pd.Timestamp(b.signal.train_end), line=dict(color=T.MUTED, width=1.5, dash="dash"), annotation_text="end of training sample")
    T.style(fig, "Backtest: growth of 100,000 (execution at next open, 10 bp per side, long-only)", "Date", "Portfolio value (USD, log scale)", height=480)
    fig.update_yaxes(type="log")
    bm = bt_.by_period
    te = bm["Test (out of sample)"]
    out.append(Chart("backtest_nav", 9, fig, "Backtest: growth of 100,000",
                     f"Out of sample (after {b.signal.train_end.date()}): strategy Sharpe {te.loc['Strategy','Sharpe']:.2f} vs ORCL buy-and-hold {te.loc['ORCL buy & hold','Sharpe']:.2f}; "
                     f"max drawdown {te.loc['Strategy','Max drawdown']:.0%} vs {te.loc['ORCL buy & hold','Max drawdown']:.0%}. The comparison that matters most is the exposure-matched ORCL/cash mix - "
                     f"{bt_.placebo['p_value_one_sided']:.0%} of randomly re-ordered versions of the same exposures did at least as well, so there is "
                     f"{'some evidence of' if bt_.placebo['p_value_one_sided'] < 0.05 else 'no statistically meaningful evidence of'} timing skill (one rule, one stock, one regime). Past results do not guarantee anything.",
                     prices_src + "; " + b.sources["rf"], T.span(nav.index)))

    dd2 = pd.DataFrame({c: (nav[c] / nav[c].cummax() - 1) for c in nav.columns})
    fig = go.Figure()
    for i, c in enumerate(dd2.columns):
        fig.add_trace(go.Scatter(x=dd2.index, y=dd2[c], name=("Strategy" if c == "Strategy" else c), line=dict(color=(T.INK if c == "Strategy" else palette[(i - 1) % 4]), width=(2.6 if c == "Strategy" else 1.3))))
    T.style(fig, "Backtest drawdowns", "Date", "Decline from running peak", height=400, y_tickformat=".0%")
    out.append(Chart("backtest_dd", 9, fig, "Backtest drawdowns",
                     "Peak-to-trough losses of the strategy and its benchmarks over the same days. Lower average exposure shortens drawdowns mechanically; compare with the exposure-matched mix to separate de-risking from timing.", prices_src, T.span(dd2.index)))

    cs = bt_.cost_sensitivity
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{i} bp" for i in cs.index], y=cs["Sharpe (full)"], name="Sharpe, full period", marker_color=T.BLUE))
    fig.add_trace(go.Bar(x=[f"{i} bp" for i in cs.index], y=cs["Sharpe (test)"], name="Sharpe, out-of-sample", marker_color=T.ORANGE))
    T.style(fig, "Transaction-cost sensitivity of the strategy Sharpe ratio", "Cost per side (basis points)", "Sharpe ratio", height=380, hover="x")
    fig.update_layout(barmode="group")
    out.append(Chart("cost_sensitivity", 9, fig, "Transaction-cost sensitivity",
                     f"Re-running the identical rule with different trading costs. Moving from 0 to 40 bp per side changes the full-period Sharpe from {cs['Sharpe (full)'].iloc[0]:.2f} to {cs['Sharpe (full)'].iloc[-1]:.2f}; "
                     f"annual turnover is {bt_.metrics.loc['Strategy', 'Annual turnover']:.1f}x, so costs are a second-order effect compared with the strategy's lack of edge over simple benchmarks.", prices_src, T.span(nav.index)))
    return out
