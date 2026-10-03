"""Page 2 - price, technicals, peers, drawdowns, volatility, beta, correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme as T
from .theme import Chart


def _sl(df, start):
    return df if start is None else df[df.index >= pd.Timestamp(start)]


def _src(b):
    return b.sources["prices"]


def build(b, start: str | None = "2018-01-01") -> list[Chart]:
    out: list[Chart] = []
    tkr = b.meta["ticker"]
    tech = _sl(b.tech, start)
    src = _src(b)

    # 1 -------------------------------------------------------------------- price + moving averages + volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=tech.index, y=tech["close"], name=f"{tkr} adj. close", line=dict(color=T.BLUE, width=2)), 1, 1)
    for col, color in (("ma_20", T.AQUA), ("ma_50", T.ORANGE), ("ma_100", T.VIOLET), ("ma_200", T.INK2)):
        fig.add_trace(go.Scatter(x=tech.index, y=tech[col], name=col.replace("ma_", "") + "-day MA", line=dict(color=color, width=1.4)), 1, 1)
    fig.add_trace(go.Bar(x=tech.index, y=tech["volume"] / 1e6, name="Volume (m shares)", marker_color=T.AXIS, showlegend=False), 2, 1)
    T.style(fig, f"{tkr} price with 20/50/100/200-day moving averages", None, None, height=520)
    fig.update_yaxes(title_text="Adjusted close (USD, log scale)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Volume (m)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    out.append(Chart("price_ma", 2, fig, f"{tkr} price with 20/50/100/200-day moving averages",
                     "Dividend-adjusted close (log scale, so equal percentage moves look equal) against four moving averages. Price above a rising 200-day average "
                     f"is the classic definition of an established uptrend; a close below it marks a downtrend regime. Latest bar ({b.meta['price_date']}): price is "
                     f"{'ABOVE' if b.trend['above_200dma'] else 'BELOW'} its 200-day average and the 50-day is {'above' if b.trend['ma50_over_ma200'] else 'below'} the 200-day.",
                     src, T.span(tech.index)))

    # 2 -------------------------------------------------------------------- RSI + MACD
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.08,
                        subplot_titles=("RSI (14, Wilder)", "MACD (12, 26, 9)"))
    fig.add_trace(go.Scatter(x=tech.index, y=tech["rsi"], name="RSI", line=dict(color=T.BLUE, width=1.6)), 1, 1)
    for lvl, c in ((70, T.RED), (30, T.AQUA)):
        fig.add_hline(y=lvl, line=dict(color=c, width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Bar(x=tech.index, y=tech["macd_hist"], name="MACD histogram", marker_color=T.AXIS), 2, 1)
    fig.add_trace(go.Scatter(x=tech.index, y=tech["macd"], name="MACD line", line=dict(color=T.BLUE, width=1.5)), 2, 1)
    fig.add_trace(go.Scatter(x=tech.index, y=tech["macd_signal"], name="Signal line", line=dict(color=T.ORANGE, width=1.5)), 2, 1)
    T.style(fig, "Momentum oscillators: RSI and MACD", None, None, height=520)
    fig.update_yaxes(title_text="RSI (0-100)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="MACD (USD)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    out.append(Chart("rsi_macd", 2, fig, "Momentum oscillators: RSI and MACD",
                     "RSI measures the balance of recent up and down closes: above 70 is stretched to the upside, below 30 stretched to the downside - useful for "
                     "short-horizon context, not as a standalone signal. MACD is the gap between a fast and slow EMA; the histogram turning positive means upward momentum is rebuilding.",
                     src, T.span(tech.index)))

    # 3 -------------------------------------------------------------------- ATR + relative volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                        subplot_titles=("Average true range as % of price (14-day)", "Relative volume (vs prior 20-day average)"))
    fig.add_trace(go.Scatter(x=tech.index, y=tech["atr_pct"], name="ATR %", line=dict(color=T.BLUE, width=1.6)), 1, 1)
    fig.add_trace(go.Bar(x=tech.index, y=tech["rel_volume"], name="Relative volume", marker_color=T.AXIS), 2, 1)
    fig.add_hline(y=1.0, line=dict(color=T.INK2, width=1, dash="dot"), row=2, col=1)
    T.style(fig, "Daily range (ATR) and participation (relative volume)", None, None, height=480, legend=False)
    fig.update_yaxes(title_text="ATR / price", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Volume / 20d average (x)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    out.append(Chart("atr_relvol", 2, fig, "Daily range (ATR) and participation (relative volume)",
                     "ATR is the typical daily high-to-low range as a share of price - it translates volatility into a position-sizing or stop distance. Relative volume above 1 means "
                     "more shares traded than usual; earnings gaps appear as sharp spikes, confirming that those moves had real participation.",
                     b.sources["prices"], T.span(tech.index)))

    # 4 -------------------------------------------------------------------- rebased performance
    px = _sl(b.px, start)
    reb = px / px.iloc[0] * 100
    fig = go.Figure()
    for c in ["SPY", "QQQ", "SMH", "NVDA", "MSFT", "AMZN", "GOOGL"]:
        if c in reb:
            fig.add_trace(go.Scatter(x=reb.index, y=reb[c], name=T.lab(c), line=dict(color=T.ENTITY[c], width=1.3)))
    fig.add_trace(go.Scatter(x=reb.index, y=reb[tkr], name=tkr, line=dict(color=T.BLUE, width=3)))
    T.style(fig, f"Relative performance: {tkr} vs benchmarks and AI / cloud peers (rebased to 100)", "Date", "Total-return index (start = 100, log scale)", height=480)
    fig.update_yaxes(type="log")
    out.append(Chart("rebased", 2, fig, f"Relative performance: {tkr} vs benchmarks and AI / cloud peers",
                     "All series start at 100 on the first date shown and use dividend-adjusted prices, on a log axis so equal percentage moves are equal distances. The question is "
                     f"not just who ended higher but when the paths separated. Over the window shown ORCL returned {reb[tkr].iloc[-1]/100-1:+.0%} versus {reb['QQQ'].iloc[-1]/100-1:+.0%} for the Nasdaq 100 and {reb['SPY'].iloc[-1]/100-1:+.0%} for the S&P 500.",
                     src, T.span(reb.index)))

    # 5 -------------------------------------------------------------------- drawdowns
    dd = _sl(b.drawdowns, start)
    fig = go.Figure()
    for c in ("SPY", "QQQ", "NVDA"):
        fig.add_trace(go.Scatter(x=dd.index, y=dd[c], name=T.lab(c), line=dict(color=T.ENTITY[c], width=1.2)))
    fig.add_trace(go.Scatter(x=dd.index, y=dd[tkr], name=tkr, line=dict(color=T.BLUE, width=2.2), fill="tozeroy", fillcolor="rgba(42,120,214,0.12)"))
    T.style(fig, f"Drawdown from running peak: {tkr} vs benchmarks", "Date", "Decline from previous high", height=430, y_tickformat=".0%")
    out.append(Chart("drawdowns", 2, fig, f"Drawdown from running peak: {tkr} vs benchmarks",
                     "How far each asset sits below its own all-time-high at every date - the investor's lived experience of risk. Depth and duration both matter: "
                     "a -65% drawdown needs a +186% gain to recover.",
                     src, T.span(dd.index)))

    # 6 -------------------------------------------------------------------- rolling volatility
    rv = _sl(b.roll_vol, start)
    fig = go.Figure()
    for c, color in zip(rv.columns, (T.AXIS, T.ORANGE, T.BLUE)):
        fig.add_trace(go.Scatter(x=rv.index, y=rv[c], name=c, line=dict(color=color, width=1.6 if c != "252-day" else 2.4)))
    T.style(fig, f"{tkr} rolling annualised volatility", "Date", "Annualised volatility (sqrt(252) x daily s.d.)", height=400, y_tickformat=".0%")
    out.append(Chart("rolling_vol", 2, fig, f"{tkr} rolling annualised volatility",
                     "Standard deviation of daily returns scaled to a year, over three look-back windows. Volatility clusters: calm spells end abruptly with earnings gaps. "
                     "The 21-day line reacts fastest; the 252-day line is the regime level a risk model should respect.",
                     src, T.span(rv.index)))

    # 7 -------------------------------------------------------------------- rolling beta
    rb = _sl(b.roll_beta, start).dropna()
    fig = go.Figure()
    for c, color in zip(rb.columns, (T.MUTED, T.ORANGE)):
        fig.add_trace(go.Scatter(x=rb.index, y=rb[c], name=c, line=dict(color=color, width=2)))
    fig.add_hline(y=1.0, line=dict(color=T.AXIS, width=1, dash="dot"))
    T.style(fig, f"{tkr} rolling 252-day beta", "Date", "Beta (slope of daily ORCL return on index return)", height=400)
    cap = b.capm
    out.append(Chart("rolling_beta", 2, fig, f"{tkr} rolling 252-day beta",
                     f"Sensitivity of ORCL's daily return to the index over trailing 252 days. Full-sample CAPM beta vs the S&P 500 is {cap.beta:.2f} "
                     f"(t={cap.beta_tstat:.0f}, R2={cap.r_squared:.0%}); the rolling line shows how unstable that single number is - range {rb.iloc[:, 0].min():.2f} to {rb.iloc[:, 0].max():.2f}, latest {rb.iloc[-1, 0]:.2f}.",
                     src, T.span(rb.index)))

    # 8 -------------------------------------------------------------------- correlation heatmap
    cm = b.corr.copy()
    cm.index = [T.lab(c) for c in cm.index]
    cm.columns = [T.lab(c) for c in cm.columns]
    fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=0, zmax=1, colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
                               text=np.round(cm.values, 2), texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", colorbar=dict(title="Correlation")))
    T.style(fig, "Correlation of daily returns (full common history)", None, None, height=470, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    out.append(Chart("corr_heatmap", 2, fig, "Correlation of daily returns (full common history)",
                     f"Pairwise correlation of daily total returns. ORCL's correlation with the S&P 500 is {b.corr.loc[tkr, 'SPY']:.2f} and with the Nasdaq 100 {b.corr.loc[tkr, 'QQQ']:.2f}; "
                     f"the average correlation among the four peers is {b.corr.loc[['NVDA','MSFT','AMZN','GOOGL'], ['NVDA','MSFT','AMZN','GOOGL']].values[np.triu_indices(4, 1)].mean():.2f}. "
                     "A low correlation means much of the stock's variance is its own (earnings gaps, financing news) rather than a shared 'AI trade' factor.",
                     src, T.span(b.rets.index)))

    # 9 -------------------------------------------------------------------- rolling correlation
    rc = _sl(b.roll_corr, start).dropna()
    fig = go.Figure()
    for c in rc.columns:
        fig.add_trace(go.Scatter(x=rc.index, y=rc[c], name=T.lab(c), line=dict(color=T.ENTITY[c], width=1.5)))
    T.style(fig, f"{tkr} rolling 126-day correlation with benchmarks and peers", "Date", "Correlation of daily returns", height=420)
    out.append(Chart("rolling_corr", 2, fig, f"{tkr} rolling 126-day correlation with benchmarks and peers",
                     "A half-year rolling window of ORCL's correlation with each comparator. Correlations are not constant: they spike in market-wide sell-offs "
                     "(diversification fails when needed most) and fall when ORCL trades on its own news.",
                     src, T.span(rc.index)))

    # 10 ------------------------------------------------------------------- divergence
    dv = b.divergence
    z = _sl(dv.zscore.dropna(), start)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z.index, y=z, name="z-score of 63-day relative return", line=dict(color=T.BLUE, width=1.8)))
    thr = 1.5
    for s in (thr, -thr):
        fig.add_hline(y=s, line=dict(color=T.RED if s < 0 else T.AQUA, width=1, dash="dot"))
    ep = dv.episodes
    if not ep.empty:
        for _, r in ep[ep["end"] >= z.index[0]].iterrows():
            fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor="rgba(137,135,129,0.18)", line_width=0)
    T.style(fig, f"Divergence of {tkr} from its AI / cloud peer basket (NVDA, MSFT, AMZN, GOOGL)", "Date", "Z-score (standard deviations)", height=420, legend=False)
    out.append(Chart("divergence", 2, fig, f"Divergence of {tkr} from its AI / cloud peer basket",
                     "Rolling 63-day return of ORCL minus an equal-weight peer basket, standardised against its own history up to each date (no look-ahead). "
                     "Shaded bands are episodes where |z| stayed above 1.5 for at least a week: the stock was materially out- or under-performing its peers.",
                     src, T.span(z.index)))

    # 11 ------------------------------------------------------------------- calendar-year returns
    cy = b.calendar_returns[["ORCL", "SPY", "QQQ"]].tail(12)
    fig = go.Figure()
    for c in cy.columns:
        fig.add_trace(go.Bar(x=cy.index.astype(str), y=cy[c], name=T.lab(c), marker_color=T.ENTITY[c]))
    T.style(fig, "Calendar-year total returns", "Calendar year (latest is year-to-date)", "Total return", height=400, y_tickformat=".0%", hover="x")
    fig.update_layout(barmode="group")
    out.append(Chart("calendar_returns", 2, fig, "Calendar-year total returns",
                     f"Year-by-year return of ORCL against the S&P 500 and Nasdaq 100 (latest bar is year-to-date). ORCL's best calendar year in the window was {cy['ORCL'].idxmax()} ({cy['ORCL'].max():+.0%}) "
                     f"and its worst {cy['ORCL'].idxmin()} ({cy['ORCL'].min():+.0%}).",
                     src, f"{cy.index.min()} to {b.meta['price_date']}"))

    # 12 ------------------------------------------------------------------- CAPM scatter
    r = b.rets[[tkr, "SPY"]].dropna().tail(1260)
    xs = np.array([r["SPY"].min(), r["SPY"].max()])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=r["SPY"], y=r[tkr], mode="markers", marker=dict(color=T.BLUE, size=4, opacity=0.35), name="Daily returns"))
    c5 = np.polyfit(r["SPY"], r[tkr], 1)
    fig.add_trace(go.Scatter(x=xs, y=c5[0] * xs + c5[1], name=f"OLS fit: beta = {c5[0]:.2f}", line=dict(color=T.ORANGE, width=2.5)))
    T.style(fig, f"{tkr} vs S&P 500: daily returns, last 5 years (beta regression)", "S&P 500 (SPY) daily return", f"{tkr} daily return", height=430, hover="closest")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    out.append(Chart("capm_scatter", 2, fig, f"{tkr} vs S&P 500: daily returns, last 5 years",
                     f"Each dot is one trading day. The slope of the fitted line is the 5-year beta ({c5[0]:.2f}); the vertical scatter around it is the stock-specific part. "
                     f"R-squared over this window is {np.corrcoef(r['SPY'], r[tkr])[0, 1] ** 2:.0%}; the dots furthest from the line are mostly earnings reactions.",
                     src, T.span(r.index)))
    return out
