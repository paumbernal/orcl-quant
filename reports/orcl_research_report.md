# ORCL Quantitative Investment & AI Valuation Lab - Research Report

**Oracle Corporation (NYSE: ORCL)** | price $142.30 at the 2026-10-02 close | latest reported quarter FY27 Q1 (ended 2026-08-31) | consensus snapshot 2026-10-03 | generated 2026-10-03 15:29 UTC

> **How to read this.** Every figure is either *historical data* (SEC filings, exchange prices, FRED rates - with source and retrieval date in `data/manifest.json`), a *consensus forecast* (a dated Yahoo Finance snapshot), a *stated assumption* (`config.yaml`, with the derivation logged in `models/valuation_assumption_log.csv`) or a *model output*. The conclusion at the end is the output of a fixed voting rule - not a discretionary view. This is research, not investment advice.

**Central question:** can Oracle's AI/cloud growth justify its valuation and rapidly rising capital requirements, and what does a quantitative model imply about risk/reward?

**Short answer.** Growth is real and accelerating (revenue 30% YoY, OCI 121%, backlog $664bn) and is reaching operating income and operating cash flow - but not free cash flow (TTM FCF margin -40%, capex 1.6x operating cash flow), so the build-out is being financed by debt (net debt/EBITDA 2.6x), customer prepayments and equity (shares outstanding +6.4% YoY). The scenario model values the stock at $75 / $169 / $366 (bear / base / bull) against $142; the statistical models find essentially no out-of-sample predictability; the quantitative score is 37/100. **Model-driven conclusion: Neutral** (vote total 0).


## 1. What is driving ORCL's stock?

**1. Market and sector factors explain only about a third of daily variance.** A four-factor regression (market, tech tilt, semiconductors/AI, hyperscalers) has R2 = 34% over 4,212 days; the market beta is 1.03 (t=31) and the semiconductor/AI loading 0.07 (t=1.8). The rolling loadings swing widely (chart: *Rolling 252-day factor loadings*), so full-sample numbers describe an average that no single year resembles.

**2. Earnings events dominate the stock-specific part.** Across 62 releases the mean absolute one-day abnormal return is 6.2% (range -13% to +35%) while the mean is +0.15% (p=0.89). The size of the EPS beat/miss has no statistically significant rank relationship with the reaction (rho=0.11, p=0.38); the change in revenue-growth momentum has rho=0.34 (p=0.01). In other words, the market is trading growth trajectory, backlog and capital intensity, not the quarterly EPS print.

**3. The stock is currently trading on its own story, not its peers'.** Divergence episodes versus an AI/cloud peer basket: 2026-07-14 to 2026-07-29: underperforming peers (-32% peak relative); 2026-07-31 to 2026-08-11: underperforming peers (-28% peak relative); 2026-08-14 to 2026-09-22: underperforming peers (-42% peak relative). Latest technical state: price below the 200-day average, RSI 48, in the neutral 30-70 band, 54% below the 52-week high, ATR 4.8% of price. Annualised volatility is 56% (252-day) and the stock is 56% below its running peak.

**4. Fundamentals trajectory.** Revenue 30% YoY, OCI 121%, operating income 57%, but gross-margin proxy 60.4% (down from 70.6% two years ago) and FCF margin -40%. The variables the market appears to price are therefore (i) the pace of OCI/backlog conversion, (ii) the funding of the capex programme and (iii) the margin and asset turnover the new capacity will earn.

*Caveat:* factor loadings and event correlations describe association, not causation, and are estimated on a short sample dominated by one AI-capex cycle.

## 2. Is the current valuation justified?

**Relative to its own history**, Oracle trades at 22.3x trailing GAAP earnings (41st percentile of the last 10 years), 15.6x EV/EBITDA (42nd) and 7.4x EV/revenue (65th). Versus AI/cloud peers (Yahoo, current) its forward P/E is 12.9x against 22.3x median - but its EV multiples carry a $97bn net-debt burden and its capital intensity is far higher.

**Scenario valuation** (four methods blended with the weights in `config.yaml`; DCF 40%):

|  | DCF | Forward P/E | EV/EBITDA | EV/Revenue | Blended | Upside vs price |
|---|---|---|---|---|---|---|
| Bear | $24 | $86 | $127 | $115 | $75 | -47% |
| Base | $166 | $162 | $190 | $160 | $169 | +19% |
| Bull | $543 | $267 | $264 | $211 | $366 | +157% |

Probability-weighted value (25%/50%/25%): **$195** (+37% vs price). Base-case blended value is +19% vs price, i.e. the price sits **below** the base case. The scenario spread is wide and right-skewed (bear -47%, bull +157%) because growth, margin and capital intensity compound.

**What the price implies (reverse DCF).** Holding every other base-case assumption fixed, today's price is consistent with OCI growth running -2.7pp per year relative to the base path from FY29 (FY31 revenue $235bn vs $247bn in the base case), or with a terminal EBITDA margin of 38.8% vs 41.3%.

**Sensitivity of the base-case DCF ($166 per share):** +5pp/yr OCI growth from FY29 = $51; -5pp = -$42; +/-3pp terminal EBITDA margin = $28 / -$28; +/-0.5pp WACC = $22 / -$19.

**Is the valuation justified?** On this model the current price is less than the base-case value, so the market is discounting a path somewhat worse than the base case. The answer is **conditional**: the valuation is justified if OCI growth and margins stay within a few points of the base path *and* the capital intensity converges toward hyperscaler levels; it is not if the bear drivers materialise (value $75). Terminal value is 78% of base-case EV, so the answer hinges on the post-FY36 business and the 9.7% cost of capital.

*Methodological caveat:* the multiple-based methods use Oracle's own historical multiples, earned when capex ran at a median 4% of revenue (now 105%); they likely flatter a capital-heavy Oracle, which is why they exceed the DCF in the bear case.

## 3. Is AI growth translating into cash flow?

**Yes for revenue, operating income and operating cash flow; not (yet) for free cash flow.** Latest-quarter YoY growth: revenue, operating income, operating cash flow, EPS are growing more than 10%. Free cash flow is negative: TTM FCF is -$28.7bn (-40% of revenue) because TTM capex ($75.7bn) is 1.6x operating cash flow ($46.9bn). Contract liabilities (largely customer prepayments) are $30.8bn vs $13.4bn a year ago, which boosts operating cash flow now and is repaid through service delivery later.

**RPO -> revenue -> cash-flow scorecard** (last six quarters):

|  | FY25 Q4 | FY26 Q1 | FY26 Q2 | FY26 Q3 | FY26 Q4 | FY27 Q1 |
|---|---|---|---|---|---|---|
| RPO growth | 41% | 359% | 438% | 324% | 363% | 46% |
| OCI growth | 52% | 55% | 68% | 84% | 93% | 121% |
| Revenue growth | 11% | 12% | 14% | 22% | 21% | 30% |
| Operating income growth | 9% | 7% | 12% | 25% | 20% | 57% |
| Operating cash flow growth | 1% | 10% | 58% | 21% | 137% | 184% |
| EPS growth (GAAP) | 7% | -2% | 91% | 25% | 22% | 54% |
| FCF margin (TTM) | -1% | -10% | -22% | -39% | -35% | -40% |
| Capex / OCF (TTM) | 1.0x | 1.3x | 1.6x | 2.1x | 1.7x | 1.6x |
| Operating margin | 32% | 29% | 29% | 32% | 32% | 35% |
| Gross margin (proxy) | 70% | 67% | 67% | 65% | 65% | 60% |

**Conversion framework.** RPO is $664bn, 9.3x trailing revenue (8.6 years of revenue at the current run-rate). Oracle's own XBRL disclosure put the 12-month conversion at ~61% of RPO before FY22; today even if all of consensus current-year revenue came from RPO, 12-month conversion could be at most 14% - the backlog is far longer-dated than the legacy book, so its value depends on multi-year delivery and counterparty performance. Consensus revenue for the next two years sums to only 33% of RPO.

**Lag between RPO growth and revenue growth.** The highest rank correlation between RPO growth and later revenue growth is 0.80 at a lag of 1 quarter(s), but the 95% interval (-0.19 to 0.98) includes zero (effective sample 5 quarters). A Granger-type test finds no significant predictive precedence (min p = 0.46). **The lag cannot be pinned down statistically** with ~28 quarters and one structural break.

**Capital efficiency.** Over the OCI disclosure window the OCI run-rate grew 8.2x while net PP&E grew 10.4x; the valuation tests whether asset turnover converges toward hyperscaler levels (net PP&E/revenue of AMZN 0.62x, GOOGL 0.65x, MSFT 1.02x vs Oracle 1.78x).

**Limitations of this analysis:** (1) Small sample: RPO is only reported (in XBRL) from FY19 Q1, so growth-rate relationships rest on roughly 28 quarters. (2) Structural break: RPO rose by >$300bn in a single quarter (FY26 Q1). A handful of observations dominates every correlation. (3) RPO is contracted, not recognised: it can be cancelled/renegotiated, depends on customers' financing and on Oracle delivering data-centre capacity, and its timing profile is only partly disclosed. (4) Customer concentration: a small number of very large AI-infrastructure contracts can account for a large share of RPO; this cannot be verified from public XBRL data. (5) OCI revenue is disclosed only in the earnings releases (rounded to $0.1bn before FY26; exact in the FY26+ tables), starting FY23 Q1.

## 4. Is Oracle taking excessive balance-sheet risk?

**Elevated and rising, but within rating-agency-style limits in the base case; the bear case breaches a limit.** Debt incl. finance leases is $135bn against $37bn of cash and securities (net debt $97bn); net debt/EBITDA 2.6x, EBITDA interest cover 6.6x, EBIT interest cover 4.5x. These ratios look benign mainly because EBITDA is also growing fast and because the company raised equity: shares outstanding are +6.4% YoY and a $5.0bn preferred issue sits ahead of common equity. Operating-lease liabilities ($35bn) and not-yet-commenced data-centre leases are additional fixed commitments not in net debt.

**Scenario leverage paths** (net debt rolls with free cash flow, after-tax interest and dividends; no new equity):

| | Bear | Base | Bull |
|---|---|---|---|
| Peak net debt | $177bn | $238bn | $250bn |
| Peak net debt / EBITDA | 3.4x | 3.0x | 2.9x |
| Minimum interest cover | 2.2x | 3.9x | 6.2x |
| Largest cumulative funding gap | $46bn | $93bn | $100bn |
| First year FCF turns positive | FY30E | FY30E | FY31E |

Stress limits (assumptions in `config.yaml`): net debt/EBITDA <= 4.0x, interest cover >= 3.0x. The risk is less about solvency in the base case than about **dependence on capital markets and counterparties**: the funding gap is several times today's operating cash flow, and the revenue that justifies it sits in a concentrated set of long-dated contracts that cannot be verified from public data.

## 5. What does the quantitative model say?

**Investment score: 37/100 (unattractive).** Components (0-100): growth 89, valuation 33, momentum 4, cash flow 34, balance sheet 25, market risk 3. A polarised picture: growth near the top of Oracle's own history while momentum, balance sheet, market risk near the bottom. Historically the score did not sort future returns in the intended direction (attractive-minus-unattractive Newey-West t-stats: 63d -0.2, 126d -2.6, 252d -2.2); weights were fixed in advance and not tuned. Under 500 random re-weightings the latest score ranges 26-48.

**Experimental five-state signal: BULLISH.** Composite of momentum, valuation, earnings-surprise proxy, OCI growth, RPO growth and FCF trend (equal weights, thresholds from the training sample only). Honest backtest (next-open execution, 10 bp per side, long-only): out of sample after 2022-12-31, strategy Sharpe 0.31 vs ORCL buy-and-hold 0.47, Nasdaq 100 1.28, S&P 500 1.11; CAGR 9.8% vs 17.5%. A placebo that randomly re-orders the strategy's own exposures matches its Sharpe 45% of the time - **no demonstrable timing skill**. This is a research exercise, not a trading strategy.

**Statistical factor models.** In-sample R2 of the earnings-window regressions rises with every added variable (32% -> 62%) while leave-one-out R2 falls (24% -> -69%): overfitting at small N. The walk-forward model of the next 63-day excess return has out-of-sample R2 of -3.5% versus a constant forecast (information coefficient 0.08, Diebold-Mariano p=0.53). **The data do not support a claim that observable factors predict ORCL's returns**; the factor models describe, they do not forecast.

**Monte Carlo (one year, 20,000 paths, CAPM drift 9.6%):**

|  | Expected return | Median | 5th pct | 95th pct | P(loss) | ES (5%) | P(MDD>50%) |
|---|---|---|---|---|---|---|---|
| normal | 10.6% | 5.0% | -38.1% | 78.9% | 44.0% | -45.5% | 3.8% |
| student_t | 10.3% | 6.0% | -36.6% | 72.2% | 42.3% | -47.2% | 3.4% |
| garch_t | 10.1% | 4.6% | -44.4% | 78.6% | 44.3% | -60.9% | 9.8% |
| bootstrap | 9.9% | 6.4% | -37.0% | 70.1% | 41.9% | -45.4% | 3.9% |

A normal distribution understates tail risk for this stock (excess kurtosis 33; 14 days beyond -4 sd vs 0.13 expected), and volatility clusters, so the GARCH and bootstrap rows are the more realistic ones.

## 6. Biggest upside catalysts

- **OCI growth outrunning the base path.** Each +5pp/yr from FY29 adds about $51 per share to the base-case DCF; backlog conversion (RPO $664bn, 9x revenue) and new large contracts are the mechanism. Next-quarter guidance: total revenue growth 30-34% and cloud growth 65-71%.
- **Asset turnover improving toward hyperscaler levels.** The base case converges net PP&E/revenue to 1.02x; the bull case assumes 0.86x. Every +3pp of terminal EBITDA margin is worth $28 per share.
- **Free-cash-flow inflection.** The base case turns unlevered FCF positive in FY30E (bull FY31E); evidence of capex peaking below operating cash flow would remove the main reason the stock trades at a discount to the model.
- **Lower discount rate / risk premium.** -0.5pp on WACC is worth $22 per share; the stock's beta has been 2.0 on a rolling year, so a calmer tape matters.
- **Mean reversion in sentiment.** The stock is 56% below its high, momentum and risk components of the score are near historical lows, and sell-side targets average $238 (range $110-$400) - but targets are opinions, and the score history shows low momentum has not reliably preceded rebounds.

## 7. Biggest downside risks

- **Growth fades faster than backlog implies.** Bear-case value $75 (-47%): analyst-low revenue, OCI growth reverting at its pre-AI persistence, margin at the 25th percentile of its history. -5pp/yr of OCI growth costs -$42 per share.
- **Capex and depreciation outrun revenue.** Capex is 1.6x operating cash flow; the gross-margin proxy has fallen 10pp in two years and depreciation is 17% of revenue and rising as construction-in-progress enters service.
- **Financing risk.** The base-case funding gap peaks at $93bn (bear: interest cover falls to 2.2x vs a 3.0x floor). A closed or expensive debt/equity market, or a rating downgrade, hits both the cost of capital and the ability to deliver contracted capacity. Equity dilution is already visible (+6.4% shares YoY).
- **Counterparty and concentration risk.** RPO is 9.3x revenue and a small number of very large contracts can dominate it (cannot be verified from public data). Contract cancellation, renegotiation or customer financing stress would reduce conversion without any change in reported RPO until it is recognised.
- **Earnings-event and tail risk.** The average absolute earnings-day move is 6.2%; the worst was -13%. The simulated one-year 5th-percentile return is -44% (GARCH-t) with an expected shortfall of -61% and a 10% chance of a >50% intra-year drawdown. Realised 99% VaR was breached 66 times in 3,960 days vs 40 expected.
- **Valuation-multiple contraction.** Multiples are anchored on Oracle's capital-light history; if the market re-rates Oracle as an infrastructure utility-like business, EV/EBITDA could converge below the 13.1x bear-case anchor.

## 8. What price range does the scenario model imply?

The scenario model implies a **blended value range of $75 - $366** with a base case of **$169** (DCF only: $24 / $166 / $543), and a probability-weighted value of **$195**. The current price is $142.30.

|  | DCF | Forward P/E | EV/EBITDA | EV/Revenue | Blended | Upside vs price |
|---|---|---|---|---|---|---|
| Bear | $24 | $86 | $127 | $115 | $75 | -47% |
| Base | $166 | $162 | $190 | $160 | $169 | +19% |
| Bull | $543 | $267 | $264 | $211 | $366 | +157% |

For comparison: the Monte Carlo 12-month price distribution (GARCH-t, current price as start) has a 5th / median / 95th percentile of $79 / $149 / $254; sell-side targets are $110 / $238 / $400 (low / mean / high; consensus snapshot, not data). The scenario values describe what the business is worth under stated assumptions, the Monte Carlo describes where the *price* may trade; they answer different questions.

**Researcher degrees of freedom (disclosed).** Modelling choices made *after* seeing an intermediate result, and why:
1. *Scenario positions.* A first version put every scenario driver at the 10th/90th percentile of its range simultaneously; that produced implausible joint extremes (bull value above $1,000/share, bear equity worth roughly zero). Drivers were moved to inter-quartile positions (`valuation.scenario_position`, `margin_percentiles`) - a judgement about joint plausibility, not a fit to the current price.
2. *Terminal value.* The first version let new investment earn the terminal ROIC forever; it now earns WACC plus half of the excess return (`terminal_excess_return_retention`).
3. *Bull-case margin* is floored at today's record margin (a bull case must not assume compression); *long-run tax* is an explicit 18% assumption rather than the 12% historical median.
4. *Walk-forward model.* The first run showed an out-of-sample R2 of about -115% because ridge extrapolated regime-shifted OCI values; features are now winsorised at +/-3 s.d. and the alpha grid widened (result: about -3.5%). This was a validity fix, not a search for a better score.
Method weights, scenario probabilities, score weights, signal components/quantiles/exposures and the train/test split were fixed in `config.yaml` before any result was seen and were not changed; the backtest is therefore a single specification (number of trials = 1). The sensitivity tables show the full range.

## 9. What metrics should investors monitor next quarter?

**Next earnings release: 2026-12-10** (Q2 FY27; consensus EPS $1.89).

| Metric | Latest | What to look for |
|---|---|---|
| Cloud / OCI growth | cloud 62%, OCI 121% | Q2 guidance: cloud +65% to +71%. Above the top end = confirming; below the bottom = negative. |
| RPO and net new bookings | $664bn; QoQ change $26bn | Growth continuing without cancellations; watch disclosed 12-month and 5-year conversion shares. |
| Revenue vs guidance | 30% YoY | Guidance +30% to +34%; consensus FY revenue $90bn. |
| Non-GAAP EPS | $1.92 last quarter | Guidance $1.85-$1.93; consensus $1.89. |
| Capex and capex/OCF | $28.5bn this quarter; TTM 1.6x OCF | A decline in capex/OCF toward 1x is the FCF-inflection signal; a further rise raises financing need. |
| Free cash flow | -$5.4bn this quarter; TTM -40% margin | Base case turns FCF positive in FY30E. |
| Gross margin (proxy) and operating margin | 60.4% / 34.8% | Gross-margin compression is the cost of mix shift to AI infrastructure; operating margin should hold via opex leverage. |
| Leverage and coverage | net debt/EBITDA 2.6x; EBIT cover 4.5x | Limits used here: 4.0x / 3.0x; watch new debt, ratings actions, equity/preferred issuance and share count. |
| Customer prepayments | contract liabilities $30.8bn | Prepayments inflate operating cash flow now and reverse as capacity is delivered. |
| Stock reaction | typical absolute move 6.2% | Rank correlation of the reaction with the EPS surprise is 0.11 (p=0.38) and with revenue-growth momentum 0.34 (p=0.01). |


---

## Conclusion

| Vote | Score | Evidence |
|---|---|---|
| Valuation (scenario-implied value vs price) | +1 | base +19%, probability-weighted +37% (thresholds +/-15%) |
| Quantitative score | -1 | score 37.5 (bull >= 60, bear <= 40) |
| Experimental signal | +1 | state BULLISH |
| Cash conversion | -1 | TTM FCF margin -40%, YoY change -30pp |
| Balance sheet | +0 | net debt/EBITDA 2.6x (limit 4.0x); base path breach: False; bear path breach: True |

Rule: Bull if the vote total >= +2, Bear if <= -2, otherwise Neutral (five votes, each -1/0/+1). Vote total = **0**.

> ## Investment conclusion: **Neutral**
>
> *This conclusion is model-driven, not subjective: it is the output of the five-vote rule above applied to the model outputs on 2026-10-02. Positive: valuation, experimental signal. Negative: quantitative score, cash conversion. Neutral: balance sheet. It changes only when a model output or a stated threshold changes. Not investment advice.*
