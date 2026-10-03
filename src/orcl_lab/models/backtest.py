"""Honest backtest engine for the experimental signal.

Execution model (no look-ahead, realistic frictions)
----------------------------------------------------
* The state is computed from information available at the CLOSE of day t-1.
* Trades happen at the OPEN of day t. A position held into day t earns the overnight gap (open_t / close_{t-1}) at the OLD
  exposure and the intraday move (close_t / open_t) at the NEW exposure.
* Every unit of turnover costs ``cost_bps_per_side`` (default 10 bp: a conservative mega-cap half-spread + impact + commission).
* The uninvested fraction earns the 3-month T-bill rate. Exposure is long-only in [0, 1]: no leverage, no shorting.
* The signal is evaluated every ``rebalance_days`` sessions and a position is held at least ``min_holding_days``.

Evaluation
----------
* Thresholds were calibrated on the training window only (<= ``backtest.train_end``); the test window is genuinely out of sample.
* Benchmarks: buy-and-hold ORCL, QQQ, SPY, and an *exposure-matched* ORCL/cash mix (constant weight = the strategy's average
  exposure). The last one matters most: a strategy that is simply "less invested" will look better on drawdown and Sharpe in a
  falling market without having any timing skill.
* Skill test: a permutation placebo reshuffles the strategy's own exposure decisions in time (same distribution, same turnover)
  and asks how often a random ordering achieves the observed Sharpe ratio.
* Cost sensitivity and a stationary-bootstrap confidence interval for the Sharpe-ratio difference are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analytics import returns as R
from ..config import Config

TD = 252


def target_schedule(target: pd.Series, rebalance_days: int, min_hold: int) -> pd.Series:
    """Target exposure per day (decided at the previous close) honouring the rebalance cadence and minimum holding period."""
    tgt = target.copy()
    out = pd.Series(np.nan, index=tgt.index)
    cur = np.nan
    last_change = -10**9
    first = tgt.first_valid_index()
    if first is None:
        return out
    start = tgt.index.get_loc(first)
    for i in range(start, len(tgt)):
        if (i - start) % rebalance_days == 0 and not np.isnan(tgt.iloc[i]):
            if np.isnan(cur):
                cur, last_change = tgt.iloc[i], i
            elif tgt.iloc[i] != cur and (i - last_change) >= min_hold:
                cur, last_change = tgt.iloc[i], i
        out.iloc[i] = cur
    return out


def simulate(ohlc: pd.DataFrame, exposure_decided: pd.Series, rf_daily: pd.Series, cost_bps: float) -> pd.DataFrame:
    """Daily P&L. `exposure_decided[t]` is the exposure chosen at close t-1 and executed at the open of t."""
    o, c = ohlc["adj_open"], ohlc["adj_close"]
    prev_c = c.shift(1)
    overnight = o / prev_c - 1.0
    intraday = c / o - 1.0
    pos_new = exposure_decided.reindex(ohlc.index)
    pos_old = pos_new.shift(1)
    valid = pos_new.notna() & pos_old.notna()
    rf = rf_daily.reindex(ohlc.index).ffill()
    ret = pos_old * overnight + pos_new * intraday + (1 - 0.5 * (pos_old + pos_new)) * rf
    turnover = (pos_new - pos_old).abs()
    ret = ret - turnover * cost_bps / 1e4
    out = pd.DataFrame({"ret": ret, "exposure": pos_new, "turnover": turnover})
    return out[valid]


def metrics(ret: pd.Series, rf_daily: pd.Series, exposure: pd.Series | None = None, turnover: pd.Series | None = None) -> dict[str, float]:
    ret = ret.dropna()
    rf = rf_daily.reindex(ret.index).ffill()
    m = {
        "CAGR": R.cagr(ret), "Ann. volatility": R.annualised_vol(ret), "Sharpe": R.sharpe_ratio(ret, rf), "Sortino": R.sortino_ratio(ret, rf),
        "Max drawdown": R.max_drawdown(ret), "Calmar": R.calmar_ratio(ret), "Total return": float((1 + ret).prod() - 1),
        "Start": ret.index[0], "End": ret.index[-1], "Days": len(ret),
    }
    if exposure is not None:
        m["Avg exposure"] = float(exposure.reindex(ret.index).mean())
    if turnover is not None:
        m["Annual turnover"] = float(turnover.reindex(ret.index).sum() / (len(ret) / TD))
    return m


def stationary_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series, n: int = 2000, mean_block: int = 21, seed: int = 9) -> tuple[float, float, float]:
    d = pd.concat([a, b, rf], axis=1, keys=["a", "b", "rf"]).dropna()
    rng = np.random.default_rng(seed)
    N = len(d)
    A, B, RF = d["a"].to_numpy(), d["b"].to_numpy(), d["rf"].to_numpy()
    diffs = np.empty(n)
    p = 1.0 / mean_block
    for k in range(n):
        idx = np.empty(N, dtype=int)
        idx[0] = rng.integers(N)
        jumps = rng.random(N) < p
        starts = rng.integers(N, size=N)
        for t in range(1, N):
            idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % N
        ea, eb = A[idx] - RF[idx], B[idx] - RF[idx]
        diffs[k] = ea.mean() / ea.std(ddof=1) * np.sqrt(TD) - eb.mean() / eb.std(ddof=1) * np.sqrt(TD)
    obs = (A - RF).mean() / (A - RF).std(ddof=1) * np.sqrt(TD) - (B - RF).mean() / (B - RF).std(ddof=1) * np.sqrt(TD)
    return float(obs), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def placebo_sharpe(ohlc, exposure_decided, rf_daily, cost_bps, n=500, seed=21) -> np.ndarray:
    """Re-order the strategy's own exposure *segments* in time (same exposures, same turnover) and recompute the Sharpe."""
    rng = np.random.default_rng(seed)
    e = exposure_decided.dropna()
    seg_id = (e != e.shift()).cumsum()
    segs = [g.values for _, g in e.groupby(seg_id)]
    idx = e.index
    out = np.empty(n)
    for k in range(n):
        order = rng.permutation(len(segs))
        shuffled = np.concatenate([segs[i] for i in order])
        sim = simulate(ohlc.loc[idx[0]:], pd.Series(shuffled, index=idx[: len(shuffled)]), rf_daily, cost_bps)
        out[k] = R.sharpe_ratio(sim["ret"], rf_daily.reindex(sim.index).ffill())
    return out


@dataclass
class BacktestResult:
    daily: pd.DataFrame                    # strategy + benchmark daily returns, exposure
    nav: pd.DataFrame
    metrics: pd.DataFrame                  # rows = strategies, columns = metrics (by period)
    by_period: dict[str, pd.DataFrame]
    state_table: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    sharpe_diff: dict[str, tuple[float, float, float]]
    placebo: dict[str, float]
    signal_frame: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def run_backtest(signal_frame: pd.DataFrame, ohlcv: pd.DataFrame, px: pd.DataFrame, rf_annual: pd.Series, cfg: Config) -> BacktestResult:
    bt = cfg.backtest
    sg = cfg.signal
    tkr = cfg.project.ticker
    rfd = R.daily_rf(rf_annual)
    # state computed at close t-1 acts at the open of t
    target_prev = signal_frame["target_exposure"].shift(1).reindex(ohlcv.index)
    sched = target_schedule(target_prev, int(sg.rebalance_days), int(sg.min_holding_days))
    sim = simulate(ohlcv, sched, rfd, float(bt.cost_bps_per_side))
    # benchmarks over the same days
    r = R.simple_returns(px)
    days = sim.index
    avg_exp = float(sim["exposure"].mean())
    bench = pd.DataFrame({
        "Strategy": sim["ret"],
        "ORCL buy & hold": r[tkr].reindex(days),
        f"Nasdaq 100 ({cfg.data.benchmarks.nasdaq100})": r[cfg.data.benchmarks.nasdaq100].reindex(days),
        f"S&P 500 ({cfg.data.benchmarks.sp500})": r[cfg.data.benchmarks.sp500].reindex(days),
        f"ORCL/cash {avg_exp:.0%} constant mix": avg_exp * r[tkr].reindex(days) + (1 - avg_exp) * rfd.reindex(days).ffill(),
    })
    nav = (1 + bench).cumprod() * float(bt.initial_capital)

    cut = pd.Timestamp(bt.train_end)
    periods = {"Train (<= %s)" % bt.train_end: bench[bench.index <= cut], "Test (out of sample)": bench[bench.index > cut], "Full period": bench}
    by_period = {}
    for name, b in periods.items():
        rows = {}
        for col in b.columns:
            e = sim["exposure"] if col == "Strategy" else None
            t = sim["turnover"] if col == "Strategy" else None
            rows[col] = metrics(b[col], rfd, e, t)
        by_period[name] = pd.DataFrame(rows).T
    # state diagnostics
    st = signal_frame.reindex(days)
    fwd = r[tkr].shift(-1).reindex(days)                                        # next-day return after the close-t state
    state_table = pd.DataFrame({"days": st["state"].value_counts(), "share": st["state"].value_counts(normalize=True)})
    state_table["mean_next_day_ret_bp"] = fwd.groupby(st["state"]).mean() * 1e4
    state_table["ann_vol"] = r[tkr].reindex(days).groupby(st["state"]).std() * np.sqrt(TD)
    state_table = state_table.reindex([s for s in ["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"] if s in state_table.index])
    # cost sensitivity
    cs = {}
    for bp in (0, 5, 10, 20, 40):
        s2 = simulate(ohlcv, sched, rfd, bp)
        cs[bp] = {"Sharpe (full)": R.sharpe_ratio(s2["ret"], rfd.reindex(s2.index).ffill()), "CAGR (full)": R.cagr(s2["ret"]),
                  "Sharpe (test)": R.sharpe_ratio(s2["ret"][s2.index > cut], rfd.reindex(s2.index[s2.index > cut]).ffill()),
                  "CAGR (test)": R.cagr(s2["ret"][s2.index > cut])}
    cost_df = pd.DataFrame(cs).T
    cost_df.index.name = "cost (bp per side)"
    # statistical comparison vs the exposure-matched mix
    mix_col = f"ORCL/cash {avg_exp:.0%} constant mix"
    sd = {}
    for lab, mask in (("Full period", slice(None)), ("Test (out of sample)", bench.index > cut)):
        a, b = bench["Strategy"][mask], bench[mix_col][mask]
        sd[lab] = stationary_bootstrap_sharpe_diff(a, b, rfd.reindex(a.index).ffill(), n=600)
    # placebo (full period)
    pl = placebo_sharpe(ohlcv, sched, rfd, float(bt.cost_bps_per_side), n=300)
    obs = R.sharpe_ratio(sim["ret"], rfd.reindex(sim.index).ffill())
    placebo = {"observed_sharpe": float(obs), "placebo_mean": float(pl.mean()), "placebo_p95": float(np.quantile(pl, 0.95)),
               "p_value_one_sided": float((pl >= obs).mean()), "n": int(len(pl))}
    notes = [
        "Signal thresholds come from the training window only; the test window is out of sample but short and dominated by one AI-capex cycle.",
        "Only ONE specification was run (components, quantiles, exposures were fixed in config.yaml before any result was seen); the number of trials is 1, "
        "so no multiple-testing adjustment is needed - but also no hyper-parameter search was done that could have helped.",
        "Long-only with cash: the strategy cannot profit from falling prices, only avoid them.",
        "Costs are applied to turnover at the open; market impact for larger accounts would be higher.",
        "Past performance of a transparent rule on one stock is not evidence of future skill.",
    ]
    return BacktestResult(bench, nav, by_period["Full period"], by_period, state_table, cost_df, sd, placebo, signal_frame, notes)
