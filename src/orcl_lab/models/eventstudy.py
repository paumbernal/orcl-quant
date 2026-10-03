"""Earnings-announcement event study.

Method (MacKinlay 1997)
-----------------------
* Event day 0 = the first session whose close reflects the release. Oracle publishes after the US close
  (the EDGAR acceptance timestamp, ~16:12 ET, is used to verify this), so day 0 is the next trading day.
* Market model estimated by OLS over the window [-250, -30] trading days before day 0:  r_ORCL = a + b r_MKT + e.
  Abnormal return AR_t = r_t - (a + b m_t). Cumulative abnormal returns over [0,0], [0,2], [0,4], [0,19] give the
  1-, 3-, 5- and 20-day reactions requested. Primary benchmark: S&P 500 (ORCL's own index); robustness: Nasdaq 100 and
  a market-adjusted model (b = 1, a = 0).
* Test statistic per event: CAR / (sigma_e * sqrt(L)); across events: cross-sectional t-test of the mean CAR (robust to
  event-induced variance) plus a sign test and a bootstrap confidence interval.
* The cross-section asks whether the *size* of the reaction is associated with the surprise, growth, RPO/OCI news or guidance.
  With ~60 events (fewer for RPO/OCI/guidance) every such regression is fragile - results are reported with robust standard
  errors, rank correlations, leave-one-out influence checks and explicit sample sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..analytics import returns as R
from ..config import Config


@dataclass
class EventResult:
    table: pd.DataFrame                  # one row per event: AR/CAR windows, test stats, pre-event drift, estimation diagnostics
    summary: pd.DataFrame                # aggregate statistics per window
    robustness: pd.DataFrame             # mean CAR under alternative benchmarks
    cross_section: pd.DataFrame          # regressions / correlations of CAR(0) on surprise & news
    correlations: pd.DataFrame
    influence: pd.DataFrame              # leave-one-out sensitivity of the EPS-surprise coefficient
    paths: pd.DataFrame                  # average cumulative abnormal return path (-5..+20)
    guidance: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _market_model(r: pd.Series, m: pd.Series, t0: int, est: tuple[int, int], min_obs: int):
    lo, hi = t0 + est[0], t0 + est[1]
    if lo < 0:
        return None
    y, x = r.iloc[lo:hi + 1], m.iloc[lo:hi + 1]
    if len(y) < min_obs:
        return None
    X = sm.add_constant(x.values)
    res = sm.OLS(y.values, X).fit()
    return float(res.params[0]), float(res.params[1]), float(np.sqrt(res.scale)), int(len(y)), float(res.rsquared)


def run_event_study(px: pd.DataFrame, ev: pd.DataFrame, q: pd.DataFrame, cfg: Config) -> EventResult:
    es = cfg.event_study
    tkr = cfg.project.ticker
    bench = cfg.data.benchmarks[es.benchmark]
    bench2 = cfg.data.benchmarks[es.robustness_benchmark]
    rets = R.simple_returns(px)
    idx = rets.index
    windows = {k: tuple(v) for k, v in es.windows.to_dict().items()}
    est = tuple(es.estimation_window)
    rows, paths = [], {}
    for pe, row in ev.iterrows():
        d0 = pd.Timestamp(row["reaction_date"])
        if pd.isna(d0) or d0 not in idx:
            continue
        t0 = idx.get_loc(d0)
        mm = _market_model(rets[tkr], rets[bench], t0, est, es.min_estimation_obs)
        if mm is None or t0 + 20 >= len(idx):
            continue
        a, b, s, n_est, r2 = mm
        seg = rets.iloc[t0 - 5: t0 + 20]
        ar = seg[tkr] - (a + b * seg[bench])
        ar0 = ar.reset_index(drop=True)
        ar0.index = range(-5, len(ar0) - 5)
        paths[pe] = ar0.cumsum() - ar0.loc[[i for i in ar0.index if i < 0]].sum()      # CAR path re-based at day 0 start
        rec = {"period_end": pe, "reaction_date": d0, "alpha": a, "beta": b, "sigma": s, "est_obs": n_est, "est_r2": r2}
        for name, (lo, hi) in windows.items():
            L = hi - lo + 1
            car = float(ar0.loc[lo:hi].sum())
            rec[name] = car
            rec[f"{name}_t"] = car / (s * np.sqrt(L))
        rec["car_pre_5d"] = float(ar0.loc[-5:-1].sum())
        # robustness: Nasdaq-100 market model and market-adjusted return
        mm2 = _market_model(rets[tkr], rets[bench2], t0, est, es.min_estimation_obs)
        if mm2:
            ar2 = rets[tkr].iloc[t0:t0 + 20] - (mm2[0] + mm2[1] * rets[bench2].iloc[t0:t0 + 20])
            rec["car_1d_nasdaq"], rec["car_5d_nasdaq"] = float(ar2.iloc[0]), float(ar2.iloc[:5].sum())
        rec["car_1d_mktadj"] = float(rets[tkr].iloc[t0] - rets[bench].iloc[t0])
        rec["raw_return_1d"] = float(rets[tkr].iloc[t0])
        rows.append(rec)
    tab = pd.DataFrame(rows).set_index("period_end")
    # attach event-level news variables (all known at the event)
    join = pd.DataFrame(index=tab.index)
    join["eps_surprise_pct"] = ev["eps_surprise_pct"]
    join["revenue_yoy"] = q["revenue_yoy"]
    join["revenue_accel"] = q["revenue_yoy"] - q["revenue_yoy"].shift(1)
    join["eps_ttm_yoy"] = q["eps_ttm_yoy"]
    join["oci_growth"] = q["oci_growth"]
    join["rpo_yoy"] = q["rpo_yoy"]
    join["rpo_qoq"] = q["rpo_qoq"]
    join["guid_rev_mid"] = (q["guid_rev_growth_lo"] + q["guid_rev_growth_hi"]) / 2
    join["guid_cloud_mid"] = (q["guid_cloud_growth_lo"] + q["guid_cloud_growth_hi"]) / 2
    join["guid_eps_mid"] = (q["guid_eps_nongaap_lo"] + q["guid_eps_nongaap_hi"]) / 2
    join["fcf_margin_chg"] = q["fcf_margin_yoy_change"]
    tab = tab.join(join)

    # ---- aggregate ----
    summ = []
    for name in windows:
        x = tab[name].dropna()
        t_cs = stats.ttest_1samp(x, 0.0)
        boot = np.random.default_rng(3).choice(x.values, size=(5000, len(x))).mean(axis=1)
        summ.append({"window": name, "n": len(x), "mean_CAR": x.mean(), "median_CAR": x.median(), "std": x.std(ddof=1),
                     "t_stat": t_cs.statistic, "p_value": t_cs.pvalue, "boot_ci_low": np.quantile(boot, 0.025), "boot_ci_high": np.quantile(boot, 0.975),
                     "share_positive": (x > 0).mean(), "mean_abs_CAR": x.abs().mean(), "max": x.max(), "min": x.min()})
    summary = pd.DataFrame(summ).set_index("window")
    robust = pd.DataFrame({
        "S&P 500 market model (primary)": [tab["car_1d"].mean(), tab["car_1d"].abs().mean()],
        "Nasdaq 100 market model": [tab["car_1d_nasdaq"].mean(), tab["car_1d_nasdaq"].abs().mean()],
        "Market-adjusted (ORCL - S&P 500)": [tab["car_1d_mktadj"].mean(), tab["car_1d_mktadj"].abs().mean()],
        "Raw return": [tab["raw_return_1d"].mean(), tab["raw_return_1d"].abs().mean()]}, index=["mean 1-day abnormal return", "mean |1-day|"]).T

    # ---- cross-section: what is associated with the reaction? ----
    cs_rows, cor_rows = [], []
    for y in ("car_1d", "car_5d"):
        for name, cols in {"EPS surprise": ["eps_surprise_pct"], "EPS surprise + revenue acceleration": ["eps_surprise_pct", "revenue_accel"],
                           "+ EPS growth": ["eps_surprise_pct", "revenue_accel", "eps_ttm_yoy"],
                           "RPO news (qoq) + OCI growth": ["rpo_qoq", "oci_growth"], "Guidance (next-Q revenue growth midpoint)": ["guid_rev_mid"]}.items():
            d = tab[[y] + cols].dropna()
            if len(d) < len(cols) + 5:
                cs_rows.append({"dependent": y, "model": name, "n": len(d), "note": "insufficient observations"})
                continue
            plain = sm.OLS(d[y], sm.add_constant(d[cols])).fit()
            res = sm.OLS(d[y], sm.add_constant(d[cols])).fit(cov_type="HC3")
            cooks = plain.get_influence().cooks_distance[0]
            top = int(np.argmax(cooks))
            for c in cols:
                cs_rows.append({"dependent": y, "model": name, "n": int(res.nobs), "variable": c, "coef": res.params[c], "robust_t": res.tvalues[c],
                                "p_value": res.pvalues[c], "r2": res.rsquared, "max_cooks_d": float(cooks[top]),
                                "most_influential_event": str(d.index[top].date()),
                                "single_event_dominates": bool(cooks[top] > 4.0 / len(d))})
    for c in ["eps_surprise_pct", "revenue_yoy", "revenue_accel", "eps_ttm_yoy", "oci_growth", "rpo_yoy", "rpo_qoq", "fcf_margin_chg", "guid_rev_mid", "guid_cloud_mid"]:
        for y, ycol in (("signed CAR(0)", "car_1d"), ("|CAR(0)|", "car_1d")):
            d = tab[[c, ycol]].dropna()
            if len(d) < 8:
                continue
            yv = d[ycol].abs() if y.startswith("|") else d[ycol]
            xv = d[c].abs() if y.startswith("|") else d[c]
            rho, p = stats.spearmanr(xv, yv)
            perm = np.random.default_rng(5).permutation
            null = np.array([stats.spearmanr(xv, perm(yv.values))[0] for _ in range(2000)])
            cor_rows.append({"variable": c, "target": y, "n": len(d), "spearman_rho": rho, "p_value_permutation": float((np.abs(null) >= abs(rho)).mean())})
    corr = pd.DataFrame(cor_rows)
    # influence of single events on the EPS-surprise slope
    d = tab[["car_1d", "eps_surprise_pct"]].dropna()
    infl = []
    if len(d) > 15:
        base = sm.OLS(d["car_1d"], sm.add_constant(d["eps_surprise_pct"])).fit().params["eps_surprise_pct"]
        for i in d.index:
            sl = sm.OLS(d["car_1d"].drop(i), sm.add_constant(d["eps_surprise_pct"].drop(i))).fit().params["eps_surprise_pct"]
            infl.append({"left_out": i, "label": q.loc[i, "label"] if i in q.index else "", "slope": sl, "change_vs_full": sl - base, "car_1d": d.loc[i, "car_1d"], "eps_surprise_pct": d.loc[i, "eps_surprise_pct"]})
    influence = pd.DataFrame(infl).set_index("left_out").sort_values("change_vs_full", key=abs, ascending=False) if infl else pd.DataFrame()
    P = pd.DataFrame(paths)
    avg_path = pd.DataFrame({"mean_CAR": P.mean(axis=1), "median_CAR": P.median(axis=1), "p25": P.quantile(0.25, axis=1), "p75": P.quantile(0.75, axis=1)})
    guid = tab[["guid_rev_mid", "guid_cloud_mid", "guid_eps_mid", "car_1d", "car_5d"]].dropna(subset=["guid_rev_mid"])
    notes = [
        f"{len(tab)} earnings events with a valid {-est[0]}-day market-model estimation window; all releases were after the close (reaction day = next session).",
        "Surprise = Yahoo consensus EPS vs reported non-GAAP EPS. Revenue surprise and guidance-vs-consensus are NOT available from free sources, so revenue acceleration and stated guidance midpoints are used as imperfect proxies.",
        "Guidance is parsed from the press release only where Oracle printed it (FY26 onward) - the guidance sample is tiny and purely descriptive.",
        "Large RPO / OCI news exists in only the last 3-5 events; their association with returns cannot be separated from the AI-regime shift.",
    ]
    return EventResult(tab, summary, robust, pd.DataFrame(cs_rows), corr, influence, avg_path, guid, notes)
