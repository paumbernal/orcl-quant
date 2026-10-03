"""Quantitative investment score (0-100).

Design principles
-----------------
* Six components with configurable weights (config.yaml `score.weights`; default Growth 25 / Valuation 20 / Momentum 20 /
  Cash flow 15 / Balance sheet 10 / Market risk 10). The weights are a statement of priorities, **not fitted** - nothing in
  this module looks at forward returns when scoring. Forward returns are only used afterwards, descriptively.
* Each component is the average of expanding-window percentile ranks (ORCL against its OWN past, using only data
  available at the time), so a score of 80 means "better than 80% of everything Oracle had shown up to that date".
  Sub-metrics that did not exist yet (e.g. OCI growth before it was disclosed) are skipped and the remaining ones
  re-weighted; the number of live sub-metrics is reported alongside the score.
* The score is a *relative-to-own-history* gauge. It cannot know that a regime has changed (see ``WHY_IT_MIGHT_FAIL``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config

# (feature column, higher_is_better)
COMPONENTS: dict[str, list[tuple[str, bool]]] = {
    "growth": [("revenue_yoy__rank", True), ("eps_ttm_yoy__rank", True), ("rpo_yoy__rank", True), ("oci_growth__rank", True)],
    "valuation": [("pe_trailing__rank", False), ("ev_ebitda__rank", False), ("ev_revenue__rank", False)],
    "momentum": [("mom_12_1__rank", True), ("price_vs_200dma__rank", True), ("rel_strength_6m_vs_qqq__rank", True)],
    "cash_flow": [("fcf_margin_ttm__rank", True), ("ocf_yoy__rank", True), ("capex_to_ocf_ttm__rank", False)],
    "balance_sheet": [("net_debt_to_ebitda_ttm__rank", False), ("interest_coverage_ttm__rank", True)],
    "market_risk": [("vol_63d__rank", False), ("beta_252d__rank", False), ("drawdown_52w__rank", True)],
}

WHY_IT_MIGHT_FAIL = [
    "It is relative to Oracle's own history. If the business changes regime (Oracle moving from a mature software/database franchise "
    "to a capital-intensive AI-infrastructure builder), 'cheap or strong versus the past' may be the wrong comparison entirely.",
    "Percentile ranks have no memory of the cause: a low valuation rank can mean 'cheap' or 'the market knows something'. "
    "Value-trap behaviour is invisible to the score.",
    "Backward-looking fundamentals. Growth, cash-flow and balance-sheet components use reported quarters, so they lag the "
    "information that actually moves the stock (backlog announcements, financing news, customer-concentration concerns).",
    "Offsetting components can hide risk: strong growth and momentum can mask a deteriorating balance sheet because the total is a weighted average, "
    "not a veto system. Always read the component breakdown, not only the total.",
    "Small effective sample: ~15 years, a few dozen independent quarters, one AI-capex cycle. Any historical relationship between score "
    "and forward returns has wide confidence intervals and overlapping forward windows overstate significance.",
    "Weights are a prior, not an estimate. Different defensible weights give different scores; see the weight-robustness table.",
    "Fixed expanding windows make early-history scores noisy (few observations) and recent-regime scores sluggish (history dominates).",
    "Mean-reversion vs momentum is regime-dependent: the same low score can precede a rebound or a continued fall.",
]


def component_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """0-100 component scores and the number of live sub-metrics for each."""
    out = {}
    live = {}
    for comp, items in COMPONENTS.items():
        vals = []
        for col, higher_better in items:
            if col not in panel:
                continue
            r = panel[col]
            vals.append(r if higher_better else 1.0 - r)   # rank in (0,1]; inverted when lower is better
        if not vals:
            continue
        m = pd.concat(vals, axis=1)
        out[comp] = m.mean(axis=1, skipna=True) * 100.0
        live[comp] = m.notna().sum(axis=1)
    sc = pd.DataFrame(out)
    for c, s in live.items():
        sc[f"{c}__n"] = s
    return sc


def total_score(scores: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    comps = [c for c in weights if c in scores.columns]
    S = scores[comps]
    w = pd.Series({c: weights[c] for c in comps})
    avail = S.notna()
    wsum = (avail * w).sum(axis=1)
    total = (S.fillna(0.0) * w).sum(axis=1) / wsum.replace(0, np.nan)
    out = S.copy()
    out["score"] = total
    out["weight_coverage"] = wsum / w.sum()
    return out


def build_score(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sc = component_scores(panel)
    tot = total_score(sc, cfg.score.weights.to_dict())
    tot = tot.join(sc[[c for c in sc.columns if c.endswith("__n")]])
    # require reasonable coverage before the score is considered "live"
    tot["score_live"] = tot["score"].where(tot["weight_coverage"] >= 0.70)
    return tot


# ---------------------------------------------------------------------------- descriptive evaluation (never used for fitting)
def forward_returns(close: pd.Series, horizons=(63, 126, 252)) -> pd.DataFrame:
    return pd.DataFrame({f"fwd_{h}d": close.shift(-h) / close - 1.0 for h in horizons})


@dataclass
class BucketResult:
    table: pd.DataFrame
    spread_tstat: dict[int, float]
    episodes: pd.DataFrame
    notes: list[str]


def bucket_analysis(score: pd.Series, close: pd.Series, horizons=(63, 126, 252), bearish=40.0, bullish=60.0) -> BucketResult:
    """Average forward returns by score regime. Overlapping windows => t-stats use Newey-West with lag = horizon."""
    import statsmodels.api as sm

    fr = forward_returns(close, horizons)
    df = pd.concat([score.rename("score"), fr], axis=1).dropna(subset=["score"])
    df["bucket"] = pd.cut(df["score"], [-np.inf, bearish, bullish, np.inf], labels=["Unattractive (<=40)", "Neutral (40-60)", "Attractive (>=60)"])
    rows = []
    for b, g in df.groupby("bucket", observed=True):
        row = {"bucket": b, "days": len(g), "share_of_days": len(g) / len(df)}
        for h in horizons:
            x = g[f"fwd_{h}d"].dropna()
            row[f"mean_fwd_{h}d"] = x.mean()
            row[f"hit_rate_{h}d"] = (x > 0).mean() if len(x) else np.nan
            row[f"n_{h}d"] = len(x)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("bucket")
    tstats = {}
    for h in horizons:
        d = df.dropna(subset=[f"fwd_{h}d"]).copy()
        if len(d) < 200:
            tstats[h] = np.nan
            continue
        d["attractive"] = (d["score"] >= bullish).astype(float)
        d["unattractive"] = (d["score"] <= bearish).astype(float)
        X = sm.add_constant(d[["attractive", "unattractive"]])
        res = sm.OLS(d[f"fwd_{h}d"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        tstats[h] = float(res.params["attractive"] - res.params["unattractive"]) / float(np.sqrt(res.cov_params().loc["attractive", "attractive"]
                    + res.cov_params().loc["unattractive", "unattractive"] - 2 * res.cov_params().loc["attractive", "unattractive"]))
    # contiguous regimes
    state = pd.Series(np.where(df["score"] >= bullish, "attractive", np.where(df["score"] <= bearish, "unattractive", "neutral")), index=df.index)
    grp = (state != state.shift()).cumsum()
    ep = []
    for _, seg in state.groupby(grp):
        if seg.iloc[0] == "neutral" or len(seg) < 10:
            continue
        s0, e0 = seg.index[0], seg.index[-1]
        ep.append({"regime": seg.iloc[0], "start": s0, "end": e0, "trading_days": len(seg),
                   "return_during": close.loc[e0] / close.loc[s0] - 1.0,
                   **{f"fwd_{h}d_from_start": df.loc[s0, f"fwd_{h}d"] for h in horizons}})
    notes = [
        "Forward windows overlap, so daily observations are heavily autocorrelated: the effective number of independent observations is far smaller than the day count.",
        "The 'attractive minus unattractive' t-statistics use Newey-West standard errors with lag equal to the horizon.",
        "These are descriptive statistics on the score; the weights were fixed in advance and never tuned on these returns.",
    ]
    return BucketResult(table, tstats, pd.DataFrame(ep), notes)


def weight_robustness(panel: pd.DataFrame, cfg: Config, n: int = 500, concentration: float = 60.0, seed: int = 7) -> pd.DataFrame:
    """Dirichlet-perturb the weights around the configured values and report how much the score and its history move."""
    rng = np.random.default_rng(seed)
    base_w = cfg.score.weights.to_dict()
    names = list(base_w)
    alpha = np.array([base_w[k] for k in names]) * concentration
    sc = component_scores(panel)
    base = total_score(sc, base_w)["score"]
    latest, corr = [], []
    for _ in range(n):
        w = dict(zip(names, rng.dirichlet(alpha)))
        s = total_score(sc, w)["score"]
        latest.append(s.dropna().iloc[-1])
        corr.append(s.corr(base))
    return pd.DataFrame({"latest_score": latest, "corr_with_base_history": corr})


def regime_label(score: float, cfg: Config) -> str:
    if np.isnan(score):
        return "n/a"
    if score >= cfg.score.bullish_threshold:
        return "ATTRACTIVE"
    if score <= cfg.score.bearish_threshold:
        return "UNATTRACTIVE"
    return "NEUTRAL"
