"""Statistical models of ORCL returns - built to be *honest* about what they can and cannot say.

Three layers, deliberately separated:

A. DAILY FACTOR ATTRIBUTION (contemporaneous, thousands of observations)
   ORCL excess return ~ market + tech-tilt + semiconductor/AI-compute + hyperscaler factors.
   Answers "what is ORCL's return exposed to, and has that exposure changed?" - an explanation, not a forecast.
   Includes OLS with HAC errors, VIF / condition number, a rolling 252-day regression and a train/test split.

B. EARNINGS-WINDOW MODEL (contemporaneous, tens of observations)
   ORCL's excess return between consecutive earnings reactions ~ market factors + fundamental news (surprise, growth,
   RPO, OCI, FCF) + start-of-window state (momentum, volatility, valuation). Nested specifications show how in-sample R2
   rises with every added variable while leave-one-out R2 does not - the signature of overfitting at small N.

C. WALK-FORWARD PREDICTION (out-of-sample, point-in-time)
   Forward ~3-month excess return from PIT features with ridge regression (random forest for importance only).
   Re-fitted yearly on an expanding window with a purge so no training label overlaps the test period.
   Scored by out-of-sample R2 against the expanding historical mean, information coefficient and a Diebold-Mariano test.

Not claimed: causality. Correlation, multicollinearity, overfitting, survivorship bias (ORCL is a survivor that became an
AI winner and then fell sharply - a sample selected on today's relevance), look-ahead bias, data leakage and small
sample size are each addressed in ``LIMITATIONS`` and tested where possible.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..analytics import returns as R
from ..config import Config

LIMITATIONS = [
    "Correlation is not causation: a factor loading describes co-movement in this sample, not a mechanism.",
    "Multicollinearity: growth metrics (RPO, OCI, cloud, revenue) move together, so individual coefficients are unstable even when "
    "joint explanatory power is stable. VIFs and the condition number are reported.",
    "Overfitting: with ~60 earnings windows (28 with RPO, 17 with OCI) each added variable can raise in-sample R2 while lowering leave-one-out R2.",
    "Survivorship / selection bias: ORCL is analysed because it is currently relevant to the AI trade. Companies that failed to benefit "
    "are not in the sample, so average 'AI exposure' effects are likely overstated.",
    "Look-ahead bias: fundamentals enter only from their release session; percentile features use expanding windows; models are re-fitted "
    "on past data only. Any mistake here would inflate results, so it is unit-tested (truncation invariance).",
    "Data leakage: overlapping forward-return labels are purged between train and test windows; imputers/scalers are fitted on training data only.",
    "Small sample and regime change: one AI-capex cycle dominates the recent data; relationships estimated before 2023 may not hold after.",
    "Consensus-revision data are not available historically (free sources); the last earnings surprise is used as a point-in-time proxy.",
]

DAILY_FACTORS = ["MKT", "TECH", "SEMI", "CLOUD"]


# ============================================================================ A. daily factor model
def daily_factors(px: pd.DataFrame, rf_annual: pd.Series, cfg: Config) -> pd.DataFrame:
    spy, qqq, smh = cfg.data.benchmarks.sp500, cfg.data.benchmarks.nasdaq100, cfg.data.benchmarks.semis
    r = R.simple_returns(px)
    rfd = R.daily_rf(rf_annual).reindex(r.index).ffill()
    hyper = r[["MSFT", "AMZN", "GOOGL"]].mean(axis=1)
    f = pd.DataFrame({
        "MKT": r[spy] - rfd,           # market excess return
        "TECH": r[qqq] - r[spy],       # growth / mega-cap tech tilt
        "SEMI": r[smh] - r[qqq],       # AI-compute (semiconductor) vs the broader Nasdaq 100
        "CLOUD": hyper - r[qqq],       # hyperscaler peers vs the Nasdaq 100
    })
    f["y"] = r[cfg.project.ticker] - rfd
    return f.dropna()


def vif_table(X: pd.DataFrame) -> pd.Series:
    Xc = sm.add_constant(X)
    return pd.Series({c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"})


@dataclass
class OLSSummary:
    coefs: pd.DataFrame
    r2: float
    adj_r2: float
    n: int
    vif: pd.Series
    condition_number: float
    durbin_watson: float
    breusch_pagan_p: float


def ols_summary(y: pd.Series, X: pd.DataFrame, cov_type: str = "HAC", maxlags: int = 5) -> OLSSummary:
    Xc = sm.add_constant(X)
    kw = {"cov_kwds": {"maxlags": maxlags}} if cov_type == "HAC" else {}
    res = sm.OLS(y, Xc).fit(cov_type=cov_type, **kw)
    plain = sm.OLS(y, Xc).fit()
    bp = sm.stats.diagnostic.het_breuschpagan(plain.resid, Xc)[1]
    tab = pd.DataFrame({"coef": res.params, "std_err": res.bse, "t": res.tvalues, "p_value": res.pvalues})
    return OLSSummary(tab, float(res.rsquared), float(res.rsquared_adj), int(res.nobs), vif_table(X) if X.shape[1] > 1 else pd.Series(dtype=float),
                      float(np.linalg.cond(Xc.values)), float(sm.stats.stattools.durbin_watson(plain.resid)), float(bp))


@dataclass
class DailyFactorResult:
    full: OLSSummary
    capm_only: OLSSummary
    rolling: pd.DataFrame
    train: OLSSummary
    oos_r2: float
    oos_r2_capm: float
    factor_corr: pd.DataFrame
    by_year: pd.DataFrame


def daily_factor_model(px: pd.DataFrame, rf_annual: pd.Series, cfg: Config) -> DailyFactorResult:
    f = daily_factors(px, rf_annual, cfg)
    X, y = f[DAILY_FACTORS], f["y"]
    full = ols_summary(y, X)
    capm = ols_summary(y, X[["MKT"]])
    w = cfg.analytics.rolling_beta_window
    roll = RollingOLS(y, sm.add_constant(X), window=w).fit()
    rolling = roll.params.rename(columns={"const": "alpha"})
    rolling["r2"] = roll.rsquared
    # train / test split
    cut = pd.Timestamp(cfg.backtest.train_end)
    tr, te = f[f.index <= cut], f[f.index > cut]
    train = ols_summary(tr["y"], tr[DAILY_FACTORS])
    b = train.coefs["coef"]
    pred = b["const"] + te[DAILY_FACTORS] @ b[DAILY_FACTORS]
    sse, sst = ((te["y"] - pred) ** 2).sum(), ((te["y"] - tr["y"].mean()) ** 2).sum()
    cb = ols_summary(tr["y"], tr[["MKT"]]).coefs["coef"]
    pred_c = cb["const"] + te["MKT"] * cb["MKT"]
    sse_c = ((te["y"] - pred_c) ** 2).sum()
    by_year = f.groupby(f.index.year).apply(lambda g: pd.Series(
        sm.OLS(g["y"], sm.add_constant(g[DAILY_FACTORS])).fit().params.rename({"const": "alpha"}).to_dict()
        | {"r2": sm.OLS(g["y"], sm.add_constant(g[DAILY_FACTORS])).fit().rsquared, "n": len(g)}))
    return DailyFactorResult(full, capm, rolling, train, float(1 - sse / sst), float(1 - sse_c / sst), f[DAILY_FACTORS].corr(), by_year)


# ============================================================================ B. earnings-window model
def window_dataset(px: pd.DataFrame, q: pd.DataFrame, ev: pd.DataFrame, panel: pd.DataFrame, rf_annual: pd.Series, cfg: Config) -> pd.DataFrame:
    """One row per earnings window: (previous reaction date, this reaction date]. Everything about quarter q becomes
    known exactly at this window's end, so the fundamental variables are *contemporaneous news*, not predictors."""
    f = daily_factors(px, rf_annual, cfg)
    react = pd.to_datetime(ev["reaction_date"]).dropna().sort_values()
    react = react[react >= f.index[0]]
    rows = []
    dates = list(react.items())
    for (pe_prev, d_prev), (pe_cur, d_cur) in zip(dates[:-1], dates[1:]):
        win = f[(f.index > d_prev) & (f.index <= d_cur)]
        if len(win) < 30 or pe_cur not in q.index:
            continue
        cum = lambda s: float(np.expm1(np.log1p(s).sum()))
        row = {"period_end": pe_cur, "start": d_prev, "end": d_cur, "n_days": len(win), "y": cum(win["y"])}
        for c in DAILY_FACTORS:
            row[c] = cum(win[c]) if c != "MKT" else cum(win["MKT"])
        news = q.loc[pe_cur]
        prev_q = q.index[q.index.get_loc(pe_cur) - 1] if q.index.get_loc(pe_cur) > 0 else None
        row["eps_surprise_pct"] = news.get("eps_surprise_pct")
        row["revenue_yoy"] = news["revenue_yoy"]
        row["revenue_accel"] = news["revenue_yoy"] - (q.loc[prev_q, "revenue_yoy"] if prev_q is not None else np.nan)
        row["eps_ttm_yoy"] = news["eps_ttm_yoy"]
        row["fcf_margin_chg"] = news["fcf_margin_yoy_change"]
        row["rpo_yoy"] = news["rpo_yoy"]
        row["oci_growth"] = news["oci_growth"]
        st = panel.loc[:d_prev].iloc[-1] if len(panel.loc[:d_prev]) else None
        if st is not None:
            row["mom_12_1_start"], row["vol_63d_start"], row["pe_rank_start"] = st["mom_12_1"], st["vol_63d"], st["pe_trailing__rank"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("period_end")


SPECS = {
    "S1: market factors only": ["MKT", "TECH", "SEMI"],
    "S2: + earnings news": ["MKT", "TECH", "SEMI", "eps_surprise_pct", "revenue_accel", "eps_ttm_yoy"],
    "S3: + start-of-window state": ["MKT", "TECH", "SEMI", "eps_surprise_pct", "revenue_accel", "eps_ttm_yoy", "mom_12_1_start", "vol_63d_start", "pe_rank_start"],
    "S4: + RPO and FCF trend": ["MKT", "TECH", "SEMI", "eps_surprise_pct", "revenue_accel", "eps_ttm_yoy", "mom_12_1_start", "vol_63d_start", "pe_rank_start", "rpo_yoy", "fcf_margin_chg"],
    "S5: + OCI growth": ["MKT", "TECH", "SEMI", "eps_surprise_pct", "revenue_accel", "eps_ttm_yoy", "mom_12_1_start", "vol_63d_start", "pe_rank_start", "rpo_yoy", "fcf_margin_chg", "oci_growth"],
}


def loo_r2(y: pd.Series, X: pd.DataFrame) -> float:
    """Leave-one-out cross-validated R2 via the hat-matrix shortcut (exact for OLS)."""
    Xc = sm.add_constant(X).values
    H = Xc @ np.linalg.pinv(Xc.T @ Xc) @ Xc.T
    resid = y.values - H @ y.values
    loo = resid / (1 - np.diag(H))
    return float(1 - (loo ** 2).sum() / ((y - y.mean()) ** 2).sum())


def window_models(ds: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, OLSSummary]]:
    rows, fits = [], {}
    for name, cols in SPECS.items():
        d = ds[["y"] + cols].dropna()
        if len(d) < len(cols) + 8:
            rows.append({"specification": name, "n": len(d), "k": len(cols), "r2": np.nan, "adj_r2": np.nan, "loo_r2": np.nan})
            continue
        s = ols_summary(d["y"], d[cols], cov_type="HC3")
        fits[name] = s
        rows.append({"specification": name, "n": s.n, "k": len(cols), "r2": s.r2, "adj_r2": s.adj_r2, "loo_r2": loo_r2(d["y"], d[cols]),
                     "max_vif": float(s.vif.max()) if len(s.vif) else np.nan, "condition_number": s.condition_number})
    return pd.DataFrame(rows).set_index("specification"), fits


# ============================================================================ correlation analysis
PRED_FEATURES = ["mom_12_1", "vol_63d", "beta_252d", "drawdown_52w", "pe_trailing__rank", "ev_ebitda__rank", "revenue_yoy", "eps_ttm_yoy",
                 "fcf_margin_ttm", "fcf_margin_yoy_change", "rpo_yoy", "oci_growth", "eps_surprise_pct", "spy_ret_3m", "qqq_ret_3m", "smh_ret_3m"]


def correlation_analysis(panel: pd.DataFrame, horizon: int = 63) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation of each feature with the forward excess return (monthly samples) + feature-feature matrix."""
    fwd = panel["close"].shift(-horizon) / panel["close"] - 1.0 - (panel["rf_3m"] * horizon / 252)
    d = panel[PRED_FEATURES].assign(fwd=fwd).iloc[::21].dropna(subset=["fwd"])
    rows = []
    for c in PRED_FEATURES:
        x = d[[c, "fwd"]].dropna()
        if len(x) < 24:
            continue
        rho, p = stats.spearmanr(x[c], x["fwd"])
        neff = len(x) / max(1, horizon // 21)             # 63d labels sampled every 21d overlap ~3x
        z = np.arctanh(np.clip(rho, -0.999, 0.999)) * np.sqrt(max(neff - 3, 1))
        rows.append({"feature": c, "spearman_rho": rho, "n_obs": len(x), "n_effective": neff, "p_value_adj": 2 * (1 - stats.norm.cdf(abs(z)))})
    return pd.DataFrame(rows).set_index("feature").sort_values("spearman_rho", key=abs, ascending=False), d[PRED_FEATURES].corr(method="spearman")


# ============================================================================ C. walk-forward prediction
@dataclass
class WalkForward:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    importance: pd.DataFrame
    ridge_coefs: pd.Series
    folds: pd.DataFrame
    dm_pvalue: float


def _clip3(X):
    """Winsorise standardised features at +/-3 sd: regime-shifted values (e.g. AI-era OCI growth) must not be extrapolated linearly."""
    return np.clip(X, -3.0, 3.0)


def _make_ridge(alphas, n_splits=4):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), FunctionTransformer(_clip3),
                         RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=n_splits)))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 3) -> tuple[float, float]:
    """H0: equal squared-error accuracy; HAC variance with h-1 lags (overlapping h-step labels). Negative stat favours model 1."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dm = d.mean()
    gamma = [np.mean((d[k:] - dm) * (d[: n - k] - dm)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dm / np.sqrt(max(var, 1e-18))
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def walk_forward(panel: pd.DataFrame, cfg: Config, horizon: int | None = None, features: list[str] | None = None, seed: int = 11) -> WalkForward:
    horizon = horizon or cfg.factor_model.forward_horizon_days
    features = [f for f in (features or PRED_FEATURES) if f in panel.columns]
    fwd = panel["close"].shift(-horizon) / panel["close"] - 1.0 - (panel["rf_3m"] * horizon / 252)
    d = panel[features].assign(y=fwd).iloc[::21]                           # monthly sampling
    d = d.dropna(subset=["y"])
    # features with no history yet remain NaN and are median-imputed (train median) - flagged by the imputer's missingness
    first = d.index[d[features].notna().sum(axis=1) >= 6][0]
    d = d[d.index >= first]
    years = sorted({t.year for t in d.index})
    test_years = [y for y in years if y >= first.year + 4]                 # >= ~4 years of training before the first test year
    preds, folds, imps = [], [], []
    for y in test_years:
        test = d[d.index.year == y]
        train = d[d.index < test.index[0] - pd.Timedelta(days=int(horizon * 1.5))]    # purge: no train label overlaps the test window
        if len(train) < cfg.factor_model.min_train_obs * 3 or test.empty:
            continue
        mdl = _make_ridge(list(cfg.factor_model.ridge_alphas))
        mdl.fit(train[features], train["y"])
        p = pd.Series(mdl.predict(test[features]), index=test.index)
        bench = pd.Series(train["y"].mean(), index=test.index)             # expanding historical-mean forecast
        preds.append(pd.DataFrame({"y": test["y"], "ridge": p, "hist_mean": bench}))
        folds.append({"test_year": y, "n_train": len(train), "n_test": len(test), "alpha": float(mdl[-1].alpha_)})
        # permutation importance on this test fold
        base_mse = ((test["y"] - p) ** 2).mean()
        rng = np.random.default_rng(seed + y)
        imp = {}
        for f in features:
            drops = []
            for _ in range(10):
                Xp = test[features].copy()
                Xp[f] = rng.permutation(Xp[f].values)
                drops.append(((test["y"] - mdl.predict(Xp)) ** 2).mean() - base_mse)
            imp[f] = np.mean(drops)
        imps.append(pd.Series(imp, name=y))
    P = pd.concat(preds)
    e_m, e_b = (P["y"] - P["ridge"]).values, (P["y"] - P["hist_mean"]).values
    r2_oos = 1 - (e_m ** 2).sum() / (e_b ** 2).sum()
    ic = stats.spearmanr(P["ridge"], P["y"])[0]
    dm_stat, dm_p = diebold_mariano(e_m, e_b, h=max(1, horizon // 21))
    hit = float(((P["ridge"] > P["ridge"].median()) == (P["y"] > P["y"].median())).mean())
    # final fit for coefficients / RF importance
    final = _make_ridge(list(cfg.factor_model.ridge_alphas)).fit(d[features], d["y"])
    coefs = pd.Series(final[-1].coef_, index=features).sort_values(key=abs, ascending=False)
    rf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                       RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=8, random_state=seed, n_jobs=-1)).fit(d[features], d["y"])
    imp_df = pd.DataFrame({"permutation_importance_oos": pd.concat(imps, axis=1).mean(axis=1), "ridge_std_coef": coefs.reindex(features),
                           "random_forest_importance": pd.Series(rf[-1].feature_importances_, index=features)}).sort_values("permutation_importance_oos", ascending=False)
    metrics = {"oos_r2_vs_hist_mean": float(r2_oos), "information_coefficient": float(ic), "directional_hit_rate": hit,
               "dm_statistic": dm_stat, "n_oos": int(len(P)), "first_test": str(P.index[0].date()), "last_test": str(P.index[-1].date()),
               "horizon_days": horizon}
    return WalkForward(P, metrics, imp_df, coefs, pd.DataFrame(folds), dm_p)


# ============================================================================ orchestration
@dataclass
class FactorModelResults:
    daily: DailyFactorResult
    window_dataset: pd.DataFrame
    window_specs: pd.DataFrame
    window_fits: dict[str, OLSSummary]
    corr_target: pd.DataFrame
    corr_features: pd.DataFrame
    walk_forward: WalkForward
    findings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=lambda: list(LIMITATIONS))


def run_factor_models(st, cfg: Config, panel: pd.DataFrame) -> FactorModelResults:
    px, q, ev, rf = st.prices(), st.fundamentals(), st.events(), st.risk_free()["rf_3m"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily = daily_factor_model(px, rf, cfg)
        ds = window_dataset(px, q, ev, panel, rf, cfg)
        specs, fits = window_models(ds)
        corr_t, corr_f = correlation_analysis(panel)
        wf = walk_forward(panel, cfg)
    res = FactorModelResults(daily, ds, specs, fits, corr_t, corr_f, wf)
    res.findings = _findings(res, cfg)
    return res


def _findings(r: FactorModelResults, cfg: Config) -> list[str]:
    f = []
    c = r.daily.full.coefs["coef"]
    t = r.daily.full.coefs["t"]
    f.append(f"Daily factor attribution (n={r.daily.full.n:,}): R2 = {r.daily.full.r2:.0%} with four factors vs {r.daily.capm_only.r2:.0%} market-only. "
             f"Loadings: market {c['MKT']:.2f} (t={t['MKT']:.1f}), tech tilt {c['TECH']:.2f}, semis/AI {c['SEMI']:.2f} (t={t['SEMI']:.1f}), hyperscalers {c['CLOUD']:.2f}. "
             f"About {1 - r.daily.full.r2:.0%} of daily variance is stock-specific (earnings gaps, backlog and financing news).")
    roll = r.daily.rolling.dropna()
    if len(roll):
        f.append(f"Rolling 252-day multi-factor loadings are unstable: the market loading ranged {roll['MKT'].min():.2f} to {roll['MKT'].max():.2f} (latest {roll['MKT'].iloc[-1]:.2f}), "
                 f"the tech-tilt loading {roll['TECH'].min():.2f} to {roll['TECH'].max():.2f} (latest {roll['TECH'].iloc[-1]:.2f}) and the semis/AI loading {roll['SEMI'].min():.2f} to {roll['SEMI'].max():.2f} "
                 f"(latest {roll['SEMI'].iloc[-1]:.2f}). Full-sample coefficients therefore describe an average that no single year resembles.")
    f.append(f"Out-of-sample (fit through {cfg.backtest.train_end}): four-factor R2 = {r.daily.oos_r2:.0%} vs market-only {r.daily.oos_r2_capm:.0%}.")
    s = r.window_specs.dropna(subset=["r2"])
    if len(s):
        last = s.iloc[-1]
        f.append(f"Earnings-window models: in-sample R2 rises from {s['r2'].iloc[0]:.0%} (market factors) to {last['r2']:.0%} ({last.name}), "
                 f"but leave-one-out R2 goes {s['loo_r2'].iloc[0]:.0%} -> {last['loo_r2']:.0%}: added variables are largely fitting noise at n={int(last['n'])}.")
    wf = r.walk_forward.metrics
    f.append(f"Walk-forward prediction of the next {wf['horizon_days']}-day excess return ({wf['first_test']} to {wf['last_test']}, {wf['n_oos']} monthly samples): "
             f"OOS R2 vs the historical mean = {wf['oos_r2_vs_hist_mean']:.1%}, information coefficient = {wf['information_coefficient']:.2f}, "
             f"Diebold-Mariano p = {r.walk_forward.dm_pvalue:.2f}. "
             + ("The model does NOT beat a constant forecast out of sample." if wf["oos_r2_vs_hist_mean"] <= 0 or r.walk_forward.dm_pvalue > 0.10
                else "The model beats a constant forecast out of sample, but with a wide uncertainty band (small effective sample)."))
    top = r.walk_forward.importance.head(3).index.tolist()
    f.append(f"Most influential features out of sample (permutation importance): {', '.join(top)} - an importance ranking is not evidence of a causal driver.")
    return f
