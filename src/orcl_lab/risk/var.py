"""Value-at-Risk and Expected Shortfall.

All numbers are reported as **positive loss fractions** (0.05 = a 5% loss).

Methods
-------
* Historical simulation  - empirical quantile of realised returns (h-day returns are compounded from history,
  not scaled by sqrt(h), so they inherit real volatility clustering and jumps).
* Parametric normal      - mu*h - z * sigma * sqrt(h).      Light tails by construction.
* Parametric Student-t   - fitted degrees of freedom (MLE).  Heavier tails.
* Cornish-Fisher         - normal quantile adjusted for sample skewness and excess kurtosis.

``var_backtest`` checks a VaR model out-of-sample with Kupiec's proportion-of-failures test and
Christoffersen's independence test: a good VaR is breached about (1 - level) of the time AND the breaches
are not clustered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def horizon_returns(r: pd.Series, h: int) -> pd.Series:
    """Overlapping h-day compounded returns."""
    if h == 1:
        return r.dropna()
    return ((1.0 + r.dropna()).rolling(h).apply(np.prod, raw=True) - 1.0).dropna()


# ---------------------------------------------------------------------------- VaR / ES
def historical_var(r: pd.Series, level: float = 0.95, h: int = 1) -> float:
    rh = horizon_returns(r, h)
    return float(-np.quantile(rh, 1.0 - level))


def historical_es(r: pd.Series, level: float = 0.95, h: int = 1) -> float:
    rh = horizon_returns(r, h)
    cutoff = np.quantile(rh, 1.0 - level)
    tail = rh[rh <= cutoff]
    return float(-tail.mean())


def _cf_valid(skew: float, ex_kurt: float) -> bool:
    """The Cornish-Fisher map z -> z_cf(z) must be increasing over the tail (z in [-4, 0]); with very fat tails
    (e.g. ORCL's excess kurtosis > 30) it is not, and the 'quantile' it returns is meaningless."""
    z = np.linspace(-4.0, 0.0, 400)
    g = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * ex_kurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
    return bool(np.all(np.diff(g) > 0))


def parametric_var(r: pd.Series, level: float = 0.95, h: int = 1, dist: str = "normal") -> float:
    r = r.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    alpha = 1.0 - level
    if dist == "normal":
        q = stats.norm.ppf(alpha)
        return float(-(mu * h + q * sd * np.sqrt(h)))
    if dist == "t":
        nu, loc, scale = stats.t.fit(r)
        nu = max(nu, 2.05)
        sd_t = scale * np.sqrt(nu / (nu - 2))             # std of the fitted t
        q = stats.t.ppf(alpha, nu) * scale / sd_t         # standardised t quantile
        return float(-(mu * h + q * sd * np.sqrt(h)))
    if dist == "cornish-fisher":
        z = stats.norm.ppf(alpha)
        s, k = stats.skew(r), stats.kurtosis(r)           # excess kurtosis
        if not _cf_valid(s, k):
            return float("nan")                           # expansion is non-monotonic -> quantile is meaningless
        zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return float(-(mu * h + zcf * sd * np.sqrt(h)))
    raise ValueError(dist)


def parametric_es(r: pd.Series, level: float = 0.95, h: int = 1, dist: str = "normal") -> float:
    r = r.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    alpha = 1.0 - level
    if dist == "normal":
        z = stats.norm.ppf(alpha)
        return float(-(mu * h) + sd * np.sqrt(h) * stats.norm.pdf(z) / alpha)
    if dist == "t":
        nu, loc, scale = stats.t.fit(r)
        nu = max(nu, 2.05)
        sd_t = scale * np.sqrt(nu / (nu - 2))
        tq = stats.t.ppf(alpha, nu)
        es_std = stats.t.pdf(tq, nu) * (nu + tq**2) / ((nu - 1) * alpha) * scale / sd_t
        return float(-(mu * h) + sd * np.sqrt(h) * es_std)
    raise ValueError(dist)


def var_table(r: pd.Series, levels=(0.95, 0.99), horizons=(1, 10)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        for lv in levels:
            rows.append({
                "horizon_days": h, "confidence": lv,
                "Historical VaR": historical_var(r, lv, h),
                "Parametric VaR (normal)": parametric_var(r, lv, h, "normal"),
                "Parametric VaR (Student-t)": parametric_var(r, lv, h, "t"),
                "Cornish-Fisher VaR": parametric_var(r, lv, h, "cornish-fisher"),
                "Historical ES": historical_es(r, lv, h),
                "Parametric ES (normal)": parametric_es(r, lv, h, "normal"),
                "Parametric ES (Student-t)": parametric_es(r, lv, h, "t"),
            })
    return pd.DataFrame(rows)


def rolling_var_es(r: pd.Series, window: int = 252, level: float = 0.95) -> pd.DataFrame:
    """1-day historical VaR/ES estimated from the trailing window (uses data up to and including t)."""
    q = r.rolling(window, min_periods=window).quantile(1.0 - level)
    es = r.rolling(window, min_periods=window).apply(lambda x: x[x <= np.quantile(x, 1.0 - level)].mean(), raw=True)
    return pd.DataFrame({"var": -q, "es": -es})


# ---------------------------------------------------------------------------- tail diagnostics
def tail_diagnostics(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    z = (r - r.mean()) / r.std(ddof=1)
    out = {"skew": float(stats.skew(r)), "excess_kurtosis": float(stats.kurtosis(r))}
    jb = stats.jarque_bera(r)
    out["jarque_bera_p"] = float(jb.pvalue)
    for k in (3, 4, 5):
        out[f"observed_days_below_-{k}sd"] = int((z < -k).sum())
        out[f"normal_expected_days_below_-{k}sd"] = float(stats.norm.cdf(-k) * len(r))
    out["n_obs"] = int(len(r))
    return out


# ---------------------------------------------------------------------------- VaR backtests
@dataclass
class VaRBacktest:
    level: float
    n: int
    exceptions: int
    expected: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    verdict: str


def kupiec_pof(exceptions: int, n: int, p: float) -> tuple[float, float]:
    x = exceptions
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        pi = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    h = hits.astype(int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(h[:-1], h[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    if n01 + n11 == 0 or (n00 + n01) == 0:
        return 0.0, 1.0
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p, a, b):  # a successes, b failures
        return (a * np.log(p) if a else 0.0) + (b * np.log(1 - p) if b else 0.0)

    l0 = ll(pi, n01 + n11, n00 + n10)
    l1 = ll(pi01, n01, n00) + ll(pi11, n11, n10)
    lr = -2 * (l0 - l1)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def var_backtest(r: pd.Series, level: float = 0.95, window: int = 252, method: str = "historical") -> tuple[VaRBacktest, pd.DataFrame]:
    """Out-of-sample check: VaR_t is estimated from the `window` returns ending at t-1 and compared to r_t."""
    r = r.dropna()
    if method == "historical":
        var = (-r.rolling(window).quantile(1 - level)).shift(1)
    elif method == "normal":
        z = stats.norm.ppf(1 - level)
        var = (-(r.rolling(window).mean() + z * r.rolling(window).std())).shift(1)
    else:
        raise ValueError(method)
    df = pd.DataFrame({"ret": r, "var": var}).dropna()
    df["hit"] = df["ret"] < -df["var"]
    n, x = len(df), int(df["hit"].sum())
    p = 1 - level
    lr_k, p_k = kupiec_pof(x, n, p)
    lr_c, p_c = christoffersen_independence(df["hit"].to_numpy())
    verdict = "accept" if (p_k > 0.05 and p_c > 0.05) else "reject (frequency)" if p_k <= 0.05 and p_c > 0.05 else \
        "reject (clustering)" if p_k > 0.05 else "reject (both)"
    return VaRBacktest(level, n, x, n * p, lr_k, p_k, lr_c, p_c, verdict), df
