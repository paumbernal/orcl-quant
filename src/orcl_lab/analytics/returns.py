"""Return, volatility, drawdown and risk-adjusted performance statistics.

Conventions
-----------
* Returns are *total returns* computed from dividend-adjusted closes.
* Annualisation uses ``td`` = 252 trading days. Volatility scales with sqrt(td); means scale with td.
* Sharpe / Sortino use the 3-month T-bill (FRED DGS3MO) converted to a daily rate as the risk-free rate.
* Beta / alpha come from an OLS regression of *excess* returns with Newey-West (HAC) standard errors,
  so reported t-statistics are robust to autocorrelation and heteroskedasticity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

TD = 252


# ---------------------------------------------------------------- basic series
def simple_returns(px: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return px.pct_change().dropna(how="all")


def log_returns(px: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(px).diff().dropna(how="all")


def cumulative_returns(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Cumulative simple return path starting at 0 (i.e. wealth - 1)."""
    return (1.0 + r.fillna(0.0)).cumprod() - 1.0


def daily_rf(rf_annual: pd.Series, td: int = TD) -> pd.Series:
    """Daily compounded risk-free return from an annualised decimal rate."""
    return (1.0 + rf_annual) ** (1.0 / td) - 1.0


def rolling_vol(r: pd.Series | pd.DataFrame, window: int, td: int = TD):
    return r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(td)


def annualised_vol(r: pd.Series, td: int = TD) -> float:
    return float(r.std(ddof=1) * np.sqrt(td))


def cagr(r: pd.Series, td: int = TD) -> float:
    r = r.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    return growth ** (td / len(r)) - 1.0 if growth > 0 else -1.0


# ---------------------------------------------------------------- drawdowns
def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


@dataclass
class DrawdownStats:
    max_drawdown: float
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: pd.Timestamp | None
    current_drawdown: float
    days_to_trough: int
    days_underwater_current: int


def drawdown_stats(r: pd.Series) -> DrawdownStats:
    dd = drawdown_series(r)
    trough = dd.idxmin()
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    rec = after[after >= wealth.loc[peak]]
    cur_peak_idx = wealth.idxmax()
    return DrawdownStats(
        max_drawdown=float(dd.min()),
        peak_date=peak,
        trough_date=trough,
        recovery_date=rec.index[0] if len(rec) else None,
        current_drawdown=float(dd.iloc[-1]),
        days_to_trough=int((trough - peak).days),
        days_underwater_current=int((dd.index[-1] - wealth[:dd.index[-1]].idxmax()).days),
    )


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown_series(r).min())


# ---------------------------------------------------------------- risk-adjusted ratios
def downside_deviation(r: pd.Series, target: float | pd.Series = 0.0, td: int = TD) -> float:
    """Annualised downside deviation: sqrt(mean(min(r - target, 0)^2)) * sqrt(td) over ALL observations."""
    d = np.minimum((r - target).to_numpy(), 0.0)
    return float(np.sqrt(np.mean(d ** 2)) * np.sqrt(td))


def sharpe_ratio(r: pd.Series, rf_daily: float | pd.Series = 0.0, td: int = TD) -> float:
    ex = (r - rf_daily).dropna()
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(td)) if sd > 0 else float("nan")


def sortino_ratio(r: pd.Series, rf_daily: float | pd.Series = 0.0, td: int = TD) -> float:
    ex = (r - rf_daily).dropna()
    dd = downside_deviation(ex, 0.0, td)
    return float(ex.mean() * td / dd) if dd > 0 else float("nan")


def calmar_ratio(r: pd.Series, td: int = TD) -> float:
    mdd = abs(max_drawdown(r))
    return cagr(r, td) / mdd if mdd > 0 else float("nan")


# ---------------------------------------------------------------- CAPM regression
@dataclass
class CAPMResult:
    beta: float
    alpha_annual: float
    alpha_daily: float
    beta_se: float
    alpha_tstat: float
    beta_tstat: float
    r_squared: float
    n: int
    correlation: float


def capm(r_asset: pd.Series, r_mkt: pd.Series, rf_daily: float | pd.Series = 0.0, td: int = TD, hac_lags: int = 5) -> CAPMResult:
    df = pd.concat([r_asset, r_mkt], axis=1, keys=["a", "m"]).dropna()
    rf = rf_daily.reindex(df.index).ffill() if isinstance(rf_daily, pd.Series) else rf_daily
    y, x = df["a"] - rf, df["m"] - rf
    X = sm.add_constant(x.rename("mkt"))
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return CAPMResult(
        beta=float(res.params["mkt"]),
        alpha_annual=float((1 + res.params["const"]) ** td - 1),
        alpha_daily=float(res.params["const"]),
        beta_se=float(res.bse["mkt"]),
        alpha_tstat=float(res.tvalues["const"]),
        beta_tstat=float(res.tvalues["mkt"]),
        r_squared=float(res.rsquared),
        n=int(res.nobs),
        correlation=float(df["a"].corr(df["m"])),
    )


def rolling_beta(r_asset: pd.Series, r_mkt: pd.Series, window: int = 252) -> pd.Series:
    df = pd.concat([r_asset, r_mkt], axis=1, keys=["a", "m"]).dropna()
    cov = df["a"].rolling(window, min_periods=window).cov(df["m"])
    var = df["m"].rolling(window, min_periods=window).var()
    return (cov / var).rename("rolling_beta")


def rolling_alpha(r_asset: pd.Series, r_mkt: pd.Series, window: int = 252, td: int = TD) -> pd.Series:
    """Rolling CAPM-style alpha (annualised intercept) from rolling means and beta, no rf adjustment."""
    b = rolling_beta(r_asset, r_mkt, window)
    ma = r_asset.rolling(window, min_periods=window).mean()
    mm = r_mkt.rolling(window, min_periods=window).mean()
    return ((ma - b * mm) * td).rename("rolling_alpha")


# ---------------------------------------------------------------- one-stop summary
def performance_summary(
    returns: pd.DataFrame,
    market: pd.Series,
    rf_daily: pd.Series,
    td: int = TD,
) -> pd.DataFrame:
    """Per-column performance table (CAGR, vol, Sharpe, Sortino, drawdown, beta, alpha, tail shape)."""
    rows = {}
    for c in returns.columns:
        r = returns[c].dropna()
        rf = rf_daily.reindex(r.index).ffill()
        cp = capm(r, market, rf_daily, td) if c != market.name else None
        rows[c] = {
            "CAGR": cagr(r, td),
            "Ann. volatility": annualised_vol(r, td),
            "Sharpe": sharpe_ratio(r, rf, td),
            "Sortino": sortino_ratio(r, rf, td),
            "Max drawdown": max_drawdown(r),
            "Calmar": calmar_ratio(r, td),
            "Beta (vs S&P 500)": cp.beta if cp else 1.0,
            "Alpha (ann.)": cp.alpha_annual if cp else 0.0,
            "Skew": float(r.skew()),
            "Excess kurtosis": float(r.kurt()),
            "Best day": float(r.max()),
            "Worst day": float(r.min()),
            "Start": r.index[0],
            "End": r.index[-1],
        }
    return pd.DataFrame(rows).T
