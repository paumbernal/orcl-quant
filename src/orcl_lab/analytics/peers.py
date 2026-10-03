"""Cross-sectional comparison of ORCL against benchmarks and AI / cloud peers.

Questions answered
------------------
* How correlated is ORCL with the market, the Nasdaq 100 and the AI-compute complex - and has that changed?
* Who out/under-performed, with what volatility and what drawdowns?
* When did ORCL *diverge* from its peers, and for how long?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import returns as R


def rebase(px: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    px = px.dropna()
    return px / px.iloc[0] * base


def correlation_matrix(rets: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return rets.dropna().corr(method=method)


def rolling_correlation(rets: pd.DataFrame, target: str, window: int = 126) -> pd.DataFrame:
    others = [c for c in rets.columns if c != target]
    return pd.DataFrame({c: rets[target].rolling(window, min_periods=window).corr(rets[c]) for c in others})


def peer_table(prices: pd.DataFrame, rf_annual: pd.Series, market: str, td: int = 252) -> pd.DataFrame:
    """CAGR / vol / Sharpe / drawdown / beta for every column over their common history."""
    px = prices.dropna()
    rets = R.simple_returns(px)
    rf = R.daily_rf(rf_annual, td).reindex(rets.index).ffill()
    tab = R.performance_summary(rets, rets[market], rf, td)
    tab["Total return"] = (px.iloc[-1] / px.iloc[0] - 1.0).reindex(tab.index)
    dd_now = {c: R.drawdown_series(rets[c]).iloc[-1] for c in rets.columns}
    tab["Current drawdown"] = pd.Series(dd_now)
    return tab


def period_returns(prices: pd.DataFrame, periods: dict[str, int] | None = None) -> pd.DataFrame:
    """Trailing total returns over standard windows (trading days)."""
    periods = periods or {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}
    out = {}
    for name, n in periods.items():
        if len(prices) > n:
            out[name] = prices.iloc[-1] / prices.iloc[-n - 1] - 1.0
    return pd.DataFrame(out)


def calendar_year_returns(prices: pd.DataFrame) -> pd.DataFrame:
    yr = prices.resample("YE").last()
    first = prices.iloc[[0]].rename(index=lambda _: yr.index[0] - pd.offsets.YearEnd(1))
    ret = pd.concat([first, yr]).pct_change().dropna(how="all")
    ret.index = ret.index.year
    return ret


# --------------------------------------------------------------------------- divergence analysis
@dataclass
class DivergenceResult:
    relative_return: pd.Series      # rolling ORCL minus basket return (log)
    zscore: pd.Series
    episodes: pd.DataFrame


def divergence(
    prices: pd.DataFrame,
    target: str,
    peers: list[str],
    window: int = 63,
    z_threshold: float = 1.5,
    min_history: int = 504,
) -> DivergenceResult:
    """Episodes where ORCL's rolling relative return vs an equal-weight peer basket is extreme.

    relative_t = sum_{k<window} (log r_ORCL - mean_peers log r)
    z_t        = (relative_t - mean_{<=t}) / std_{<=t}     (expanding statistics: uses no future data)
    An *episode* is a maximal run of days with |z| >= threshold.
    """
    lr = np.log(prices[[target] + peers].dropna()).diff().dropna()
    basket = lr[peers].mean(axis=1)
    rel = (lr[target] - basket).rolling(window, min_periods=window).sum()
    mu = rel.expanding(min_periods=min_history).mean()
    sd = rel.expanding(min_periods=min_history).std()
    z = (rel - mu) / sd
    flag = z.abs() >= z_threshold
    grp = (flag != flag.shift()).cumsum()
    rows = []
    for _, seg in z[flag].groupby(grp[flag]):
        peak = seg.abs().idxmax()
        rows.append({
            "start": seg.index[0],
            "end": seg.index[-1],
            "trading_days": len(seg),
            "direction": "ORCL outperforming peers" if seg.mean() > 0 else "ORCL underperforming peers",
            "peak_zscore": float(seg.loc[peak]),
            "peak_relative_return": float(np.expm1(rel.loc[peak])),
        })
    ep = pd.DataFrame(rows)
    if not ep.empty:
        ep = ep[ep["trading_days"] >= 5].reset_index(drop=True)
    return DivergenceResult(np.expm1(rel), z, ep)


def drawdown_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in prices.columns:
        s = R.drawdown_stats(R.simple_returns(prices[c].dropna()))
        rows[c] = {
            "Max drawdown": s.max_drawdown,
            "Peak": s.peak_date,
            "Trough": s.trough_date,
            "Recovered": s.recovery_date,
            "Current drawdown": s.current_drawdown,
            "Days underwater (current)": s.days_underwater_current,
        }
    return pd.DataFrame(rows).T
