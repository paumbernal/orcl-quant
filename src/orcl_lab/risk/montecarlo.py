"""Monte Carlo price-path simulation with four return models.

Why more than one model?
------------------------
A normal (GBM) model is the textbook baseline but ORCL's daily returns are fat-tailed and its volatility
clusters (earnings gaps of 10-35% in a day, long calm spells in between). A model that ignores this will
*understate the probability and size of large losses*. We therefore simulate the same one-year horizon under:

    normal      i.i.d. Gaussian daily returns                       (baseline, thin tails)
    student_t   i.i.d. Student-t, degrees of freedom fitted by MLE  (fat tails)
    garch_t     GARCH(1,1) with Student-t innovations (arch pkg)    (fat tails + volatility clustering,
                                                                     starts from TODAY's conditional vol)
    bootstrap   moving-block bootstrap of demeaned history      (distribution-free; preserves clustering
                                                                     and the real jump sizes)

Drift is a modelling *choice*, not something to be estimated from a noisy, regime-dependent sample:
``capm`` (default) = risk-free + beta x equity-risk-premium; ``historical`` = sample mean; ``zero``.
All models are given the same drift so differences between them come from the *shape* of the distribution only.

Daily returns are modelled directly as simple returns r_t = mu_d + sigma_d * z_t (clipped at -99%), so
E[r_t] = mu_d exactly and paths are products of (1 + r_t).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ("normal", "student_t", "garch_t", "bootstrap")


@dataclass
class SimulationResult:
    model: str
    params: dict
    terminal: np.ndarray                  # terminal values, shape (n_sims,)
    max_drawdown: np.ndarray              # shape (n_sims,)
    paths_sample: np.ndarray              # (min(200,n), T+1) for fan-chart style plotting
    quantile_paths: pd.DataFrame          # percentiles of the value path over time
    daily_returns_sample: np.ndarray      # flat sample of simulated daily returns (tail comparison)

    def summary(self, start_value: float = 100.0) -> dict[str, float]:
        ret = self.terminal / start_value - 1.0
        q5 = np.quantile(ret, 0.05)
        return {
            "model": self.model,
            "expected_return": float(ret.mean()),
            "median_return": float(np.median(ret)),
            "p05_return": float(q5),
            "p95_return": float(np.quantile(ret, 0.95)),
            "prob_loss": float((ret < 0).mean()),
            "prob_loss_over_30pct": float((ret < -0.30).mean()),
            "prob_loss_over_50pct": float((ret < -0.50).mean()),
            "expected_shortfall_5pct": float(ret[ret <= q5].mean()),
            "p05_value": float(np.quantile(self.terminal, 0.05)),
            "median_value": float(np.median(self.terminal)),
            "p95_value": float(np.quantile(self.terminal, 0.95)),
            "mean_max_drawdown": float(self.max_drawdown.mean()),
            "median_max_drawdown": float(np.median(self.max_drawdown)),
            "worst5pct_max_drawdown": float(np.quantile(self.max_drawdown, 0.05)),   # 5th pct of (negative) MDD = worst 5%
            "prob_drawdown_over_50pct": float((self.max_drawdown < -0.50).mean()),
        }


# ---------------------------------------------------------------------------- drift / vol inputs
def drift_annual(mode: str, r: pd.Series, rf_annual: float, beta: float, erp: float, td: int = 252) -> float:
    if mode == "capm":
        return float(rf_annual + beta * erp)
    if mode == "historical":
        return float(r.mean() * td)
    if mode == "zero":
        return 0.0
    raise ValueError(mode)


def _paths_from_returns(R: np.ndarray, s0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R: (n, T) simple returns -> wealth paths (n, T+1), terminal, max drawdown."""
    R = np.clip(R, -0.99, None)
    W = np.concatenate([np.ones((R.shape[0], 1)), np.cumprod(1.0 + R, axis=1)], axis=1)
    peak = np.maximum.accumulate(W, axis=1)
    mdd = (W / peak - 1.0).min(axis=1)
    return W * s0, W[:, -1] * s0, mdd


# ---------------------------------------------------------------------------- generators of standardised shocks
def _shocks_normal(rng, n, T):
    return rng.standard_normal((n, T))


def _fit_t(r: pd.Series) -> float:
    nu, _, _ = stats.t.fit(r.dropna() - r.dropna().mean())
    return float(np.clip(nu, 2.5, 30.0))


def _shocks_t(rng, n, T, nu):
    return rng.standard_t(nu, size=(n, T)) / np.sqrt(nu / (nu - 2.0))


def _shocks_bootstrap(rng, n, T, resid: np.ndarray, block: int):
    """Stationary block bootstrap of standardised, demeaned historical returns."""
    L = len(resid)
    nblocks = int(np.ceil(T / block))
    starts = rng.integers(0, L - block, size=(n, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    return resid[idx]


def _garch_returns(r: pd.Series, n: int, T: int, seed: int) -> tuple[np.ndarray, dict]:
    """Simulate demeaned daily returns (in decimal) from GARCH(1,1)-t fitted on the history."""
    from arch import arch_model

    pct = (r.dropna() * 100.0)
    am = arch_model(pct, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", options={"maxiter": 500})
    mu = res.params["mu"]
    # simulate manually from fitted parameters, starting at the last conditional variance (today's regime)
    om, a, b, nu = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]
    rng = np.random.default_rng(seed)
    h = np.full(n, float(res.conditional_volatility.iloc[-1] ** 2))
    last_eps = float(res.resid.iloc[-1])
    h = om + a * last_eps**2 + b * h               # one-step-ahead variance for day T+1
    out = np.empty((n, T))
    scale = np.sqrt((nu - 2.0) / nu)
    for t in range(T):
        z = rng.standard_t(nu, size=n) * scale     # unit-variance t shock
        eps = np.sqrt(h) * z
        out[:, t] = eps
        h = om + a * eps**2 + b * h
    params = {"omega": float(om), "alpha": float(a), "beta": float(b), "nu": float(nu), "persistence": float(a + b),
              "current_annualised_vol": float(np.sqrt(res.conditional_volatility.iloc[-1] ** 2 * 252) / 100.0),
              "long_run_annualised_vol": float(np.sqrt(om / max(1 - a - b, 1e-6) * 252) / 100.0)}
    return out / 100.0, params


# ---------------------------------------------------------------------------- main entry
def simulate(
    r: pd.Series,
    model: str,
    n_sims: int = 20000,
    horizon: int = 252,
    drift: float = 0.10,
    seed: int = 42,
    block: int = 21,
    s0: float = 100.0,
    td: int = 252,
) -> SimulationResult:
    if n_sims < 10000:
        raise ValueError("n_sims must be >= 10,000")
    r = r.dropna()
    rng = np.random.default_rng(seed)
    mu_d = drift / td
    sd_d = float(r.std(ddof=1))
    params: dict = {"drift_annual": drift, "daily_vol": sd_d, "annualised_vol": sd_d * np.sqrt(td)}
    if model == "normal":
        eps = sd_d * _shocks_normal(rng, n_sims, horizon)
    elif model == "student_t":
        nu = _fit_t(r)
        params["nu"] = nu
        eps = sd_d * _shocks_t(rng, n_sims, horizon, nu)
    elif model == "bootstrap":
        resid = ((r - r.mean()) / 1.0).to_numpy()
        eps = _shocks_bootstrap(rng, n_sims, horizon, resid, block)
        params["block"] = block
    elif model == "garch_t":
        eps, gp = _garch_returns(r, n_sims, horizon, seed)
        params.update(gp)
    else:
        raise ValueError(f"unknown model {model}")
    R = mu_d + eps
    W, term, mdd = _paths_from_returns(R, s0)
    qp = pd.DataFrame({f"p{int(q*100):02d}": np.quantile(W, q, axis=0) for q in (0.05, 0.25, 0.5, 0.75, 0.95)})
    qp.index.name = "day"
    flat = np.clip(R, -0.99, None).ravel()
    sample = rng.choice(flat, size=min(400_000, flat.size), replace=False)
    return SimulationResult(model, params, term, mdd, W[: min(200, n_sims)], qp, sample)


def run_all(r: pd.Series, drift: float, n_sims: int = 20000, horizon: int = 252, seed: int = 42, block: int = 21) -> dict[str, SimulationResult]:
    return {m: simulate(r, m, n_sims, horizon, drift, seed, block) for m in MODELS}


def summary_table(results: dict[str, SimulationResult], s0: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame({m: res.summary(s0) for m, res in results.items()}).T.drop(columns=["model"])


def tail_comparison(hist: pd.Series, results: dict[str, SimulationResult], ks=(2, 3, 4, 5)) -> pd.DataFrame:
    """Frequency of 'k-sigma' down days: history vs what each simulation model implies (k measured in historical sd)."""
    hist = hist.dropna()
    mu, sd = hist.mean(), hist.std(ddof=1)
    rows = {"Historical (observed)": {k: float((hist < mu - k * sd).mean()) for k in ks}}
    for m, res in results.items():
        rows[f"{m}"] = {k: float((res.daily_returns_sample < mu - k * sd).mean()) for k in ks}
    rows["Normal (theory)"] = {k: float(stats.norm.cdf(-k)) for k in ks}
    out = pd.DataFrame(rows).T
    out.columns = [f"<= -{k} sd" for k in ks]
    return out
