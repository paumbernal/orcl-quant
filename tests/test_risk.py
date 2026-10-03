import numpy as np
import pandas as pd
import pytest
from scipy import stats

from orcl_lab.risk import montecarlo as MC
from orcl_lab.risk import var as V


@pytest.fixture()
def normal_returns():
    return pd.Series(np.random.default_rng(5).normal(0.0005, 0.02, 20000))


def test_historical_var_equals_empirical_quantile(normal_returns):
    assert V.historical_var(normal_returns, 0.95) == pytest.approx(-np.quantile(normal_returns, 0.05))


def test_parametric_normal_var_matches_analytic(normal_returns):
    mu, sd = normal_returns.mean(), normal_returns.std(ddof=1)
    expected = -(mu + stats.norm.ppf(0.01) * sd)
    assert V.parametric_var(normal_returns, 0.99, 1, "normal") == pytest.approx(expected)
    # on truly normal data, historical and parametric agree closely
    assert V.historical_var(normal_returns, 0.99) == pytest.approx(expected, rel=0.08)


def test_es_exceeds_var_and_fat_tails_break_normal():
    rng = np.random.default_rng(6)
    fat = pd.Series(rng.standard_t(3, 20000) * 0.01)
    assert V.historical_es(fat, 0.99) > V.historical_var(fat, 0.99)
    # for fat-tailed returns the normal model understates 99% ES
    assert V.parametric_es(fat, 0.99, 1, "normal") < V.historical_es(fat, 0.99)


def test_multiday_historical_var_uses_compounding():
    r = pd.Series(np.random.default_rng(7).normal(0, 0.01, 3000))
    assert V.historical_var(r, 0.95, 10) > V.historical_var(r, 0.95, 1)


def test_cornish_fisher_validity_flag():
    assert V._cf_valid(0.0, 0.5)
    assert not V._cf_valid(1.7, 33.0)             # ORCL-like skew / kurtosis: expansion is non-monotonic


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    lr, p = V.kupiec_pof(50, 1000, 0.05)
    assert p > 0.9
    lr, p = V.kupiec_pof(120, 1000, 0.05)
    assert p < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=bool)
    clustered[100:130] = True
    spread = np.zeros(1000, dtype=bool)
    spread[::33] = True
    assert V.christoffersen_independence(clustered)[1] < 0.01
    assert V.christoffersen_independence(spread)[1] > 0.05


def test_monte_carlo_rejects_too_few_sims(normal_returns):
    with pytest.raises(ValueError):
        MC.simulate(normal_returns, "normal", n_sims=5000)


def test_monte_carlo_is_reproducible_and_matches_drift(normal_returns):
    a = MC.simulate(normal_returns, "normal", 10000, 252, 0.08, seed=3)
    b = MC.simulate(normal_returns, "normal", 10000, 252, 0.08, seed=3)
    assert np.array_equal(a.terminal, b.terminal)
    # E[terminal / start] ~ (1 + drift/252)^252 under this construction
    assert a.terminal.mean() / 100 - 1 == pytest.approx(np.exp(0.08) - 1, abs=0.03)
    assert (a.max_drawdown <= 0).all() and (a.max_drawdown >= -1).all()


def test_monte_carlo_models_run_and_fat_tails_widen_left_tail():
    rng = np.random.default_rng(8)
    r = pd.Series(rng.standard_t(3, 3000) * 0.012)
    res = MC.run_all(r, 0.08, n_sims=10000, horizon=126, seed=1)
    assert set(res) == set(MC.MODELS)
    tails = MC.tail_comparison(r, res)
    assert tails.loc["Historical (observed)", "<= -4 sd"] > tails.loc["Normal (theory)", "<= -4 sd"] * 10
    assert tails.loc["student_t", "<= -4 sd"] > tails.loc["normal", "<= -4 sd"]
