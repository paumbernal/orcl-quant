import numpy as np
import pandas as pd
import pytest

from orcl_lab.analytics import returns as R
from orcl_lab.analytics import technicals as T


def test_drawdown_known_path():
    r = pd.Series([0.10, -0.20, 0.05, 0.30], index=pd.bdate_range("2024-01-01", periods=4))
    dd = R.drawdown_series(r)
    # wealth: 1.10, 0.88, 0.924, 1.2012  -> trough is -20% below the 1.10 peak
    assert dd.min() == pytest.approx(-0.20)
    assert dd.iloc[-1] == pytest.approx(0.0)
    assert R.max_drawdown(r) == pytest.approx(-0.20)
    s = R.drawdown_stats(r)
    assert s.trough_date == r.index[1] and s.recovery_date == r.index[3]


def test_cagr_and_vol_scaling():
    r = pd.Series(0.001, index=pd.bdate_range("2024-01-01", periods=252))
    assert R.cagr(r) == pytest.approx(1.001 ** 252 - 1)
    x = pd.Series(np.random.default_rng(0).normal(0, 0.01, 5000))
    assert R.annualised_vol(x) == pytest.approx(x.std() * np.sqrt(252))


def test_sharpe_sortino_properties():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 2000))
    assert R.sharpe_ratio(r, 0.0) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    # adding a positive risk-free rate lowers Sharpe
    assert R.sharpe_ratio(r, 0.0002) < R.sharpe_ratio(r, 0.0)
    # no negative days -> downside deviation zero -> Sortino undefined (NaN), not inf
    assert np.isnan(R.sortino_ratio(pd.Series([0.01, 0.02, 0.03]), 0.0))


def test_capm_recovers_known_beta_and_alpha():
    rng = np.random.default_rng(2)
    m = pd.Series(rng.normal(0.0004, 0.01, 4000), index=pd.bdate_range("2005-01-01", periods=4000))
    a = 1.7 * m + 0.0002 + pd.Series(rng.normal(0, 0.004, 4000), index=m.index)
    res = R.capm(a, m, 0.0)
    assert res.beta == pytest.approx(1.7, abs=0.05)
    assert res.alpha_daily == pytest.approx(0.0002, abs=0.0004)
    rb = R.rolling_beta(a, m, 252).dropna()
    assert abs(rb.mean() - 1.7) < 0.1


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert T.rsi(up).dropna().iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(100, 1, -1, dtype=float))
    assert T.rsi(down).dropna().iloc[-1] == pytest.approx(0.0, abs=1e-9)
    noisy = pd.Series(100 + np.cumsum(np.random.default_rng(3).normal(0, 1, 500)))
    r = T.rsi(noisy).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_macd_zero_on_flat_and_atr_constant_range():
    flat = pd.Series(50.0, index=range(200))
    assert T.macd(flat).dropna().abs().max().max() == pytest.approx(0.0)
    close = pd.Series(100.0, index=range(100))
    atr = T.atr(close + 1, close - 1, close, 14).dropna()
    assert atr.iloc[-1] == pytest.approx(2.0)


def test_relative_volume_excludes_today():
    v = pd.Series([100.0] * 25 + [300.0])
    rv = T.relative_volume(v, 20)
    assert rv.iloc[-1] == pytest.approx(3.0)       # today's 300 is divided by the PRIOR 20-day average (100)


def test_moving_average_windows():
    s = pd.Series(np.arange(300, dtype=float))
    ma = T.moving_averages(s, (20, 50, 100, 200))
    assert list(ma.columns) == ["ma_20", "ma_50", "ma_100", "ma_200"]
    assert ma["ma_200"].first_valid_index() == 199
    assert ma["ma_20"].iloc[-1] == pytest.approx(s.iloc[-20:].mean())
