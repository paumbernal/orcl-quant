import numpy as np
import pandas as pd
import pytest

from orcl_lab.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def rng():
    return np.random.default_rng(123)


def make_prices(n=900, seed=1, drift=0.0004, vol=0.015, start="2020-01-01", cols=("ORCL", "SPY", "QQQ", "SMH", "NVDA", "MSFT", "AMZN", "GOOGL")):
    r = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    mkt = r.normal(drift, vol, n)
    data = {}
    for i, c in enumerate(cols):
        beta = 0.8 + 0.1 * i
        data[c] = 100 * np.cumprod(1 + beta * mkt + r.normal(0, vol / 2, n))
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def prices():
    return make_prices()
