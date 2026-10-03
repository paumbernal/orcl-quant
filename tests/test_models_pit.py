"""Look-ahead, leakage and honesty checks for the statistical models, signal and backtest."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from orcl_lab.config import load_config, validate_config
from orcl_lab.models import backtest as BT
from orcl_lab.models import features as F
from orcl_lab.models import score as SC
from orcl_lab.models import signal as SG
from orcl_lab.models.eventstudy import run_event_study


# ---------------------------------------------------------------- config guards
def test_config_loads_and_weights_sum_to_one(cfg):
    assert sum(cfg.score.weights.to_dict().values()) == pytest.approx(1.0)
    assert cfg.risk.monte_carlo.n_sims >= 10_000


def test_config_validation_rejects_bad_weights(cfg):
    bad = cfg.with_overrides(score={"weights": {**cfg.score.weights.to_dict(), "growth": 0.9}})
    with pytest.raises(ValueError):
        validate_config(bad)
    with pytest.raises(ValueError):
        validate_config(cfg.with_overrides(risk={"monte_carlo": {**cfg.risk.monte_carlo.to_dict(), "n_sims": 1000}}))


# ---------------------------------------------------------------- point-in-time features
def test_expanding_rank_ignores_the_future():
    s = pd.Series(np.random.default_rng(1).normal(size=600))
    full = F.expanding_pct_rank(s, 50)
    for T in (100, 300, 599):
        trunc = F.expanding_pct_rank(s.iloc[: T + 1], 50)
        assert trunc.iloc[-1] == pytest.approx(full.iloc[T])


def test_asof_join_never_uses_a_quarter_before_its_availability_date():
    q = pd.DataFrame({"x": [1.0, 2.0, 3.0], "avail_date": pd.to_datetime(["2024-03-12", "2024-06-12", "2024-09-11"])},
                     index=pd.to_datetime(["2023-11-30", "2024-02-29", "2024-05-31"]))
    days = pd.bdate_range("2024-03-01", "2024-09-30")
    out = F.asof_join(days, q, ["x"])
    assert out.loc["2024-03-11", "x"] != out.loc["2024-03-11", "x"]               # NaN: nothing available yet
    assert out.loc["2024-03-12", "x"] == 1.0
    assert out.loc["2024-06-11", "x"] == 1.0 and out.loc["2024-06-12", "x"] == 2.0
    assert out.loc["2024-09-10", "x"] == 2.0 and out.loc["2024-09-11", "x"] == 3.0


def test_score_weights_and_range():
    idx = pd.bdate_range("2020-01-01", periods=300)
    comp = pd.DataFrame({c: np.random.default_rng(i).uniform(0, 100, 300) for i, c in enumerate(["growth", "valuation", "momentum", "cash_flow", "balance_sheet", "market_risk"])}, index=idx)
    w = {"growth": .25, "valuation": .20, "momentum": .20, "cash_flow": .15, "balance_sheet": .10, "market_risk": .10}
    t = SC.total_score(comp, w)
    assert t["score"].between(0, 100).all()
    assert t["score"].iloc[0] == pytest.approx(sum(comp.iloc[0][k] * v for k, v in w.items()))
    comp.iloc[0, 0] = np.nan                                                      # missing component -> remaining weights renormalised
    t2 = SC.total_score(comp, w)
    assert t2["weight_coverage"].iloc[0] == pytest.approx(0.75)
    assert 0 <= t2["score"].iloc[0] <= 100


def test_signal_thresholds_use_only_training_sample():
    idx = pd.bdate_range("2012-01-01", periods=2000)
    comp = pd.Series(np.random.default_rng(2).normal(size=2000), index=idx)
    q = {"strong_bearish": .10, "bearish": .30, "bullish": .70, "strong_bullish": .90}
    cut = idx[999]
    t1 = SG.fit_thresholds(comp, cut, q)
    comp2 = comp.copy()
    comp2.iloc[1000:] += 50                                                       # change ONLY post-training data
    assert SG.fit_thresholds(comp2, cut, q) == t1
    states = SG.classify(comp.iloc[:5], t1)
    assert set(states.dropna()) <= set(SG.STATES)


# ---------------------------------------------------------------- backtest honesty
def _ohlc(n=600, seed=4):
    px = make_prices(n, seed)["ORCL"]
    idx = px.index
    rng = np.random.default_rng(seed)
    gap = rng.normal(0, 0.005, n)
    open_ = px.shift(1).fillna(px.iloc[0]) * (1 + gap)
    return pd.DataFrame({"adj_open": open_, "adj_close": px}, index=idx)


def test_full_exposure_without_costs_matches_buy_and_hold():
    o = _ohlc()
    sim = BT.simulate(o, pd.Series(1.0, index=o.index), pd.Series(0.0, index=o.index), 0.0)
    bh = o["adj_close"].pct_change().reindex(sim.index)
    assert ((1 + sim["ret"]).prod()) == pytest.approx((1 + bh).prod(), rel=1e-9)


def test_costs_reduce_returns_and_zero_exposure_earns_cash():
    o = _ohlc()
    rf = pd.Series(0.0001, index=o.index)
    flip = pd.Series(np.tile([1.0, 0.0], len(o) // 2 + 1)[: len(o)], index=o.index)
    free, costly = BT.simulate(o, flip, rf, 0.0), BT.simulate(o, flip, rf, 20.0)
    assert (1 + costly["ret"]).prod() < (1 + free["ret"]).prod()
    cash = BT.simulate(o, pd.Series(0.0, index=o.index), rf, 10.0)
    assert (cash["ret"] == pytest.approx(0.0001)) if False else np.allclose(cash["ret"], 0.0001)


def test_future_signal_changes_do_not_alter_past_pnl():
    o = _ohlc()
    base = pd.Series(np.where(np.arange(len(o)) % 40 < 20, 1.0, 0.25), index=o.index)
    t = 300
    changed = base.copy()
    changed.iloc[t:] = 0.0                                                         # alter only decisions from day t on
    rf = pd.Series(0.0, index=o.index)
    a, b = BT.simulate(o, base, rf, 10.0), BT.simulate(o, changed, rf, 10.0)
    assert np.allclose(a["ret"].iloc[: t - 1], b["ret"].iloc[: t - 1])


def test_schedule_respects_rebalance_cadence_and_min_hold():
    idx = pd.bdate_range("2020-01-01", periods=100)
    target = pd.Series(np.where(np.arange(100) % 2 == 0, 1.0, 0.0), index=idx)     # whipsawing target
    s = BT.target_schedule(target, rebalance_days=5, min_hold=10)
    changes = s.dropna().ne(s.dropna().shift()).sum() - 1
    assert changes <= 100 // 10
    assert s.iloc[: 0].empty and s.dropna().index[0] == idx[0]


# ---------------------------------------------------------------- event study
def test_event_study_recovers_injected_abnormal_return(cfg):
    px = make_prices(700, seed=9)
    rets = px.pct_change()
    day = 500
    px_adj = px.copy()
    shock = 0.10
    factor = np.ones(len(px))
    factor[day:] = 1 + shock                                                       # +10% jump on the reaction day, persists
    px_adj["ORCL"] = px["ORCL"] * factor
    react = px.index[day]
    ev = pd.DataFrame({"reaction_date": [react], "eps_surprise_pct": [5.0]}, index=pd.to_datetime(["2021-08-31"]))
    q = pd.DataFrame({"label": ["x"], "revenue_yoy": [0.1], "eps_ttm_yoy": [0.1], "oci_growth": [np.nan], "rpo_yoy": [np.nan], "rpo_qoq": [np.nan],
                      "guid_rev_growth_lo": [np.nan], "guid_rev_growth_hi": [np.nan], "guid_cloud_growth_lo": [np.nan], "guid_cloud_growth_hi": [np.nan],
                      "guid_eps_nongaap_lo": [np.nan], "guid_eps_nongaap_hi": [np.nan], "fcf_margin_yoy_change": [0.0]}, index=ev.index)
    res = run_event_study(px_adj, ev, q, cfg)
    assert res.table["car_1d"].iloc[0] == pytest.approx(shock, abs=0.03)
    assert res.table["car_20d"].iloc[0] == pytest.approx(shock, abs=0.08)
