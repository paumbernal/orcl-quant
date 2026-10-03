"""DCF / projection engine tests on a synthetic company (no network, no data files)."""
from types import SimpleNamespace

import numpy as np
import pytest

from orcl_lab.valuation.model import Assumptions, project, value_dcf, value_multiples


@pytest.fixture()
def anchors():
    return SimpleNamespace(
        base_fy=2026, seg_base={"oci": 18000.0, "saas": 16000.0, "other": 33000.0}, revenue_base=67000.0,
        cur_q_revenue=19000.0, elapsed_quarters=1, margin_now=0.47, ppe_now=128000.0, amort_ttm=1400.0, nwc_now=-30000.0,
        ttm={"revenue_ttm": 72000.0}, dps=2.0, shares_m=3000.0, net_debt=97000.0, preferred=5000.0, minority=400.0,
    )


def base_assumptions(**kw):
    d = dict(name="t", cons_rev_fy1=90000.0, cons_rev_fy2=130000.0, saas_g1=0.10, other_g1=0.01, phi=0.5, g_term=0.03,
             margin_target=0.42, k_target=1.0, dep_rate=0.10, stub_capex=57000.0, nwc_target=-0.12, tax0=0.13, tax_lr=0.18,
             r_eff=0.044, r_marg=0.068, wacc=0.095, ke=0.11, pe=25.0, ev_ebitda=15.0, ev_revenue=6.0)
    d.update(kw)
    return Assumptions(**d)


def test_segments_sum_to_consensus_and_stub_fraction(anchors):
    P = project(anchors, base_assumptions())
    assert P["revenue_fy"].iloc[0] == pytest.approx(90000.0)
    assert P["revenue_fy"].iloc[1] == pytest.approx(130000.0)
    assert (P["oci_revenue"] + P["saas_revenue"] + P["other_revenue"]).iloc[1] == pytest.approx(130000.0)
    assert P["fraction_of_year"].iloc[0] == pytest.approx(0.75)
    assert P["revenue"].iloc[0] == pytest.approx(90000.0 - 19000.0)          # stub = full year less the reported quarter


def test_net_debt_roll_forward_identity(anchors):
    P = project(anchors, base_assumptions())
    prev = anchors.net_debt
    for _, r in P.iterrows():
        expected = prev - r["ufcf"] + r["interest"] * (1 - r["tax_rate"]) + r["dividends"]
        assert r["net_debt_end"] == pytest.approx(expected)
        prev = r["net_debt_end"]


def test_ppe_rollforward_and_capex_floor(anchors):
    S = base_assumptions()
    P = project(anchors, S)
    ppe_prev = P["ppe_end"].shift(1)
    chk = (ppe_prev + P["capex"] - P["dep"]).iloc[1:]
    assert np.allclose(chk, P["ppe_end"].iloc[1:])
    assert (P["capex"].iloc[1:] >= S.maint_floor * P["dep"].iloc[1:] - 1e-6).all()


def test_dcf_monotonicity(anchors):
    S = base_assumptions()
    p0 = value_dcf(anchors, S).price
    assert value_dcf(anchors, S.with_(wacc=0.11)).price < p0                    # higher discount rate -> lower value
    assert value_dcf(anchors, S.with_(g_term=0.035)).price > p0                 # higher terminal growth -> higher value
    assert value_dcf(anchors, S.with_(growth_shift=0.05)).price > p0            # faster OCI growth -> higher value
    assert value_dcf(anchors, S.with_(margin_target=0.46)).price > p0           # higher margin -> higher value
    assert value_dcf(anchors, S.with_(k_target=1.5)).price < p0                 # heavier capital intensity -> lower value


def test_dcf_rejects_wacc_not_above_growth(anchors):
    with pytest.raises(ValueError):
        value_dcf(anchors, base_assumptions(wacc=0.03, g_term=0.03))


def test_equity_bridge(anchors):
    d = value_dcf(anchors, base_assumptions())
    assert d.equity_value == pytest.approx(d.enterprise_value - anchors.net_debt - anchors.preferred - anchors.minority)
    assert d.price == pytest.approx(d.equity_value / anchors.shares_m)
    assert d.pv_explicit + d.pv_terminal == pytest.approx(d.enterprise_value)


def test_multiples_scale_with_multiple(anchors):
    S = base_assumptions()
    a = value_multiples(anchors, S)
    b = value_multiples(anchors, S.with_(ev_ebitda=20.0, pe=35.0, ev_revenue=9.0))
    assert b.ev_ebitda > a.ev_ebitda and b.forward_pe > a.forward_pe and b.ev_revenue > a.ev_revenue
