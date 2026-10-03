"""EXPERIMENTAL five-state classifier for ORCL.

    STRONG BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG BEARISH

This is a research device for studying whether a transparent, pre-specified combination of the project's own variables
carried any information - it is NOT a trading recommendation and makes no accuracy claim.

Construction (no fitted weights, no look-ahead)
-----------------------------------------------
Six components, each an expanding-window percentile rank (so a value only ever compares with the past) re-centred to (-1, 1]:

    momentum           mean rank of 12-1 month return and price vs 200-day average
    valuation          cheapness = 1 - mean rank of trailing P/E and EV/EBITDA
    earnings revision  rank of the latest EPS surprise vs consensus (point-in-time proxy: historical consensus *revisions*
                       are not available from free sources; the snapshot history file will supply them going forward)
    OCI growth         rank of cloud-infrastructure revenue growth
    RPO growth         rank of remaining-performance-obligation growth
    FCF trend          rank of the 4-quarter change in TTM free-cash-flow margin

The composite is the EQUAL-weighted mean of the components available on the day (at least ``min_components``; OCI and RPO only
exist from FY23 / FY20 respectively, so the live mix changes over time - reported as `n_components`).
State thresholds are quantiles of the composite over the TRAINING sample only (config `signal.quantiles`); exposure per state
is fixed in advance (config `signal.exposure`). Nothing is optimised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config

STATES = ["STRONG_BEARISH", "BEARISH", "NEUTRAL", "BULLISH", "STRONG_BULLISH"]


def component_frame(panel: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame(index=panel.index)
    c["momentum"] = panel[["mom_12_1__rank", "price_vs_200dma__rank"]].mean(axis=1, skipna=False)
    c["valuation"] = 1.0 - panel[["pe_trailing__rank", "ev_ebitda__rank"]].mean(axis=1, skipna=False)
    c["earnings_revision"] = panel["eps_surprise_pct__rank"]
    c["oci_growth"] = panel["oci_growth__rank"]
    c["rpo_growth"] = panel["rpo_yoy__rank"]
    c["fcf_trend"] = panel["fcf_margin_yoy_change__rank"]
    return 2.0 * c - 1.0


def composite(panel: pd.DataFrame, components: list[str], min_components: int = 3) -> pd.DataFrame:
    c = component_frame(panel)[components]
    n = c.notna().sum(axis=1)
    comp = c.mean(axis=1, skipna=True).where(n >= min_components)
    return pd.DataFrame({"composite": comp, "n_components": n}).join(c)


@dataclass
class SignalModel:
    thresholds: dict[str, float]
    frame: pd.DataFrame
    train_end: pd.Timestamp


def fit_thresholds(comp: pd.Series, train_end: pd.Timestamp, q: dict[str, float]) -> dict[str, float]:
    tr = comp[comp.index <= train_end].dropna()
    if len(tr) < 250:
        raise ValueError("not enough training observations to set thresholds")
    return {k: float(tr.quantile(v)) for k, v in q.items()}


def classify(comp: pd.Series, thr: dict[str, float]) -> pd.Series:
    def f(x):
        if np.isnan(x):
            return None
        if x <= thr["strong_bearish"]:
            return "STRONG_BEARISH"
        if x <= thr["bearish"]:
            return "BEARISH"
        if x >= thr["strong_bullish"]:
            return "STRONG_BULLISH"
        if x >= thr["bullish"]:
            return "BULLISH"
        return "NEUTRAL"
    return comp.map(f)


def build_signal(panel: pd.DataFrame, cfg: Config) -> SignalModel:
    comps = list(cfg.signal.components)
    fr = composite(panel, comps)
    train_end = pd.Timestamp(cfg.backtest.train_end)
    thr = fit_thresholds(fr["composite"], train_end, cfg.signal.quantiles.to_dict())
    fr["state"] = classify(fr["composite"], thr)
    fr["target_exposure"] = fr["state"].map(cfg.signal.exposure.to_dict())
    return SignalModel(thr, fr, train_end)
