"""Deterministic, model-driven investment conclusion.

No discretion: five votes (+1 / 0 / -1) are computed from model outputs with thresholds fixed in config.yaml. The sum decides
Bull / Neutral / Bear. Changing the conclusion therefore requires changing a model output or a stated threshold - never a
judgement call made after the fact.

    1. Valuation       base-case AND probability-weighted blended value both >= +15% vs price -> +1; both <= -15% -> -1
    2. Quant score     0-100 score >= 60 -> +1 ; <= 40 -> -1
    3. Experimental signal   (STRONG) BULLISH -> +1 ; (STRONG) BEARISH -> -1
    4. Cash conversion TTM FCF margin > 0 -> +1 ; FCF margin < 0 AND deteriorating YoY -> -1 ; otherwise 0
    5. Balance sheet   base-case path breaches a stress limit (net debt/EBITDA, interest cover) or today's net debt/EBITDA is above the
                       limit -> -1 ; no breach even in the BEAR path and net debt/EBITDA < 0.625 x limit -> +1 ; otherwise 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def build_conclusion(cfg: Config, val, score_latest: float, signal_state: str, q: pd.DataFrame) -> dict:
    c = cfg.conclusion
    A = val.anchors
    base_up = val.scenarios["base"].blended / A.price - 1.0
    pw_up = val.upside_prob_weighted
    if base_up >= c.upside_bull and pw_up >= c.upside_bull:
        v_val = 1
    elif base_up <= c.upside_bear and pw_up <= c.upside_bear:
        v_val = -1
    else:
        v_val = 0
    v_score = 0 if np.isnan(score_latest) else (1 if score_latest >= c.score_bull else -1 if score_latest <= c.score_bear else 0)
    v_sig = {"STRONG_BULLISH": 1, "BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1, "STRONG_BEARISH": -1}.get(signal_state, 0)
    last = q.iloc[-1]
    if last["fcf_margin_ttm"] > 0:
        v_cash = 1
    elif last["fcf_margin_ttm"] < 0 and last["fcf_margin_yoy_change"] < 0:
        v_cash = -1
    else:
        v_cash = 0
    stress = cfg.valuation.stress
    base_d, bear_d = val.scenarios["base"].diagnostics, val.scenarios["bear"].diagnostics
    nd_ebitda = float(last["net_debt_to_ebitda_ttm"])
    base_breach = base_d["first_year_leverage_breach"] != "none" or base_d["interest_cover_breach"] or nd_ebitda > stress.net_debt_to_ebitda_max
    bear_breach = bear_d["first_year_leverage_breach"] != "none" or bear_d["interest_cover_breach"]
    if base_breach:
        v_bs = -1
    elif (not bear_breach) and nd_ebitda < 0.625 * stress.net_debt_to_ebitda_max:
        v_bs = 1
    else:
        v_bs = 0
    votes = pd.DataFrame([
        ("Valuation (scenario-implied value vs price)", v_val, f"base {base_up:+.0%}, probability-weighted {pw_up:+.0%} (thresholds +/-{c.upside_bull:.0%})"),
        ("Quantitative score", v_score, f"score {score_latest:.1f} (bull >= {c.score_bull}, bear <= {c.score_bear})"),
        ("Experimental signal", v_sig, f"state {signal_state.replace('_', ' ')}"),
        ("Cash conversion", v_cash, f"TTM FCF margin {last['fcf_margin_ttm']:.0%}, YoY change {last['fcf_margin_yoy_change']*100:+.0f}pp"),
        ("Balance sheet", v_bs, f"net debt/EBITDA {nd_ebitda:.1f}x (limit {stress.net_debt_to_ebitda_max:.1f}x); base path breach: {base_breach}; bear path breach: {bear_breach}"),
    ], columns=["vote", "value", "evidence"])
    total = int(votes["value"].sum())
    label = "Bull" if total >= c.total_bull else "Bear" if total <= c.total_bear else "Neutral"
    return {"votes": votes, "total": total, "label": label, "base_upside": base_up, "pw_upside": pw_up,
            "rule": f"Bull if the vote total >= +{c.total_bull}, Bear if <= {c.total_bear}, otherwise Neutral (five votes, each -1/0/+1)."}
