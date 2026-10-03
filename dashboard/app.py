"""ORCL Quantitative Investment & AI Valuation Lab - Streamlit dashboard.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="ORCL Quant Lab", page_icon=None, layout="wide", initial_sidebar_state="expanded")

from common import get_bundle  # noqa: E402
import pages_impl as P  # noqa: E402

PAGES = [
    st.Page(P.page_overview, title="1. Project overview & research report", url_path="overview", default=True),
    st.Page(P.page_summary, title="2. Executive summary", url_path="summary"),
    st.Page(P.page_price, title="3. Price & technical analysis", url_path="price"),
    st.Page(P.page_fundamentals, title="4. Fundamentals", url_path="fundamentals"),
    st.Page(P.page_rpo, title="5. AI / RPO analysis", url_path="rpo"),
    st.Page(P.page_valuation, title="6. Valuation", url_path="valuation"),
    st.Page(P.page_factor, title="7. Quantitative factor model", url_path="factor"),
    st.Page(P.page_risk, title="8. Risk & Monte Carlo", url_path="risk"),
    st.Page(P.page_scenarios, title="9. Scenario analysis", url_path="scenarios"),
    st.Page(P.page_score, title="10. Investment score", url_path="score"),
]

b = get_bundle()
with st.sidebar:
    st.markdown("### ORCL Quant Lab")
    st.caption("Can Oracle's AI/cloud growth justify its valuation and capital needs?")
    st.markdown(f"**Price** ${b.meta['price']:,.2f}  \n*close {b.meta['price_date']}*")
    st.markdown(f"**Latest quarter** {b.meta['latest_quarter']}  \n*ended {b.meta['latest_quarter_end']}*")
    st.markdown(f"**Consensus snapshot** {b.meta['consensus_date']}")
    st.markdown(f"**Model-driven conclusion:** `{b.conclusion['label']}`")
    st.caption("Research and education only - not investment advice. Historical data, consensus forecasts, assumptions and model outputs are labelled on every page.")

st.navigation(PAGES).run()
