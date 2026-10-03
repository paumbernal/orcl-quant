"""Shared dashboard helpers: cached bundle/chart loading and the standard chart + caption block."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Streamlit renders "$...$" in Markdown as LaTeX, which silently swallows dollar amounts ("$137.30 ... $169" -> maths).
# Escape every unescaped "$" in text passed to the text elements below so financial figures always display literally.
_DOLLAR = re.compile(r"(?<!\\)\$")


def _escape(x):
    return _DOLLAR.sub(r"\$", x) if isinstance(x, str) else x


def _wrap_text_element(fn):
    def wrapped(*args, **kwargs):
        return fn(*[_escape(a) for a in args], **{k: _escape(v) for k, v in kwargs.items()})
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped


if not getattr(st, "_orcl_dollar_patch", False):
    for _name in ("markdown", "caption", "info", "warning", "error", "success", "subheader", "header", "title", "metric"):
        setattr(st, _name, _wrap_text_element(getattr(st, _name)))
    st._orcl_dollar_patch = True

from orcl_lab.config import load_config  # noqa: E402
from orcl_lab.reporting.runner import load_bundle, run_all_analyses  # noqa: E402
from orcl_lab.viz.charts import build_all  # noqa: E402


@st.cache_resource(show_spinner="Loading analysis bundle (run `orcl-lab all` to refresh)...")
def get_cfg():
    return load_config()


@st.cache_resource(show_spinner="Preparing the analysis (first visit on a fresh install takes about a minute; later pages are instant)...")
def get_bundle():
    # No st.* messages here: Streamlit replays elements from cached functions on every page.
    cfg = get_cfg()
    try:
        return load_bundle(cfg, rebuild_if_missing=False)
    except FileNotFoundError:
        return run_all_analyses(cfg)


@st.cache_resource(show_spinner="Building charts...")
def get_charts(price_start: str | None):
    return build_all(get_bundle(), price_start)


def chart(key: str, price_start: str | None = "2018-01-01", height: int | None = None):
    """Render one chart with title, explanation and provenance (every chart carries all of these)."""
    c = get_charts(price_start)[key]
    fig = c.fig
    if height:
        fig = fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch", key=f"{key}-{price_start}", config={"displaylogo": False})
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key=f"{key}-{price_start}", config={"displaylogo": False})
    st.caption(f"**What this shows.** {c.explanation}")
    st.caption(f"Source: {c.source}  |  Data: {c.date_range}")
    st.divider()


def df_show(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def kind_badge(kind: str) -> str:
    return {"HISTORICAL": ":blue[HISTORICAL DATA]", "CONSENSUS": ":orange[CONSENSUS FORECAST]", "ASSUMPTION": ":violet[ASSUMPTION]",
            "MODEL": ":green[MODEL OUTPUT]", "MANUAL": ":gray[MANUAL]"}.get(kind, kind)


def fmt_pct(x, d=0):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}%}"


def fmt_usd(x, d=0):
    return "n/a" if x is None or pd.isna(x) else (f"-${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}")


def header(title: str, subtitle: str, kinds: list[str]):
    st.title(title)
    st.markdown(subtitle)
    st.markdown(" ".join(kind_badge(k) for k in kinds))
