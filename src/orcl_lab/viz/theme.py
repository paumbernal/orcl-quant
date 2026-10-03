"""Visual system: one validated categorical palette, one chart anatomy.

Palette and surface tokens come from a validated reference palette (categorical order chosen so adjacent hues stay
distinguishable under colour-vision deficiency). Colour follows the ENTITY, never its rank: ORCL is always blue, the market
is always neutral grey, bear/base/bull are always red/blue/aqua - a filter can never repaint a series.

Every chart carries: a title, labelled axes, a source + retrieval date footer, the covered date range, and a plain-English
explanation (rendered under the chart in the dashboard and burned into the exported PNG).
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]      # blue <-> neutral <-> red
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

ENTITY = {"ORCL": BLUE, "SPY": MUTED, "QQQ": ORANGE, "SMH": VIOLET, "NVDA": AQUA, "MSFT": MAGENTA, "AMZN": YELLOW, "GOOGL": GREEN}
SCENARIO = {"bear": RED, "base": BLUE, "bull": AQUA}
LABEL = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "SMH": "Semiconductors (SMH)"}
STATE_COLORS = {"STRONG_BEARISH": RED, "BEARISH": "#f2a09f", "NEUTRAL": AXIS, "BULLISH": "#8fd3b8", "STRONG_BULLISH": AQUA}


def lab(t: str) -> str:
    return LABEL.get(t, t)


@dataclass
class Chart:
    key: str
    page: int
    fig: go.Figure
    title: str
    explanation: str
    source: str
    date_range: str

    @property
    def footer(self) -> str:
        return f"Source: {self.source}  |  Data: {self.date_range}"


def span(idx) -> str:
    idx = pd.DatetimeIndex(idx)
    return f"{idx.min().date()} to {idx.max().date()}"


def style(fig: go.Figure, title: str, x_title: str | None = None, y_title: str | None = None, height: int = 430,
          legend: bool = True, y_tickformat: str | None = None, x_tickformat: str | None = None, hover: str = "x unified") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, color=INK, family=FONT)),
        template="none", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK2),
        height=height, margin=dict(l=64, r=24, t=64, b=64), hovermode=hover, showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0, font=dict(size=11, color=INK2), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), title_text=x_title, title_font=dict(color=INK2, size=12),
                     zeroline=False, tickformat=x_tickformat, automargin=True)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor="rgba(0,0,0,0)", tickfont=dict(color=MUTED), title_text=y_title,
                     title_font=dict(color=INK2, size=12), zeroline=False, tickformat=y_tickformat, automargin=True)
    return fig


def add_zero_line(fig: go.Figure, y: float = 0.0, **kw) -> None:
    fig.add_hline(y=y, line=dict(color=AXIS, width=1), **kw)


def wrap(text: str, width: int = 120) -> str:
    return "<br>".join(textwrap.wrap(text, width))


def with_footer(fig: go.Figure, chart: Chart, width: int = 1200, include_explanation: bool = True) -> go.Figure:
    """Copy of the figure with the explanation + source/date footer burned in below the plot (for static PNG export)."""
    f = go.Figure(fig)
    wrap_at = max(60, int(width / 7.4))
    lines = textwrap.wrap(chart.explanation, wrap_at) if include_explanation else []
    foot_lines = textwrap.wrap(chart.footer, wrap_at)
    n_lines = len(lines) + len(foot_lines) + (2 if include_explanation else 0)
    extra = 30 + 17 * n_lines
    base_h = int(fig.layout.height or 430)
    m = fig.layout.margin
    f.update_layout(width=width, height=base_h + extra, margin=dict(l=m.l or 64, r=m.r or 24, t=m.t or 64, b=(m.b or 64) + extra))
    body = ""
    if include_explanation:
        body += "<b>What this shows:</b> " + "<br>".join(lines) + "<br><br>"
    body += f"<span style='color:{MUTED}'>" + "<br>".join(foot_lines) + "</span>"
    f.add_annotation(text=body, xref="paper", yref="paper", x=0, y=0, yshift=-(70), xanchor="left", yanchor="top",
                     showarrow=False, align="left", font=dict(size=11, color=INK2))
    return escape_dollars(f)


def _esc(o):
    if isinstance(o, str):
        return o.replace("$", "&#36;")
    if isinstance(o, dict):
        return {k: (v if k in ("hovertemplate", "texttemplate") else _esc(v)) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_esc(v) for v in o)
    return o


def escape_dollars(fig: go.Figure) -> go.Figure:
    """Kaleido treats '$...$' as LaTeX; dollar amounts must render literally, so '$' becomes the HTML entity &#36;."""
    return go.Figure(_esc(fig.to_dict()))
