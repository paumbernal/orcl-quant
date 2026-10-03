"""Static export of every chart to ``visualisations/`` (PNG via kaleido, plus a self-contained interactive HTML index)."""
from __future__ import annotations

import logging
from pathlib import Path

import plotly.io as pio

from ..config import Config, load_config
from ..reporting.runner import load_bundle
from . import theme as T
from .charts import PAGES, build_all

log = logging.getLogger(__name__)
pio.defaults.mathjax = None      # '$...$' in chart text (dollar amounts) must never be parsed as LaTeX


def _png_ok() -> bool:
    try:
        import kaleido  # noqa: F401
        return True
    except Exception:
        return False


def export_all(cfg: Config | None = None, formats=("png", "html"), price_start: str | None = "2018-01-01") -> int:
    cfg = cfg or load_config()
    bundle = load_bundle(cfg)
    charts = build_all(bundle, price_start)
    out = cfg.charts_dir
    n = 0
    do_png = "png" in formats and _png_ok()
    if "png" in formats and not do_png:
        log.warning("kaleido not available - skipping PNG export")
    for c in charts.values():
        page_dir = out / f"{c.page}_{PAGES[c.page].lower().replace(' & ', '_').replace(' / ', '_').replace(' ', '_')}"
        page_dir.mkdir(parents=True, exist_ok=True)
        fig = T.with_footer(c.fig, c)
        if do_png:
            try:
                fig.write_image(str(page_dir / f"{c.key}.png"), width=1200, scale=2)
            except Exception as exc:  # pragma: no cover - environment dependent (Chrome missing)
                log.warning("PNG export failed for %s: %s", c.key, exc)
                do_png = False
        n += 1
    if "html" in formats:
        _write_index(charts, out)
    return n


def _write_index(charts, out: Path) -> None:
    parts = ["<!doctype html><meta charset='utf-8'><title>ORCL Quant Lab - charts</title>"
             "<style>body{font-family:system-ui,sans-serif;max-width:1240px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}"
             "h1{font-size:22px}h2{margin-top:40px;font-size:18px;border-bottom:1px solid #e1e0d9;padding-bottom:6px}"
             ".note{color:#52514e;font-size:13px;margin:4px 0 28px}</style><h1>ORCL Quantitative Investment & AI Valuation Lab - all charts</h1>"]
    first = True
    for page, name in PAGES.items():
        cs = [c for c in charts.values() if c.page == page]
        if not cs:
            continue
        parts.append(f"<h2>Page {page}: {name}</h2>")
        for c in cs:
            parts.append(c.fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displaylogo": False}))
            parts.append(f"<p class='note'><b>What this shows:</b> {c.explanation}<br><i>Source: {c.source} | Data: {c.date_range}</i></p>")
            first = False
    (out / "all_charts.html").write_text("\n".join(parts), encoding="utf-8")
