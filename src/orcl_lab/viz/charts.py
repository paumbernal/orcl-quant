"""Chart registry: builds every chart from an AnalysisBundle."""
from __future__ import annotations

from . import charts_fundamentals, charts_market, charts_models, charts_valuation
from .theme import Chart

PAGES = {
    1: "Executive summary", 2: "Price & technical analysis", 3: "Fundamentals", 4: "AI / RPO analysis", 5: "Valuation",
    6: "Quantitative factor model", 7: "Risk & Monte Carlo", 8: "Scenario analysis", 9: "Investment score",
}


def build_all(bundle, price_start: str | None = "2018-01-01") -> dict[str, Chart]:
    charts: list[Chart] = []
    charts += charts_market.build(bundle, price_start)
    charts += charts_fundamentals.build(bundle)
    charts += charts_valuation.build(bundle)
    charts += charts_models.build(bundle, price_start)
    return {c.key: c for c in charts}


def for_page(charts: dict[str, Chart], page: int) -> list[Chart]:
    return [c for c in charts.values() if c.page == page]
