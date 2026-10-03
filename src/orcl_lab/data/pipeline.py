"""End-to-end, reproducible data pipeline.

    prices -> risk-free -> SEC XBRL -> earnings events / press-release KPIs -> consensus snapshot
           -> fundamentals assembly -> validation report -> manifest

Every step writes into ``data/processed`` and registers itself (source, timestamp, hash) in
``data/manifest.json``. ``use_cache=True`` re-uses ``data/raw`` instead of re-downloading.
"""
from __future__ import annotations

import json
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from ..analytics.fundamentals import build_annual, build_fundamentals
from ..config import Config
from . import consensus, earnings, peer_fundamentals, prices, riskfree, sec_facts
from .manifest import Manifest
from .store import Store

log = logging.getLogger(__name__)
STEPS = ("prices", "riskfree", "sec", "earnings", "consensus", "peers", "fundamentals")


def assemble_fundamentals(cfg: Config, manifest: Manifest) -> pd.DataFrame:
    st = Store(cfg)
    q_x = pd.read_csv(cfg.processed_dir / "fundamentals_quarterly_xbrl.csv", index_col=0, parse_dates=["period_end", "filed"])
    a_x = pd.read_csv(cfg.processed_dir / "fundamentals_annual_xbrl.csv", index_col=0, parse_dates=True)
    q = build_fundamentals(q_x, st.events(), st.kpis(), cfg)
    a = build_annual(a_x)
    q.to_csv(cfg.processed_dir / "fundamentals_quarterly.csv")
    a.to_csv(cfg.processed_dir / "fundamentals_annual.csv")
    manifest.record(
        "fundamentals_quarterly", kind="HISTORICAL", file=cfg.processed_dir / "fundamentals_quarterly.csv", df=q,
        source="SEC EDGAR XBRL + Oracle earnings releases (derived ratios computed in analytics/fundamentals.py)",
        notes="USD in $ millions. avail_date = first session reflecting the release (point-in-time safe).",
    )
    manifest.record(
        "fundamentals_annual", kind="HISTORICAL", file=cfg.processed_dir / "fundamentals_annual.csv", df=a,
        source="SEC EDGAR XBRL (10-K fiscal-year values)",
    )
    return q


def validate(cfg: Config) -> dict:
    """Hard data-quality checks. Raises on structural errors, records softer findings in the report."""
    st = Store(cfg)
    q, a, px = st.fundamentals(), st.annual(), st.prices()
    report: dict = {"checks": [], "warnings": []}

    def ok(name: str, passed: bool, detail: str = "", hard: bool = True):
        report["checks"].append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed and hard:
            raise AssertionError(f"data validation failed: {name} {detail}")
        if not passed:
            report["warnings"].append(f"{name}: {detail}")

    # statements
    ident = (q["revenue"] - q["costs_and_expenses"] - q["operating_income"]).abs().max()
    ok("income statement identity (revenue - costs = operating income)", ident < 1.0, f"max abs error ${ident:.3f}m")
    for col in ("revenue", "operating_income", "ocf", "capex"):
        s = q[col].rolling(4).sum()
        both = pd.concat([s[s.index.month == cfg.project.fiscal_year_end_month], a[col]], axis=1, keys=["q", "a"]).dropna()
        err = ((both.q - both.a).abs() / both.a.abs().clip(lower=1)).max()
        ok(f"four quarters sum to fiscal year: {col}", err < 0.005, f"max rel. error {err:.4%}")
    gaps = q.index.to_series().diff().dt.days.dropna()
    ok("no missing fiscal quarters", gaps.max() <= 100, f"largest gap {int(gaps.max())} days")
    ok("availability after period end", bool((q["avail_date"] > q.index).all()))
    # prices
    ok("price index strictly increasing", px.index.is_monotonic_increasing and not px.index.has_duplicates)
    last = px.index[-1]
    ok("price data is recent", (pd.Timestamp.today() - last).days <= 10, f"last bar {last.date()}", hard=False)
    # KPI parse consistency
    cons = q["oci_revenue"].notna().sum()
    bad = int(((q["cloud_revenue"] - q["oci_revenue"] - q["saas_revenue"]).abs() > 210).sum()) if cons else 0
    ok("cloud = OCI + SaaS where all three parsed", bad == 0, f"{bad} quarters inconsistent", hard=False)
    if cons == 0:
        report["warnings"].append("No OCI revenue parsed - set SEC_USER_AGENT and re-run, or fill data/manual/kpi_overrides.csv")
    report["n_quarters"] = int(len(q))
    report["latest_quarter"] = str(q.index[-1].date())
    report["price_last_date"] = str(last.date())
    (cfg.processed_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_pipeline(cfg: Config, steps: Iterable[str] = STEPS, use_cache: bool = False) -> Manifest:
    manifest = Manifest(cfg.path("data", "manifest.json"))
    steps = list(steps)
    if "prices" in steps:
        log.info("[1/7] prices")
        prices.build_price_files(cfg, manifest, use_cache=use_cache)
    if "riskfree" in steps:
        log.info("[2/7] risk-free rates")
        riskfree.build_riskfree(cfg, manifest)
    if "sec" in steps:
        log.info("[3/7] SEC XBRL fundamentals")
        sec_facts.run(cfg, manifest, use_cache=use_cache)
    if "earnings" in steps:
        log.info("[4/7] earnings events + press-release KPIs")
        earnings.run(cfg, manifest, use_cache=use_cache)
    if "consensus" in steps:
        log.info("[5/7] consensus snapshot")
        consensus.run(cfg, manifest)
    if "peers" in steps:
        log.info("[6/7] peer fundamentals (capital-intensity anchors)")
        peer_fundamentals.run(cfg, manifest)
    if "fundamentals" in steps:
        log.info("[7/7] fundamentals assembly + validation")
        assemble_fundamentals(cfg, manifest)
        validate(cfg)
    manifest.save()
    return manifest
