"""Quarterly / annual fundamentals from the SEC EDGAR XBRL ``companyfacts`` API.

Why derive periods ourselves?
-----------------------------
In ``companyfacts`` the ``fy`` / ``fp`` fields describe the *filing* a fact was reported in, not the
period it measures (a FY2026 10-K also re-reports FY2024 numbers with ``fy=2026``). We therefore
classify every fact by its own ``start`` / ``end`` dates.

* Income-statement items appear as 3-month values in 10-Qs but cash-flow items only as
  year-to-date (3, 6, 9 months) values, and a standalone fourth quarter is never filed - it is
  derived as ``FY - 9M YTD``.
* Share counts are *averages*, so they are not differenced: Q4 = 4 x FY - (Q1 + Q2 + Q3).
* Where the same period was reported several times we keep the **first-filed** value, so the data
  reflects what investors could actually see on the filing date (point-in-time safe).

Every value therefore traces back to a specific US-GAAP XBRL tag recorded in ``TAG_MAP``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..config import Config
from .http import PoliteClient, sec_headers
from .manifest import Manifest

log = logging.getLogger(__name__)

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


@dataclass(frozen=True)
class TagSpec:
    tags: tuple[str, ...]        # us-gaap tags in priority order (first available per period wins)
    kind: str                    # 'flow' (additive over time), 'avg' (period average), 'instant' (balance sheet)
    unit: str = "USD"
    note: str = ""


TAG_MAP: dict[str, TagSpec] = {
    # ---- income statement ----
    "revenue": TagSpec(("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"), "flow"),
    "operating_income": TagSpec(("OperatingIncomeLoss",), "flow"),
    "net_income": TagSpec(("NetIncomeLoss",), "flow"),
    "eps_diluted": TagSpec(("EarningsPerShareDiluted",), "flow", "USD/shares", "GAAP diluted EPS to common"),
    "costs_and_expenses": TagSpec(("CostsAndExpenses",), "flow"),
    "sales_marketing": TagSpec(("SellingAndMarketingExpense",), "flow"),
    "research_development": TagSpec(("ResearchAndDevelopmentExpense",), "flow"),
    "general_admin": TagSpec(("GeneralAndAdministrativeExpense",), "flow"),
    "restructuring": TagSpec(("RestructuringCharges",), "flow"),
    "acquisition_related": TagSpec(("BusinessCombinationAcquisitionRelatedCosts",), "flow"),
    "amortization_intangibles_is": TagSpec(("AmortizationOfIntangibleAssets",), "flow"),
    "interest_expense": TagSpec(("InterestExpense",), "flow"),
    "investment_income": TagSpec(("InvestmentIncomeInterest",), "flow"),
    "income_tax": TagSpec(("IncomeTaxExpenseBenefit",), "flow"),
    "diluted_shares": TagSpec(("WeightedAverageNumberOfDilutedSharesOutstanding",), "avg", "shares"),
    "basic_shares": TagSpec(("WeightedAverageNumberOfSharesOutstandingBasic",), "avg", "shares"),
    # ---- cash flow ----
    "ocf": TagSpec(("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"), "flow"),
    "capex": TagSpec(("PaymentsToAcquirePropertyPlantAndEquipment",), "flow", note="cash capex (excludes finance leases)"),
    "depreciation": TagSpec(("Depreciation",), "flow"),
    "sbc": TagSpec(("ShareBasedCompensation",), "flow"),
    "dividends_paid": TagSpec(("PaymentsOfDividendsCommonStock",), "flow"),
    "buybacks": TagSpec(("PaymentsForRepurchaseOfCommonStock",), "flow"),
    # ---- balance sheet (instant) ----
    "cash": TagSpec(("CashAndCashEquivalentsAtCarryingValue",), "instant"),
    "marketable_securities": TagSpec(("AvailableForSaleSecuritiesDebtSecuritiesCurrent",), "instant"),
    "debt_current": TagSpec(("NotesPayableCurrent", "DebtCurrent"), "instant"),
    "commercial_paper": TagSpec(("CommercialPaper",), "instant"),
    "debt_noncurrent": TagSpec(("LongTermNotesAndLoans", "LongTermNotesPayable", "LongTermDebtNoncurrent"), "instant"),
    "finance_lease_liability": TagSpec(("FinanceLeaseLiability",), "instant"),
    "operating_lease_liability": TagSpec(("OperatingLeaseLiability",), "instant"),
    "preferred_stock": TagSpec(("PreferredStockValue",), "instant"),
    "minority_interest": TagSpec(("MinorityInterest",), "instant"),
    "total_assets": TagSpec(("Assets",), "instant"),
    "total_equity": TagSpec(("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"), "instant"),
    "current_assets": TagSpec(("AssetsCurrent",), "instant"),
    "current_liabilities": TagSpec(("LiabilitiesCurrent",), "instant"),
    "receivables": TagSpec(("AccountsReceivableNetCurrent",), "instant"),
    "payables": TagSpec(("AccountsPayableCurrent",), "instant"),
    "contract_liabilities": TagSpec(("ContractWithCustomerLiability", "DeferredRevenueCurrent"), "instant", note="deferred revenue"),
    "ppe_net": TagSpec(("PropertyPlantAndEquipmentNet",), "instant"),
    "rpo": TagSpec(("RevenueRemainingPerformanceObligation",), "instant", note="remaining performance obligations"),
    "rpo_pct_next_12m": TagSpec(("RevenueRemainingPerformanceObligationPercentage",), "instant", "pure",
                                "share of RPO to be recognised within 12 months, where disclosed non-dimensionally"),
}

SHARES_COVER = ("dei", "EntityCommonStockSharesOutstanding")


# --------------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------------
def download_companyfacts(cfg: Config, manifest: Manifest) -> dict:
    cik = str(cfg.project.cik).zfill(10)
    url = COMPANYFACTS_URL.format(cik=cik)
    client = PoliteClient(sec_headers(), cfg.data.sec_pause_seconds)
    data = client.get(url).json()
    out = cfg.path("data", "raw", "sec", create=True) / f"companyfacts_CIK{cik}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    manifest.record(
        "sec_companyfacts_raw",
        kind="HISTORICAL",
        source=f"SEC EDGAR XBRL companyfacts API ({url})",
        file=out,
        notes="Raw JSON as downloaded; every XBRL fact Oracle has filed (non-dimensional).",
    )
    return data


# --------------------------------------------------------------------------------------------
# period classification
# --------------------------------------------------------------------------------------------
def fiscal_label(end: pd.Timestamp, fye_month: int) -> tuple[int, int]:
    """Fiscal (year, quarter) for a period-end date. Oracle FYE=May -> Aug=Q1, Nov=Q2, Feb=Q3, May=Q4."""
    fy = end.year + (1 if end.month > fye_month else 0)
    q = ((end.month - fye_month - 1) % 12) // 3 + 1
    return fy, q


def _months(start: pd.Timestamp, end: pd.Timestamp) -> int | None:
    days = (end - start).days + 1
    m = round(days / 30.4375)
    return m if m in (3, 6, 9, 12) and abs(days - m * 30.4375) <= 7 else None


def _collect(gaap: dict, tags: tuple[str, ...], unit: str, taxonomy_key: str = "us-gaap") -> pd.DataFrame:
    """All first-filed observations for ``tags`` with a ``source_tag`` column (priority order kept)."""
    frames = []
    for prio, tag in enumerate(tags):
        node = gaap.get(tag)
        if not node or unit not in node["units"]:
            continue
        df = pd.DataFrame(node["units"][unit])
        df["source_tag"] = tag
        df["prio"] = prio
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    if "start" not in df:
        df["start"] = pd.NaT
    df["start"] = pd.to_datetime(df["start"])
    return df


def _first_filed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One observation per period: the earliest filing that reported it (tag priority breaks ties).

    Selecting the earliest filing *across* candidate tags keeps all line items of a period in the same
    reporting vintage (e.g. pre-ASC 606 revenue is never paired with pre-ASC 606 costs of a different vintage)
    and makes the data point-in-time safe.
    """
    df = df.sort_values(["filed", "prio"])
    return df.drop_duplicates(keys, keep="first")


# --------------------------------------------------------------------------------------------
# series builders
# --------------------------------------------------------------------------------------------
def _flow_quarters(df: pd.DataFrame, fye_month: int, kind: str) -> pd.DataFrame:
    """Return a frame indexed by quarter-end with ``value``, ``filed`` and ``derived`` columns.

    ``kind='flow'``: Q = direct 3-month fact, else YTD differencing (Q4 = FY - 9M).
    ``kind='avg'``: Q1-Q3 direct, Q4 = 4*FY - (Q1+Q2+Q3).
    """
    if df.empty:
        return pd.DataFrame(columns=["value", "filed", "derived"])
    df = df.dropna(subset=["start"]).copy()
    df["months"] = [_months(s, e) for s, e in zip(df["start"], df["end"])]
    df = df.dropna(subset=["months"])
    df["months"] = df["months"].astype(int)
    df = _first_filed(df, ["start", "end"])
    rows = []
    fy_ends = sorted({e for e in df["end"]})
    # group by fiscal year via the end date's fiscal label
    df["fy"] = [fiscal_label(e, fye_month)[0] for e in df["end"]]
    for fy, g in df.groupby("fy"):
        fy_start = pd.Timestamp(year=fy - 1, month=fye_month % 12 + 1, day=1)
        if fye_month == 12:
            fy_start = pd.Timestamp(year=fy, month=1, day=1)
        cum, direct = {}, {}
        for r in g.itertuples():
            if abs((r.start - fy_start).days) <= 6:
                cum[r.months] = (r.val, r.filed, r.end)
            if r.months == 3:
                qn = fiscal_label(r.end, fye_month)[1]
                if qn != 4:  # Q4 is NEVER taken directly: 10-K "Q4" contexts are unreliable (see module docstring)
                    direct[qn] = (r.val, r.filed, r.end)
        qvals: dict[int, tuple[float, pd.Timestamp, pd.Timestamp, bool]] = {}
        if kind == "flow":
            if 1 in direct: qvals[1] = (*direct[1], False)
            elif 3 in cum: qvals[1] = (*cum[3], False)
            for q, (hi, lo) in {2: (6, 3), 3: (9, 6), 4: (12, 9)}.items():
                diff = None
                if hi in cum and lo in cum:
                    diff = (cum[hi][0] - cum[lo][0], max(cum[hi][1], cum[lo][1]), cum[hi][2], True)
                if q in direct:
                    # prefer the directly reported 3-month value unless it disagrees materially with YTD differencing
                    if diff is not None and abs(direct[q][0] - diff[0]) > 0.02 * max(abs(direct[q][0]), 1e-9) and kind == "flow":
                        qvals[q] = diff
                    else:
                        qvals[q] = (*direct[q], False)
                elif diff is not None:
                    qvals[q] = diff
            if 4 not in qvals and 12 in cum and all(k in qvals for k in (1, 2, 3)):
                v = cum[12][0] - sum(qvals[k][0] for k in (1, 2, 3))
                qvals[4] = (v, cum[12][1], cum[12][2], True)
        else:  # avg
            for q in (1, 2, 3):
                if q in direct: qvals[q] = (*direct[q], False)
            if 12 in cum and all(k in qvals for k in (1, 2, 3)):
                v = 4 * cum[12][0] - sum(qvals[k][0] for k in (1, 2, 3))
                qvals[4] = (v, cum[12][1], cum[12][2], True)
        for q, (v, filed, end, derived) in qvals.items():
            rows.append({"period_end": end, "value": v, "filed": filed, "derived": derived})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["value", "filed", "derived"])
    out["period_end"] = out["period_end"] + pd.offsets.MonthEnd(0)
    return out.drop_duplicates("period_end", keep="first").set_index("period_end").sort_index()


def _annual(df: pd.DataFrame, fye_month: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["value", "filed"])
    df = df.dropna(subset=["start"]).copy()
    df["months"] = [_months(s, e) for s, e in zip(df["start"], df["end"])]
    df = df[df["months"] == 12]
    df = _first_filed(df, ["start", "end"])
    df["period_end"] = df["end"] + pd.offsets.MonthEnd(0)
    return df.set_index("period_end")[["val", "filed"]].rename(columns={"val": "value"}).sort_index()


def _instants(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["value", "filed"])
    df = _first_filed(df, ["end"])
    df["period_end"] = df["end"] + pd.offsets.MonthEnd(0)
    return df.set_index("period_end")[["val", "filed"]].rename(columns={"val": "value"}).sort_index()


def build_quarterly_annual(facts: dict, fye_month: int, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gaap = facts["facts"]["us-gaap"]
    q_cols, a_cols, filed_cols = {}, {}, {}
    for name, spec in TAG_MAP.items():
        raw = _collect(gaap, spec.tags, spec.unit)
        if spec.kind == "instant":
            s = _instants(raw)
            q_cols[name] = s["value"]
            a_cols[name] = s["value"][[d.month == fye_month for d in s.index]]
            filed_cols[name] = s["filed"]
        else:
            qs = _flow_quarters(raw, fye_month, spec.kind)
            q_cols[name] = qs["value"]
            filed_cols[name] = qs["filed"]
            if spec.kind == "flow":
                a_cols[name] = _annual(raw, fye_month)["value"]
            else:  # average-type: annual average from the FY fact
                a_cols[name] = _annual(raw, fye_month)["value"]
    q = pd.DataFrame(q_cols).sort_index()
    filed = pd.DataFrame(filed_cols).sort_index()
    # a quarter's availability is when its income-statement / cash-flow facts were first filed
    q["filed"] = filed[["revenue", "operating_income", "ocf"]].max(axis=1)
    q = q[q.index >= pd.Timestamp(start)]
    q = q.dropna(how="all", subset=[c for c in q.columns if c != "filed"])
    fl = [fiscal_label(d, fye_month) for d in q.index]
    q.insert(0, "fiscal_year", [x[0] for x in fl])
    q.insert(1, "fiscal_quarter", [x[1] for x in fl])
    q.insert(2, "label", [f"FY{str(x[0])[-2:]} Q{x[1]}" for x in fl])
    a = pd.DataFrame(a_cols).sort_index()
    a = a[a.index >= pd.Timestamp(start)]
    a.insert(0, "fiscal_year", [fiscal_label(d, fye_month)[0] for d in a.index])
    a.insert(1, "label", [f"FY{str(y)[-2:]}" for y in a["fiscal_year"]])
    return q, a


def shares_outstanding_cover(facts: dict, quarter_ends: pd.DatetimeIndex) -> pd.Series:
    """Cover-page share count (``dei``) mapped to the quarter it belongs to (latest quarter-end before the cover date)."""
    node = facts["facts"][SHARES_COVER[0]][SHARES_COVER[1]]["units"]["shares"]
    df = pd.DataFrame(node)
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df = df.sort_values("filed").drop_duplicates("end", keep="first")
    out = {}
    for r in df.itertuples():
        prior = quarter_ends[quarter_ends < r.end]
        if len(prior):
            out[prior[-1]] = r.val
    return pd.Series(out, name="shares_outstanding_cover").sort_index()


def run(cfg: Config, manifest: Manifest, use_cache: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cik = str(cfg.project.cik).zfill(10)
    cache = cfg.path("data", "raw", "sec", create=True) / f"companyfacts_CIK{cik}.json"
    if use_cache and cache.exists():
        facts = json.loads(cache.read_text(encoding="utf-8"))
    else:
        facts = download_companyfacts(cfg, manifest)
    q, a = build_quarterly_annual(facts, cfg.project.fiscal_year_end_month, cfg.data.fundamentals_start)
    q["shares_outstanding_cover"] = shares_outstanding_cover(facts, q.index)
    q_path = cfg.processed_dir / "fundamentals_quarterly_xbrl.csv"
    a_path = cfg.processed_dir / "fundamentals_annual_xbrl.csv"
    q.to_csv(q_path)
    a.to_csv(a_path)
    src = "SEC EDGAR XBRL companyfacts (us-gaap); quarters derived from 10-Q/10-K, first-filed values"
    manifest.record("fundamentals_quarterly_xbrl", kind="HISTORICAL", source=src, file=q_path, df=q,
                    notes="Q4 = FY - 9M YTD; share counts are averages (Q4 = 4*FY - sum(Q1..Q3)). 'filed' = first filing date.")
    manifest.record("fundamentals_annual_xbrl", kind="HISTORICAL", source=src, file=a_path, df=a)
    return q, a
