"""Earnings events and press-release KPIs from SEC EDGAR (8-K Item 2.02) + Yahoo consensus surprises.

Sources
-------
* **EDGAR submissions API** - every 8-K with Item 2.02 ("Results of Operations"). The acceptance
  timestamp gives the exact release time (Oracle publishes after the US close), so the market
  reaction day is identified from data rather than assumed.
* **Exhibit 99.1 (the press release)** - parsed with regular expressions for the KPIs that are not in
  XBRL: total cloud revenue, cloud infrastructure (IaaS / "OCI") revenue, cloud applications (SaaS)
  revenue (levels in $bn rounded to 0.1 and USD growth rates) and forward guidance when stated.
  Anything that cannot be parsed is left as NaN - values are never inferred or filled in.
* **Yahoo Finance** - consensus EPS estimate / reported (non-GAAP) EPS / surprise %.

A user-maintained ``data/manual/kpi_overrides.csv`` (columns: period_end, metric, value, source_url,
retrieved) can supply or correct a KPI; overrides are applied last and flagged in the output.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from ..config import Config
from .http import PoliteClient, sec_headers
from .manifest import Manifest

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}"


# --------------------------------------------------------------------------------------------
# 1. earnings 8-K list
# --------------------------------------------------------------------------------------------
def _quarter_ends(start: str, end: str, fye_month: int) -> list[pd.Timestamp]:
    months = {(fye_month + 3 * k - 1) % 12 + 1 for k in range(1, 5)}
    rng = pd.date_range(start, end, freq="ME")
    return [d for d in rng if d.month in months]


def list_earnings_filings(cfg: Config, client: PoliteClient) -> pd.DataFrame:
    cik = str(cfg.project.cik).zfill(10)
    main = client.get(SUBMISSIONS.format(name=f"CIK{cik}.json")).json()
    frames = [pd.DataFrame(main["filings"]["recent"])]
    for f in main["filings"].get("files", []):
        frames.append(pd.DataFrame(client.get(SUBMISSIONS.format(name=f["name"])).json()))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["form"] == "8-K") & df["items"].fillna("").str.contains(r"\b2\.02\b")].copy()
    df["filing_date"] = pd.to_datetime(df["filingDate"])
    df["accepted_utc"] = pd.to_datetime(df["acceptanceDateTime"], utc=True)
    df["accepted_et"] = df["accepted_utc"].dt.tz_convert(NY)

    # Attach each filing to the fiscal quarter it reports: the most recent quarter end that is 5-60 days old.
    qe = _quarter_ends("2009-01-01", str(date.today()), cfg.project.fiscal_year_end_month)
    def attach(d: pd.Timestamp):
        cands = [q for q in qe if 5 <= (d - q).days <= 60]
        return max(cands) if cands else pd.NaT
    df["period_end"] = df["filing_date"].map(attach)
    df = df.dropna(subset=["period_end"]).sort_values("filing_date").drop_duplicates("period_end", keep="first")
    keep = ["period_end", "accessionNumber", "filing_date", "accepted_et", "primaryDocument"]
    return df[keep].rename(columns={"accessionNumber": "accession"}).reset_index(drop=True)


def reaction_date(accepted_et: pd.Timestamp, trading_days: pd.DatetimeIndex, after_close_hour: int = 16) -> pd.Timestamp:
    """First trading session whose close reflects the release."""
    d = pd.Timestamp(accepted_et.date())
    if accepted_et.hour >= after_close_hour:
        nxt = trading_days[trading_days > d]
    else:
        nxt = trading_days[trading_days >= d]
    return nxt[0] if len(nxt) else pd.NaT


# --------------------------------------------------------------------------------------------
# 2. press releases
# --------------------------------------------------------------------------------------------
def _find_exhibit_name(index_json: dict) -> str | None:
    items = [i["name"] for i in index_json["directory"]["item"] if i["name"].lower().endswith((".htm", ".html"))]
    ex = [n for n in items if re.search(r"ex-?99", n, re.I)]
    return ex[0] if ex else None


def fetch_press_release(cfg: Config, client: PoliteClient, accession: str) -> str | None:
    cache = cfg.path("data", "raw", "press_releases", create=True) / f"{accession}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    base = ARCHIVE.format(cik=int(cfg.project.cik), acc=accession.replace("-", ""))
    try:
        idx = client.get(base + "/index.json").json()
        name = _find_exhibit_name(idx)
        if not name:
            return None
        html = client.get(f"{base}/{name}").content
    except Exception as exc:  # network / 404 - leave the KPI missing rather than guess
        log.warning("press release %s unavailable: %s", accession, exc)
        return None
    text = re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" ")).strip()
    cache.write_text(text, encoding="utf-8")
    return text


_CUT = re.compile("(?:\ufffd|\u2022|" + r"\s[o]\s(?=Q[1-4]\b)|\sQ[1-4]\s)")
_AMOUNT = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million)", re.I)
_GROWTH = re.compile(r"\b(up|down|increased|decreased|grew|declined)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s?%", re.I)


def _segment_after(text: str, label_re: str, window: int = 190) -> str | None:
    m = re.search(label_re, text, re.I)
    if not m:
        return None
    seg = text[m.end(): m.end() + window]
    c = _CUT.search(seg)
    return seg[: c.start()] if c else seg


def _amount_growth(text: str, label_re: str) -> tuple[float, float]:
    """Headline parse. Accepted ONLY when a $ amount is present: a bare '... Cloud Revenue up 32%' in older
    releases refers to a *different* KPI (e.g. Fusion ERP), so growth-without-level is rejected."""
    seg = _segment_after(text, label_re)
    if seg is None:
        return (np.nan, np.nan)
    a = _AMOUNT.search(seg)
    if not a:
        return (np.nan, np.nan)
    g = _GROWTH.search(seg)
    amt = float(a.group(1).replace(",", "")) * (1.0 if a.group(2).lower() == "billion" else 1e-3)
    gr = np.nan
    if g:
        gr = float(g.group(2)) * (1 if g.group(1).lower() in ("up", "increased", "grew") else -1)
    return (amt, gr)


HEADLINE_LABELS = {
    "cloud": r"(?:Total\s+)?Cloud\s+Revenues?\s*(?:\(IaaS\s*(?:plus|\+)\s*SaaS\))?",
    "iaas": r"Cloud\s+Infra(?:structure)?\s*(?:Revenues?\s*)?\(IaaS\)\s*(?:Revenues?)?",
    "saas": r"Cloud\s+App(?:lication)?s?\s*(?:Revenues?\s*)?\(SaaS\)\s*(?:Revenues?)?",
}
TABLE_LABELS = {"saas": "Cloud applications", "iaas": "Cloud infrastructure", "cloud": "Total cloud revenues"}
_NUM_RUN = r"((?:\$?\s*\d{1,3}(?:,\d{3})*\s+)+)"


def _table_row(text: str, label: str, k: int) -> tuple[float, float] | None:
    """Exact ($m) current-quarter and prior-year-quarter values from Oracle's supplemental revenue table.

    Row layout: prior-FY Q1..Q4, prior-FY total, current-FY Q1..Qk [, YTD]. Accepted only if the four prior-year
    quarters sum to the prior-FY total (guards against mis-aligned columns).
    """
    for m in re.finditer(re.escape(label) + r"\s+" + _NUM_RUN, text):
        nums = [int(x.replace(",", "")) for x in re.findall(r"\d{1,3}(?:,\d{3})*", m.group(1))]
        if len(nums) < 5 + k:
            continue
        if abs(sum(nums[:4]) - nums[4]) <= 3:
            return float(nums[4 + k]), float(nums[k - 1])
    return None


def parse_press_release(text: str, k: int) -> dict[str, float]:
    """KPIs from one release. ``k`` = fiscal quarter number (1-4) of the period being reported."""
    out: dict[str, float] = {}
    for key in ("cloud", "iaas", "saas"):
        row = _table_row(text, TABLE_LABELS[key], k)
        if row:
            cur, prior = row
            out[f"pr_{key}_rev_bn"] = cur / 1000.0
            out[f"pr_{key}_prior_year_bn"] = prior / 1000.0
            out[f"pr_{key}_growth_pct"] = (cur / prior - 1.0) * 100.0
            out[f"pr_{key}_source"] = "table"
        else:
            amt, gr = _amount_growth(text, HEADLINE_LABELS[key])
            out[f"pr_{key}_rev_bn"], out[f"pr_{key}_growth_pct"] = amt, gr
            out[f"pr_{key}_prior_year_bn"] = np.nan
            out[f"pr_{key}_source"] = "headline" if not np.isnan(amt) else ""
    c, i, s_ = (out["pr_cloud_rev_bn"], out["pr_iaas_rev_bn"], out["pr_saas_rev_bn"])
    out["pr_parse_consistent"] = float(abs(c - (i + s_)) <= 0.21) if not np.isnan([c, i, s_]).any() else np.nan
    out.update(parse_guidance(text))
    return out


def _usd_pair(seg: str, money: bool = False) -> tuple[float, float] | tuple[None, None]:
    """Pick the (lo, hi) range the sentence states in USD (falls back to the first range)."""
    num = r"\$([\d.]+)" if money else r"(\d+)%"
    pairs = list(re.finditer(num + r"\s*(?:and|to|-)\s*" + num, seg))
    if not pairs:
        return (None, None)
    pick = pairs[0]
    for pm in pairs:
        if "USD" in seg[pm.end(): pm.end() + 30]:
            pick = pm
    return float(pick.group(1)), float(pick.group(2))


def parse_guidance(text: str) -> dict[str, float]:
    """Next-quarter guidance in USD as stated in the release; NaN where the release has none."""
    g = {k: np.nan for k in ("guid_rev_growth_lo", "guid_rev_growth_hi", "guid_cloud_growth_lo", "guid_cloud_growth_hi",
                             "guid_eps_nongaap_lo", "guid_eps_nongaap_hi")}
    i = text.find("Guidance for Q")
    if i < 0:
        i = text.find("forward-looking guidance")
    if i < 0:
        return g
    seg = text[i: i + 3000]

    def sentence(start_re: str) -> str | None:
        m = re.search(start_re, seg, re.I)
        if not m:
            return None
        rest = seg[m.start(): m.start() + 420]
        c = re.search(r"\ufffd|\u2022|\sNon-GAAP earnings", rest[20:], re.I)
        return rest[: 20 + c.start()] if c else rest

    s1 = sentence(r"Total revenues? (?:are|is) expected to grow")
    if s1:
        lo, hi = _usd_pair(s1)
        if lo is not None:
            g["guid_rev_growth_lo"], g["guid_rev_growth_hi"] = lo, hi
    s2 = sentence(r"Total Cloud revenues? (?:are|is) expected to grow")
    if s2:
        lo, hi = _usd_pair(s2)
        if lo is not None:
            g["guid_cloud_growth_lo"], g["guid_cloud_growth_hi"] = lo, hi
    m = re.search(r"Non-GAAP earnings per share (?:is|are) expected to", seg, re.I)
    if m:
        lo, hi = _usd_pair(seg[m.start(): m.start() + 330], money=True)
        if lo is not None:
            g["guid_eps_nongaap_lo"], g["guid_eps_nongaap_hi"] = lo, hi
    return g


# --------------------------------------------------------------------------------------------
# 3. Yahoo consensus surprises
# --------------------------------------------------------------------------------------------
def yahoo_eps_history(ticker: str, limit: int = 100) -> pd.DataFrame:
    import yfinance as yf

    ed = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if ed is None or ed.empty:
        return pd.DataFrame()
    ed = ed.rename(columns={"EPS Estimate": "eps_estimate", "Reported EPS": "eps_actual", "Surprise(%)": "eps_surprise_pct"})
    ed["yahoo_date"] = [pd.Timestamp(i).tz_convert(NY).tz_localize(None).normalize() for i in ed.index]
    return ed.reset_index(drop=True)[["yahoo_date", "eps_estimate", "eps_actual", "eps_surprise_pct"]]


def _match_yahoo(release: pd.Timestamp, yh: pd.DataFrame, tol_days: int = 3) -> pd.Series:
    if yh.empty:
        return pd.Series(dtype=float)
    d = (yh["yahoo_date"] - release.normalize()).abs()
    j = d.idxmin()
    return yh.loc[j, ["eps_estimate", "eps_actual", "eps_surprise_pct"]] if d[j] <= pd.Timedelta(days=tol_days) else pd.Series(dtype=float)


# --------------------------------------------------------------------------------------------
# 4. orchestration
# --------------------------------------------------------------------------------------------
def apply_overrides(kpis: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    path = cfg.manual_dir / "kpi_overrides.csv"
    kpis["override_applied"] = ""
    if not path.exists():
        return kpis
    ov = pd.read_csv(path, parse_dates=["period_end"])
    ov = ov.dropna(subset=["period_end", "metric", "value"])
    for r in ov.itertuples():
        if not str(getattr(r, "source_url", "")).startswith("http"):
            raise ValueError("kpi_overrides.csv rows must cite a source_url")
        if r.period_end in kpis.index and r.metric in kpis.columns:
            kpis.loc[r.period_end, r.metric] = r.value
            kpis.loc[r.period_end, "override_applied"] += f"{r.metric};"
    return kpis


def run(cfg: Config, manifest: Manifest, use_cache: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = PoliteClient(sec_headers(), cfg.data.sec_pause_seconds)
    filings = list_earnings_filings(cfg, client)
    prices = pd.read_csv(cfg.processed_dir / "prices_adj_close.csv", index_col=0, parse_dates=True)
    td = prices.index
    filings["release_date"] = filings["accepted_et"].map(lambda t: pd.Timestamp(t.date()))
    filings["release_time_et"] = filings["accepted_et"].map(lambda t: t.strftime("%H:%M"))
    filings["reaction_date"] = [reaction_date(t, td) for t in filings["accepted_et"]]
    if not cfg.event_study.reports_after_close:
        filings["reaction_date"] = [pd.Timestamp(t.date()) for t in filings["accepted_et"]]

    # --- press-release KPIs (only the era in which Oracle reports them) ---
    rows = []
    for r in filings.itertuples():
        if r.filing_date < pd.Timestamp(cfg.data.press_release_start):
            continue
        text = fetch_press_release(cfg, client, r.accession)
        if text is None:
            continue
        k = ((r.period_end.month - cfg.project.fiscal_year_end_month - 1) % 12) // 3 + 1
        rec = {"period_end": r.period_end, "accession": r.accession, **parse_press_release(text, k)}
        rows.append(rec)
    kpis = pd.DataFrame(rows).set_index("period_end").sort_index() if rows else pd.DataFrame(
        columns=["accession", "pr_cloud_rev_bn", "pr_cloud_growth_pct", "pr_iaas_rev_bn", "pr_iaas_growth_pct",
                 "pr_saas_rev_bn", "pr_saas_growth_pct", "pr_parse_consistent"])
    kpis.index.name = "period_end"
    kpis = apply_overrides(kpis, cfg)
    k_path = cfg.processed_dir / "press_release_kpis.csv"
    kpis.to_csv(k_path)

    # --- events table ---
    yh = yahoo_eps_history(cfg.project.ticker)
    ev = filings.copy()
    sur = [_match_yahoo(t, yh) for t in ev["release_date"]]
    ev = pd.concat([ev.reset_index(drop=True), pd.DataFrame(sur).reset_index(drop=True)], axis=1)
    ev = ev.drop(columns=["accepted_et", "primaryDocument"]).set_index("period_end")
    ev = ev.join(kpis.drop(columns=["accession"], errors="ignore"), how="left")
    e_path = cfg.processed_dir / "earnings_events.csv"
    ev.to_csv(e_path)

    manifest.record("press_release_kpis", kind="HISTORICAL", file=k_path, df=kpis,
                    source="Oracle earnings press releases (SEC EDGAR 8-K Exhibit 99.1), parsed with regex",
                    notes="Cloud / IaaS / SaaS revenue in $bn (rounded to 0.1 by Oracle) and USD growth %. Unparsed items are NaN. "
                          "Next-quarter guidance parsed only where the release states it.")
    manifest.record("earnings_events", kind="HISTORICAL", file=e_path, df=ev,
                    source="SEC EDGAR 8-K Item 2.02 acceptance timestamps + Yahoo Finance consensus EPS surprises",
                    notes="reaction_date = first session whose close reflects the release (Oracle reports after 16:00 ET). "
                          "eps_actual is the non-GAAP EPS as reported by Yahoo; revenue surprise is not available from free sources.")
    return ev, kpis
