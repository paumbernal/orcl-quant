"""SEC period derivation and earnings-release parsing (synthetic inputs with known answers)."""
import numpy as np
import pandas as pd
import pytest

from orcl_lab.data import earnings as E
from orcl_lab.data import sec_facts as S


def _facts(rows):
    df = pd.DataFrame(rows, columns=["start", "end", "val", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["prio"] = 0
    return df


def test_fiscal_label_for_may_year_end():
    f = lambda d: S.fiscal_label(pd.Timestamp(d), 5)
    assert f("2025-08-31") == (2026, 1)
    assert f("2025-11-30") == (2026, 2)
    assert f("2026-02-28") == (2026, 3)
    assert f("2026-05-31") == (2026, 4)


def test_q4_derived_as_fy_minus_nine_months_and_mistagged_q4_ignored():
    rows = [
        ("2025-06-01", "2025-08-31", 100, "2025-09-10"),    # Q1 (3m, = YTD3)
        ("2025-06-01", "2025-11-30", 220, "2025-12-10"),    # 6m YTD
        ("2025-09-01", "2025-11-30", 120, "2025-12-10"),    # Q2 direct
        ("2025-06-01", "2026-02-28", 360, "2026-03-10"),    # 9m YTD
        ("2025-12-01", "2026-02-28", 140, "2026-03-10"),    # Q3 direct
        ("2025-06-01", "2026-05-31", 520, "2026-06-20"),    # FY
        ("2026-03-01", "2026-05-31", 520, "2026-06-20"),    # Oracle-style mis-tag: full-year value with a 3-month Q4 context
    ]
    q = S._flow_quarters(_facts(rows), 5, "flow")
    assert list(q["value"]) == [100, 120, 140, 160]         # Q4 = 520 - 360, NOT the mis-tagged 520
    assert bool(q["derived"].iloc[-1]) is True


def test_average_type_q4():
    rows = [("2025-06-01", "2025-08-31", 100, "2025-09-10"), ("2025-09-01", "2025-11-30", 110, "2025-12-10"),
            ("2025-12-01", "2026-02-28", 120, "2026-03-10"), ("2025-06-01", "2026-05-31", 115, "2026-06-20")]
    q = S._flow_quarters(_facts(rows), 5, "avg")
    assert q["value"].iloc[-1] == pytest.approx(4 * 115 - (100 + 110 + 120))


def test_first_filed_value_kept_for_point_in_time():
    rows = [("2025-06-01", "2025-08-31", 100, "2025-09-10"), ("2025-06-01", "2025-08-31", 105, "2026-09-10")]   # later restatement
    df = S._first_filed(_facts(rows), ["start", "end"])
    assert len(df) == 1 and df["val"].iloc[0] == 100


RELEASE_TABLE = ("Cloud applications 3,469 3,503 3,580 3,700 14,252 3,839 3,898 4,026 11,763 "
                 "Cloud infrastructure 2,154 2,434 2,652 2,995 10,234 3,347 4,079 4,888 12,314 "
                 "Total cloud revenues $ 5,623 $ 5,937 $ 6,232 $ 6,695 $ 24,486 $ 7,186 $ 7,977 $ 8,914 $ 24,077 AS REPORTED")


def test_cloud_table_parser_extracts_current_and_prior_quarter():
    cur, prior = E._table_row(RELEASE_TABLE, "Cloud infrastructure", 3)
    assert (cur, prior) == (4888.0, 2652.0)
    out = E.parse_press_release(RELEASE_TABLE, 3)
    assert out["pr_iaas_rev_bn"] == pytest.approx(4.888)
    assert out["pr_iaas_growth_pct"] == pytest.approx((4888 / 2652 - 1) * 100)
    assert out["pr_iaas_source"] == "table"


def test_table_parser_rejects_misaligned_columns():
    bad = "Cloud infrastructure 2,154 2,434 2,652 2,995 99,999 3,347 4,079 4,888"        # quarters do not sum to the FY total
    assert E._table_row(bad, "Cloud infrastructure", 3) is None


def test_headline_parse_requires_dollar_amount():
    with_amount = "Q1 Cloud Revenue (IaaS plus SaaS) $7.2 billion, up 28% in USD and up 27% in constant currency Q1 Cloud"
    assert E._amount_growth(with_amount, E.HEADLINE_LABELS["cloud"]) == (7.2, 28.0)
    wrong_kpi = "Fusion ERP and HCM Cloud Revenue Up 32%, NetSuite ERP Cloud Revenue Up 32% REDWOOD SHORES"
    amt, gr = E._amount_growth(wrong_kpi, E.HEADLINE_LABELS["cloud"])
    assert np.isnan(amt) and np.isnan(gr)


def test_guidance_prefers_usd_range():
    txt = ("Guidance for Q2 FY 2027 Oracle is providing the following guidance: "
           "Total revenues are expected to grow between 30% and 34% in constant currency and USD. "
           "Total Cloud revenue is expected to grow between 64% and 70% in constant currency and between 65% and 71% in USD. "
           "Non-GAAP earnings per share is expected to be between $1.83 and $1.91 in constant currency and between $1.85 and $1.93 in USD, which")
    g = E.parse_guidance(txt)
    assert (g["guid_rev_growth_lo"], g["guid_rev_growth_hi"]) == (30.0, 34.0)
    assert (g["guid_cloud_growth_lo"], g["guid_cloud_growth_hi"]) == (65.0, 71.0)
    assert (g["guid_eps_nongaap_lo"], g["guid_eps_nongaap_hi"]) == (1.85, 1.93)


def test_after_close_release_reacts_next_session():
    days = pd.bdate_range("2026-09-01", "2026-09-30")
    et = pd.Timestamp("2026-09-10 16:12", tz="America/New_York")
    assert E.reaction_date(et, days) == pd.Timestamp("2026-09-11")
    morning = pd.Timestamp("2026-09-10 08:00", tz="America/New_York")
    assert E.reaction_date(morning, days) == pd.Timestamp("2026-09-10")
