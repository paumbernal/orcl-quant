# Manual KPI overrides

`kpi_overrides.csv` is intentionally **empty**. It exists so a user can correct or supply a KPI that the automated
parser could not extract (for example a quarter whose press release was formatted differently).

| column | meaning |
|---|---|
| `period_end` | fiscal quarter end, e.g. `2025-08-31` |
| `metric` | column name in `press_release_kpis.csv`, e.g. `pr_iaas_rev_bn` |
| `value` | the number (same unit as the column) |
| `source_url` | **required** - the filing / transcript the number comes from (must start with `http`) |
| `retrieved` | date you read it |

Rows without a `source_url` make the pipeline fail on purpose: no unsourced numbers enter the analysis.
Overridden cells are flagged in the `override_applied` column of the output.
