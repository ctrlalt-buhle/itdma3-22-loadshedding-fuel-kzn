"""
ITDMA3-22 Group 1 -- merge pipeline
Combines three cleaned data contributions into one monthly,
pipeline-ready dataset for OLS / Ridge / Lasso / Random Forest.

Inputs (place under data/raw/):
  kzn_cleaned/csv/kzn_monthly_historical_features_standardized.csv   (KZN set)
  itdpa_cleaned/clean_loadshedding_monthly.csv.xlsx              (ITDPA set)
  itdpa_cleaned/clean_fuel_monthly.csv.xlsx                      (ITDPA set)
  itdpa_cleaned/clean_macro_monthly.csv.xlsx                     (ITDPA set)
  controls_cleaned/Control_Variables_Prepared.xlsx                 (controls set)

Output:
  data/processed/group1_monthly_merged.csv
  data/processed/merge_log.txt

Key design decision (confirm with supervisor if this needs to change):
  PRIMARY outcome = PortWatch KZN trade-tonnes estimate (continuous,
  Jan 2022-Sep 2026). TNPA official cargo tonnage is kept as a SEPARATE
  validation column where it exists, never spliced into the primary DV
  (the KZN set's README is explicit on this point).
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
log_lines = []

MANIFEST = """\
========================================================================
DATA INPUTS MANIFEST — group1_monthly_merged.csv
========================================================================

USED IN THE MERGE (5 files, one row per source):

1. KZN set — kzn_monthly_historical_features_standardized.csv
   Frequency: monthly, Jan 2022-Sep 2026 (57 rows)
   Contributes: dv_portwatch_tonnes, portwatch_kzn_portcalls,
   dv_tnpa_tonnes_validation, tnpa_vessel_arrivals_validation,
   tnpa_teus_validation, petrol95_level, diesel_level
   Role: BACKBONE — the only source spanning the full study window

2. ITDPA set — clean_loadshedding_monthly.csv.xlsx
   Frequency: monthly, Apr 2022-Sep 2026 (54 rows)
   Contributes: mlr_avg_mw, mlr_max_mw, ls_hours, ils_avg_mw,
   residual_demand_mw, dispatch_gen_mw, ls_active, ls_share_of_hours
   Role: primary independent variable (loadshedding severity)

3. ITDPA set — clean_fuel_monthly.csv.xlsx
   Frequency: monthly, Jan-Aug 2026 only (8 rows)
   Contributes: petrol93_chg_cl_2026, diesel005_chg_cl_2026, and
   related cumulative/absolute change columns
   Role: supplementary — 2026 price MOVEMENT (not levels; see gap note)

4. ITDPA set — clean_macro_monthly.csv.xlsx
   Frequency: monthly, 2026 only (9 rows)
   Contributes: repo_rate, exports_rbn, imports_rbn, trade_balance_rbn
   Role: control variables (2026-only coverage)

5. ITDPA set — clean_macro_quarterly.csv.xlsx
   Frequency: quarterly, 2022Q1-2026Q3 (19 quarters, broadcast to months)
   Contributes: dv_bci (RMB/BER Business Confidence Index)
   Role: SECOND, broader outcome variable (cross-sector sentiment)

6. controls set — controls set_Control_Variables_Prepared.xlsx
   Sheets used: "CPI KZN Monthly" (monthly, continuous),
   "Quarterly Controls" (unemployment, USD/ZAR, repo/prime rate)
   Contributes: cpi_index, cpi_yoy_pct, unemployment_rate_pct,
   usdzar_avg, repo_rate_pct
   Role: control variables

DELIBERATELY NOT MERGED (cleaned, but redundant or unsuitable for a
monthly model — kept in data/raw for reference / write-up evidence):

- ITDPA set: clean_ports_monthly, other pre-merged summary workbooks — superseded by the KZN set's longer backbone,
  which already covers the same Jan-Aug 2026 port figures
- ITDPA set: clean_qes_sector.csv.xlsx — a single quarter, not a time
  series; cannot be modelled, use only as narrative evidence
- KZN set: kzn_quarterly_historical_features — a quarterly duplicate of
  the monthly backbone already in use
- KZN set: portwatch_daily_clean, portwatch_monthly_by_port,
  tnpa_kzn_historical_monthly, fuel_prices_historical_monthly —
  component files already aggregated into the backbone's KZN totals
- KZN set: kzn_gdp_annual, tnpa_portwatch_overlapping_months — annual
  (too coarse to merge monthly) / a validation diagnostic, not a
  model feature
- controls set: KZN_QLFS_Quarterly_Summary.xlsx — same unemployment figures
  already in "Quarterly Controls"; GDP KZN Annual, GDP Real Growth,
  USDZAR Daily, standalone "Interest Rates" sheet — annual or daily
  granularity not needed once the quarterly/monthly versions are in

========================================================================
MERGE RUN LOG
========================================================================
"""
log_lines.append(MANIFEST)


def log(msg):
    print(msg)
    log_lines.append(msg)


# ---------------------------------------------------------------------
# 1. Backbone: the KZN set's monthly historical features (Jan 2022-Sep 2026)
# ---------------------------------------------------------------------
backbone = pd.read_csv(RAW / "kzn_cleaned/csv/kzn_monthly_historical_features_standardized.csv")
backbone["month"] = pd.to_datetime(backbone["month"], format="%Y-%m")

backbone = backbone.rename(columns={
    "portwatch_kzn_trade_tonnes_estimate": "dv_portwatch_tonnes",
    "portwatch_kzn_portcalls": "portwatch_kzn_portcalls",
    "tnpa_kzn_cargo_tonnes_excl_vehicles": "dv_tnpa_tonnes_validation",
    "tnpa_kzn_vessel_arrivals_all_types": "tnpa_vessel_arrivals_validation",
    "tnpa_kzn_teu_invoiced": "tnpa_teus_validation",
    "petrol_95_coast_r_per_l": "petrol95_level",
    "diesel_005_coast_r_per_l": "diesel_level",
})

df = backbone[[
    "month", "year", "quarter",
    "dv_portwatch_tonnes", "portwatch_kzn_portcalls",
    "dv_tnpa_tonnes_validation", "tnpa_vessel_arrivals_validation", "tnpa_teus_validation",
    "petrol95_level", "diesel_level",
]].copy()

log(f"Backbone (KZN set, PortWatch+TNPA+fuel-to-Oct2025): {df.shape[0]} months, "
    f"{df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")

# ---------------------------------------------------------------------
# 2. Loadshedding severity (ITDPA set) -- covers Apr 2022-Sep 2026
# ---------------------------------------------------------------------
ls = pd.read_excel(RAW / "itdpa_cleaned/clean_loadshedding_monthly.csv.xlsx")
ls["month"] = pd.to_datetime(ls["month"])
df = df.merge(ls, on="month", how="left")
log(f"+ Loadshedding (ITDPA set): joined on month. "
    f"Missing before Apr 2022 (data starts there): "
    f"{df['mlr_avg_mw'].isna().sum()} months with no loadshedding reading.")

# ---------------------------------------------------------------------
# 3. Fuel price -- 2026 month-on-month changes (ITDPA set) as a
#    supplementary series; levels NOT chained to the KZN series because
#    the Nov-Dec 2025 anchor value is missing from both contributions.
#    This is a genuine open gap -- flagged, not fabricated.
# ---------------------------------------------------------------------
fuel_chg = pd.read_excel(RAW / "itdpa_cleaned/clean_fuel_monthly.csv.xlsx")
fuel_chg["month"] = pd.to_datetime(fuel_chg["month"])
fuel_chg = fuel_chg.rename(columns={c: f"{c}_2026" for c in fuel_chg.columns if c != "month"})
df = df.merge(fuel_chg, on="month", how="left")

gap_months = pd.date_range("2025-11-01", "2025-12-01", freq="MS")
df["fuel_level_gap_flag"] = df["month"].isin(gap_months).astype(int)
log("+ Fuel 2026 month-on-month changes (ITDPA set) added as separate columns "
    "(*_2026 suffix) rather than chained into petrol95_level/diesel_level: "
    "Nov-Dec 2025 price levels are missing from every contribution supplied so far "
    "-- OPEN ITEM, see merge_log.")

# ---------------------------------------------------------------------
# 4. Macro (ITDPA set, 2026 only: repo rate, trade balance)
# ---------------------------------------------------------------------
macro = pd.read_excel(RAW / "itdpa_cleaned/clean_macro_monthly.csv.xlsx")
macro["month"] = pd.to_datetime(macro["month"])
df = df.merge(macro, on="month", how="left", suffixes=("", "_nkaz"))

# ---------------------------------------------------------------------
# 5. Controls (controls set): CPI monthly (continuous) + quarterly
#    unemployment/USD-ZAR/repo-prime, broadcast across each quarter's months
# ---------------------------------------------------------------------
gw = pd.ExcelFile(RAW / "controls_cleaned/Control_Variables_Prepared.xlsx")

cpi = gw.parse("CPI KZN Monthly")
cpi["month"] = pd.to_datetime(cpi["Month"], unit="D", origin="1899-12-30")  # Excel serial date
cpi = cpi.rename(columns={
    "KZN CPI Index (Dec 2024=100)": "cpi_index",
    "KZN CPI YoY Inflation (%)": "cpi_yoy_pct",
})[["month", "cpi_index", "cpi_yoy_pct"]]
df = df.merge(cpi, on="month", how="left")

qc = gw.parse("Quarterly Controls").rename(columns={
    "Quarter": "quarter",
    "KZN Unemployment Rate (%)": "unemployment_rate_pct",
    "USD/ZAR Avg": "usdzar_avg",
    "Repo Rate Avg (%)": "repo_rate_pct",
})[["quarter", "unemployment_rate_pct", "usdzar_avg", "repo_rate_pct"]]
df = df.merge(qc, on="quarter", how="left")

log("+ CPI (monthly, continuous) and unemployment/USD-ZAR/repo rate "
    "(quarterly, broadcast to each month) added from the controls workbook. "
    "Repo rate is NaN before 2026Q1 -- it is only populated for 2026 in the "
    "supplied data (per the controls set's own README); do not treat pre-2026 repo "
    "rate as zero or impute it.")

# ---------------------------------------------------------------------
# 5b. Business Confidence Index (ITDPA set, quarterly) -- a SECOND,
#     broader outcome variable: RMB/BER BCI reflects general business
#     sentiment across sectors, not just port/logistics activity.
#     Continuous for all 19 quarters of the study window.
# ---------------------------------------------------------------------
macro_q = pd.read_excel(RAW / "itdpa_cleaned/clean_macro_quarterly.csv.xlsx")
macro_q["quarter"] = pd.PeriodIndex(pd.to_datetime(macro_q["quarter"]), freq="Q").astype(str)
macro_q["quarter"] = macro_q["quarter"].str.replace(r"(\d{4})Q(\d)", r"\1Q\2", regex=True)
df["quarter"] = pd.PeriodIndex(df["month"], freq="Q").astype(str)
df = df.merge(macro_q[["quarter", "bci"]].rename(columns={"bci": "dv_bci"}), on="quarter", how="left")
log(f"+ Business Confidence Index (ITDPA set, quarterly, broadcast to months): "
    f"dv_bci present for {df['dv_bci'].notna().sum()} of {len(df)} months -- "
    f"continuous across the full study window, unlike the port-based outcomes. "
    f"Use as a SECOND outcome variable so the research question covers business "
    f"performance broadly, not only port/logistics throughput.")

# ---------------------------------------------------------------------
# 6. Feature engineering (per the group's original plan)
# ---------------------------------------------------------------------
df = df.sort_values("month").reset_index(drop=True)
df["mlr_avg_mw_lag1"] = df["mlr_avg_mw"].shift(1)
df["mlr_x_petrol95"] = df["mlr_avg_mw"] * df["petrol95_level"]

# ---------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------
df.to_csv(OUT / "group1_monthly_merged.csv", index=False)
log(f"\nFinal merged dataset: {df.shape[0]} rows x {df.shape[1]} columns, "
    f"{df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
log(f"Saved to {OUT / 'group1_monthly_merged.csv'}")

with open(OUT / "merge_log.txt", "w") as f:
    f.write("\n".join(log_lines))
