"""
Liberia exporter: CBL Monthly Economic Reviews -> liberia_exporter.csv

Reads liberia_iron_ore_monthly.csv, which has the figure most issues agree on
for each month (Jan 2019 to Aug 2026). The file comes from
extract_cbl_iron_ore.py and is described in README_liberia.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
RAW_DIR = ROOT / "Data" / "Export" / "Liberia"
OUT = ROOT / "Processed" / "by_source" / "liberia_exporter.csv"

RETRIEVED = "2026-11-09"
MEASURE = "export_volume_tonnes"
FIRST_MONTH, LAST_MONTH, N_MONTHS = "2019-01-01", "2026-08-01", 92

KNOWN_SOURCE_ERROR = {
    "2025-12-01": ("bank's volume 4,977,124 t against US$78.16m is US$15.7 "
                   "per tonne (other months 45-60); PortWatch Buchanan shows "
                   "1.0 Mt and EU customs 0.67 Mt; treated as missing"),
}

# Let's read the the agreed figures, one row per month for the volume measure
agreed = pd.read_csv(RAW_DIR / "liberia_iron_ore_monthly.csv",
                     dtype={"month": str})
agreed = agreed[agreed["measure"] == MEASURE].set_index("month")
assert agreed.index.is_unique, "a month repeats in the agreed figures"

months = pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS")
assert len(months) == N_MONTHS, "wrong number of months in the window"
# every agreed month must sit inside the window
outside = sorted(set(agreed.index) - set(months.strftime("%Y-%m")))
assert not outside, f"agreed months outside the window: {outside}"

rows = []
for month in months:
    key, day = month.strftime("%Y-%m"), month.strftime("%Y-%m-%d")
    tonnes, flag, note = np.nan, "", ""
    if key not in agreed.index:
        flag, note = "missing", "no issue we hold prints this month"
    else:
        row = agreed.loc[key]
        value = float(row["agreed_value"])
        prints = row["all_prints"]
        if row["different_figures"] > 1:
            note = ("issues print different figures, we use the one most "
                    f"issues agree on (issue:tonnes) {prints}")
        if day in KNOWN_SOURCE_ERROR:
            flag, note = "known_source_error", KNOWN_SOURCE_ERROR[day]
        else:
            assert value > 0, f"{key}: zero or negative volume {value}"
            tonnes = value
            if row["slip_in_agreed"]:
                flag = "unit_slip_corrected"
                note = ("a printed figure is in thousands of tonnes and was "
                        f"multiplied by 1000 (issue:tonnes) {prints}")
            elif row["issues_printing"] == 1:
                flag = "single_source"
                note = (f"printed in one issue only ({prints[:7]}); "
                        "no second copy to compare")
    rows.append({"source": "liberia_exporter", "exporter": "LBR",
                 "counterparty": "ALL", "port": "", "month": month,
                 "tonnes": tonnes, "flag": flag, "retrieved": RETRIEVED,
                 "note": note})
out = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
n_tonnes = int(out["tonnes"].notna().sum())
n_missing = int((out["flag"] == "missing").sum())
n_error = int((out["flag"] == "known_source_error").sum())
n_single = int((out["flag"] == "single_source").sum())
n_slip = int((out["flag"] == "unit_slip_corrected").sum())
print(f"wrote {len(out)} rows to {OUT} ({n_tonnes} with tonnes; "
      f"{n_missing} missing, {n_error} known source error, "
      f"{n_single} single source, {n_slip} unit slip corrected)")

# Total per year in millions of tonnes
yearly = out.groupby(out["month"].dt.year)["tonnes"].sum() / 1e6
print(yearly.round(2).to_string())
