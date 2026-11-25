"""
Peru exporter: BCRP workbooks -> Processed/by_source/peru_exporter.csv

"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
RAW_DIR = ROOT / "Data" / "Export" / "Peru"
OUT = ROOT / "Processed" / "by_source" / "peru_exporter.csv"

RETRIEVED = "2026-10-09"
FIRST_MONTH, LAST_MONTH, N_MONTHS = "2012-01-01", "2026-08-01", 176
MONTHS = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Ago": 8, "Sep": 9, "Set": 9, "Oct": 10, "Nov": 11,
          "Dic": 12}  # both Sep and Set mean September

# Months where the source shows zero but other records show cargo
MISSING_NOTE = {
    "2020-05-01": "source shows 0 in volume and value; PortWatch San Nicolas "
                  "shows 0.53 Mt loaded and China records 0.97 Mt arriving; "
                  "treated as missing",
}
PROVISIONAL_NOTE = ("provisional (the series was last updated 2026-08-13 and "
                    "ends December 2025 in its metadata; 2026 values can be "
                    "revised)")

# Let's read both workbooks. Rows 1 and 2 have the series code and title,
# then it is one row per month
workbooks = [("by volume", "PN38789BM", "millones de toneladas"),
             ("by value", "PN38788BM", "millones US$")]
series = {}
for folder, code, unit_words in workbooks:
    path = next((RAW_DIR / folder).glob("*.xlsx"))
    sheet = pd.read_excel(path, sheet_name="Mensuales", header = None,
                          engine="openpyxl")
    assert sheet.iloc[0, 1] == code, f"{path.name}: wrong series code"
    assert unit_words in sheet.iloc[1, 1], f"{path.name}: title changed"
    data = sheet.iloc[2:, :2].copy()
    data.columns = ["label", "value"]
    assert len(data) == N_MONTHS, f"{path.name}: read {len(data)} months"
    assert data["value"].notna().all(), f"{path.name}: a blank month"

    # Labels look like 'Ene12', so month name then two digit year
    labels = data["label"].astype(str).str.strip()
    month_number = labels.str[:3].map(MONTHS)
    year_number = labels.str[3:]
    assert (labels.str.len() == 5).all(), "month labels are not 5 long"
    assert month_number.notna().all(), "unknown month name"
    data["month"] = pd.to_datetime({"year": 2000 + year_number.astype(int),
                                    "month": month_number, "day": 1})
    data["value"] = data["value"].astype(float)
    series[folder] = data.reset_index(drop=True)

volume, value = series["by volume"], series["by value"]

# Same months in both files, and none missing or extra
every_month = pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS")
assert list(volume["month"]) == list(every_month), "a month is missing"
assert list(volume["label"]) == list(value["label"]), "months differ"

# The value file is only here for these checks - same zero months in both,
# and a believable price per tonne
key = volume["month"].dt.strftime("%Y-%m-%d")
zero_volume = set(key[volume["value"] == 0])
zero_value = set(key[value["value"] == 0])
different = zero_volume ^ zero_value
assert not different, f"zero months differ: {sorted(different)}"
unexpected = zero_volume - set(MISSING_NOTE)
assert not unexpected, f"a zero month nobody looked at: {sorted(unexpected)}"

paid = volume["value"] > 0
price = value.loc[paid, "value"] / volume.loc[paid, "value"]  # US$ per tonne
low, high = price.min(), price.max()
assert 25 <= low and high <= 280, f"price is {low:.0f} to {high:.0f}"

missing = key.isin(MISSING_NOTE)
out = pd.DataFrame(
    {
        "source": "peru_exporter",
        "exporter": "PER",
        "counterparty": "ALL",
        "port": "",
        "month": volume["month"],
        "tonnes": volume["value"] * 1_000_000,  # millions of tonnes
        "flag": np.where(missing, "missing", ""),
        "retrieved": RETRIEVED,
        "note": key.map(MISSING_NOTE).fillna(""),
    }
)
out.loc[missing, "tonnes"] = np.nan
out.loc[out["month"].dt.year == 2026, "note"] = PROVISIONAL_NOTE

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
have_tonnes = int(out["tonnes"].notna().sum())
print(f"wrote {len(out)} rows to {OUT} ({have_tonnes} with tonnes, "
      f"{int(missing.sum())} missing)")

# Total per year in millions of tonnes
yearly = out.groupby(out["month"].dt.year)["tonnes"].sum() / 1e6
print(yearly.round(2).to_string())