"""
Brazil exporter: ComexStat long xlsx -> Processed/by_source/brazil_exporter.csv

"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = (ROOT / "Data" / "Export" / "Brazil"
       / "V_EXPORT_GENERAL_2016-01_2026-12_DT20261009.xlsx")
LOOKUP = ROOT / "Lookups" / "countries.csv"
OUT = ROOT / "Processed" / "by_source" / "brazil_exporter.csv"

# First order of business - read the data sheet and check the row count
data = pd.read_excel(RAW, sheet_name="Result")
details = pd.read_excel(RAW, sheet_name="Details", dtype=str)
expected = int(details["Row number"].iloc[0])
assert len(data) == expected, f"read {len(data)} rows, expected {expected}"
assert (data["SH4 Code"] == 2601).all(), "unexpected product code"

# Country names -> three letter codes, only through the the lookup file
lookup = pd.read_csv(LOOKUP, dtype=str).dropna(subset=["comex_brazil_name"])
codes = lookup.set_index(lookup["comex_brazil_name"].str.strip())["iso3"]
names = data["Country"].str.strip()
unknown = sorted(set(names) - set(codes.index))
assert not unknown, f"names missing from countries.csv: {unknown}"

# Month comes as text like '06. June' so we just grab the first two characters
month_number = data["Month"].str[:2].astype(int)
month = pd.to_datetime({"year": data["Year"], "month": month_number,
                        "day": 1})

# The download date is stamped in the file name (DT20261009 -> 2026-10-09)
stamp = re.search(r"DT(\d{4})(\d{2})(\d{2})", RAW.name)
retrieved = "-".join(stamp.groups())

out = pd.DataFrame(
    {
        "source": "brazil_exporter",
        "exporter": "BRA",
        "counterparty": names.map(codes),
        "port": "",
        "month": month,
        "tonnes": data["Net Weight"] / 1000, # Net Weight is kilograms
        "flag": "",
        "retrieved": retrieved,
        "note": "",
    }
)
out = out.sort_values(["month", "counterparty"]).reset_index(drop=True)

# No month should be missing between the first and the last
months = out["month"].drop_duplicates()
every_month = pd.date_range(months.min(), months.max(), freq="MS")
assert len(months) == len(every_month), "a month is missing"

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
print(f"wrote {len(out)} rows to {OUT}")