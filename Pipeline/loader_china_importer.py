"""
China importer: Comtrade long csv -> Processed/by_source/china_importer.csv

"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
RAW = (ROOT / "Data" / "Import" / "China"
       / "china_imports_iron_ore_all_partners_long.csv")
LOOKUP = ROOT / "Lookups" / "countries.csv"
OUT = ROOT / "Processed" / "by_source" / "china_importer.csv"

RETRIEVED = "2026-10-09"  # download date, from README_china.txt
PRODUCT_LINES = ["260111", "260112"]  # fines and lump, pellets

raw = pd.read_csv(RAW, dtype={"period": str, "partnerCode": str,
                              "cmdCode": str})

# Which partners are our exporters? Comtrade code -> our three letter code
lookup = pd.read_csv(LOOKUP, dtype=str).fillna("")
ours = lookup[lookup["project_role"].str.startswith("exporter")]
codes = dict(zip(ours["comtrade_code"], ours["iso3"]))

# Just our exporters, and just the two product lines
lines = raw[raw["cmdCode"].isin(PRODUCT_LINES)
            & raw["partnerCode"].isin(codes)].copy()

# A few lines have no weight - let's make sure they are worth next to nothing
no_weight = lines["netWgt_kg"].isna()
small = lines.loc[no_weight, "primaryValue_usd_CIF"] < 1000
assert small.all(), "a line without weight is worth real money"
lines["kg"] = lines["netWgt_kg"].fillna(0)  # under USD 1,000, so count as 0
lines["no_weight"] = no_weight

grouped = lines.groupby(["period", "partnerCode"])
listed = grouped.agg(kg=("kg", "sum"),
                     estimated=("isNetWgtEstimated", "any"),
                     no_weight=("no_weight", "any")).reset_index()
months_listed = listed["period"].nunique()

# China lists an origin only when something was was recorded, so a month
# it does not list is a real 0
covered_months = sorted(raw["period"].unique())
grid = pd.MultiIndex.from_product([covered_months, list(codes)],
                                  names=["period", "partnerCode"])
grid = grid.to_frame(index=False)
full = grid.merge(listed, how="left", on=["period", "partnerCode"])
full["kg"] = full["kg"].fillna(0)
full["estimated"] = full["estimated"].eq(True)
full["no_weight"] = full["no_weight"].eq(True)

# Notes, for the rows with a missing or estimated weight
notes = []
for i in range(len(full)):
    parts = []
    if full["no_weight"][i]:
        parts.append("weight missing on one product line (under USD 1,000), "
                     "counted as 0")
    if full["estimated"][i]:
        parts.append("weight estimated by China")
    notes.append("; ".join(parts))

out = pd.DataFrame(
    {
        "source": "china_importer",
        "exporter": full["partnerCode"].map(codes),
        "counterparty": "CHN",
        "port": "",
        "month": pd.to_datetime(full["period"], format="%Y%m"),
        "tonnes": full["kg"] / 1000,  # netWgt is in kilograms
        "flag": "",
        "retrieved": RETRIEVED,
        "note": notes,
    }
)
out = out.sort_values(["month", "exporter"]).reset_index(drop=True)

# No month should be missing between the first and the last
months = out["month"].drop_duplicates()
every_month = pd.date_range(months.min(), months.max(), freq="MS")
assert len(months) == len(every_month), "a month is missing"

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
print(f"wrote {len(out)} rows to {OUT}")