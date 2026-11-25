"""
EU importer: Comext long csv -> Processed/by_source/eu_importer.csv

"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
RAW = ROOT / "Data" / "Import" / "EU" / "eu_iron_ore_imports_long.csv"
LOOKUP = ROOT / "Lookups" / "countries.csv"
OUT = ROOT / "Processed" / "by_source" / "eu_importer.csv"

RETRIEVED = "2026-10-09"  # download date, from README_european_union.txt
TOTAL_ROW = "EU27_2020"  # the reporter row that already adds up all 27 states
PRODUCTS = ["26011100", "26011200"]  # non-agglomerated, agglomerated

raw = pd.read_csv(RAW, dtype={"product_cn8": str, "period": str})

# quantity_tonnes in the file is rounded to 0.1 t, so we convert the native
# 100 kg unit ourselves - just checking it lines up first
diff = (raw["quantity_100kg"] / 10 - raw["quantity_tonnes"]).abs()
assert (diff < 0.06).all(), "unit conversion does not match"

# Partner codes -> our three letter code, for every exporter in the lookup
lookup = pd.read_csv(LOOKUP, dtype=str).fillna("")
ours = lookup[lookup["project_role"].str.startswith("exporter")]
codes = dict(zip(ours["eurostat_partner"], ours["iso3"]))

# eu_rows - only the EU27 total reporter, no member states
eu_total = raw[raw["reporter"] == TOTAL_ROW]

# The EU27 total should equal the member states added up (this is why we
# never add both)
rows = raw[raw["partner"].isin(codes) & raw["product_cn8"].isin(PRODUCTS)]
total_rows = rows[rows["reporter"] == TOTAL_ROW]
state_rows = rows[rows["reporter"] != TOTAL_ROW]
keys = ["partner", "product_cn8", "period"]
total_sums = total_rows.groupby(keys)["quantity_100kg"].sum()
state_sums = state_rows.groupby(keys)["quantity_100kg"].sum()
both = pd.concat([total_sums, state_sums], axis=1, keys=["total", "states"])
both = both.fillna(0)
difference = (both["total"] - both["states"]).abs()
assert (difference < 1).all(), "EU27 total differs from the member states"

grouped = total_rows.groupby(["period", "partner"], as_index=False)
listed = grouped["quantity_100kg"].sum()

# Comext lists an origin only when something was recorded, so a month it
# does not list is a real 0
covered_months = sorted(eu_total["period"].unique())
grid = pd.MultiIndex.from_product([covered_months, list(codes)],
                                  names=["period", "partner"])
grid = grid.to_frame(index=False)
full = grid.merge(listed, how="left", on=["period", "partner"])
full["quantity_100kg"] = full["quantity_100kg"].fillna(0)

out = pd.DataFrame(
    {
        "source": "eu_importer",
        "exporter": full["partner"].map(codes),
        "counterparty": "EU27",
        "port": "",
        "month": pd.to_datetime(full["period"] + "-01"),
        "tonnes": full["quantity_100kg"] / 10,  # the native unit is 100 kg
        "flag": "",
        "retrieved": RETRIEVED,
        "note": "",
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