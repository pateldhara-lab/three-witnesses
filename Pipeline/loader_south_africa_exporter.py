"""
South Africa exporter: SARS workbooks -> south_africa_exporter.csv

"""
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
RAW_DIR = ROOT / "Data" / "Export" / "South Africa"
COUNTRIES = ROOT / "Lookups" / "countries.csv"
PORTS = ROOT / "Lookups" / "ports.csv"
OUT = ROOT / "Processed" / "by_source" / "south_africa_exporter.csv"

RETRIEVED = "2026-10-09"  # from the README, the file names carry no date
ORE_TARIFFS = ["26011100", "26011200"]  # fines and lump, pellets, NOT 26012000
# Rows per workbook
FILE_ROWS = {
    "2015-2016.xls": 650, "2017-2018.xls": 796, "2019-2020.xls": 740,
    "2021-2022.xls": 804, "2023-2024.xls": 705, "2025-2026.xls": 633,
}
MOZAMBIQUE_NOTE = ("transit: declared destination, ore moves on to Maputo "
                   "(README item 1); not a buyer")
PROVISIONAL_NOTE = "provisional (2026 months can be revised)"

# read all six workbooks as text so Namibia's code 'NA' stays text

frames = []
for name, expected in FILE_ROWS.items():
    sheet = pd.read_excel(io.BytesIO((RAW_DIR / name).read_bytes()),
                          sheet_name="Report", engine="openpyxl", dtype=str,
                          keep_default_na=False)
    assert len(sheet) == expected, f"{name}: read {len(sheet)} rows"
    frames.append(sheet)
data = pd.concat(frames, ignore_index=True)

assert (data["TradeType"] == "Exports").all(), "a row that is not an export"
assert (data["StatisticalUnit"] == "KG").all(), "a row not in kilograms"
assert (data["Chapter"] == "26").all(), "a row outside chapter 26"
tariffs = set(ORE_TARIFFS) | {"26012000"}
assert set(data["Tariff"]) <= tariffs, "an unexpected tariff line"
data["kg"] = pd.to_numeric(data["StatisticalQuantity"])
assert data["kg"].notna().all(), "a row without weight"
year_from_month = pd.to_numeric(data["YearMonth"]) // 100
year = pd.to_numeric(data["CalendarYear"])
assert (year_from_month == year).all(), "YearMonth and CalendarYear disagree"

# Only the iron ore tariff lines with South African origin
is_ore = data["Tariff"].isin(ORE_TARIFFS)
is_za = data["CountryOfOrigin"] == "ZA"
rows = data[is_ore & is_za].copy()

# Destination name -> three letter code, only through countries.csv
lookup = pd.read_csv(COUNTRIES, dtype=str, keep_default_na = False)
lookup = lookup[lookup["sars_name"] != ""]
assert lookup["sars_name"].is_unique, "a SARS name appears twice"
by_name = lookup.set_index("sars_name")
names = rows["CountryOfDestinationName"].str.strip()
unknown = sorted(set(names) - set(by_name.index))
assert not unknown, f"names missing from countries.csv: {unknown}"
# SARS also gives a two letter code, it has to agree with the name
wrong = rows["CountryOfDestination"] != names.map(by_name["iso2"])
bad_names = sorted(set(names[wrong]))
assert not wrong.any(), f"code and name disagree: {bad_names}"

# Customs office -> PortWatch port id, the other offices stay blank
ports = pd.read_csv(PORTS, dtype=str, keep_default_na=False)
ports = ports[(ports["country_iso3"] == "ZAF") & (ports["sars_office"] != "")]
assert ports["sars_office"].is_unique, "ports.csv repeats an office"
assert ports["portid"].is_unique, "ports.csv repeats a port id"
office_to_port = dict(zip(ports["sars_office"], ports["portid"]))
offices = rows["DistrictOfficeName"].str.strip()
port_ids = offices.map(office_to_port).fillna("")

pieces = pd.DataFrame(
    {
        "counterparty": names.map(by_name["iso3"]),
        "port": port_ids,
        "month": pd.to_datetime(rows["YearMonth"] + "01", format="%Y%m%d"),
        "tonnes": rows["kg"] / 1000,  # kilograms
    }
)
out = pieces.groupby(["counterparty", "port", "month"],
                     as_index=False)["tonnes"].sum()
out.insert(0, "exporter", "ZAF")
out.insert(0, "source", "south_africa_exporter")
out["flag"] = ""
out["retrieved"] = RETRIEVED

# Notes - Mozambique is transit, and 2026 months can still be revised
notes = pd.Series("", index=out.index)
notes = notes.mask(out["counterparty"] == "MOZ", MOZAMBIQUE_NOTE)
is_2026 = out["month"].dt.year == 2026
notes = notes.mask(is_2026 & (notes == ""), PROVISIONAL_NOTE)
notes = notes.mask(is_2026 & (notes == MOZAMBIQUE_NOTE),
                   MOZAMBIQUE_NOTE + "; " + PROVISIONAL_NOTE)
out["note"] = notes
out = out.sort_values(["month", "counterparty", "port"])
out = out.reset_index(drop=True)

# No month should be missing between the first and the last
months = out["month"].drop_duplicates()
every_month = pd.date_range(months.min(), months.max(), freq="MS")
assert len(months) == len(every_month), "a month is missing"

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
print(f"wrote {len(out)} rows to {OUT}")
print(f"left out: {int((~is_ore).sum())} pyrites rows (tariff 26012000)")
print(f"left out: {int((is_ore & ~is_za).sum())} foreign-origin ore rows")