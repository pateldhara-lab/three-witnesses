
import io
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]   
CSV = ROOT / "Processed" / "by_source" / "south_africa_exporter.csv"
RAW_DIR = ROOT / "Data" / "Export" / "South Africa"
COUNTRIES = ROOT / "Lookups" / "countries.csv"

COLUMNS = ["source", "exporter", "counterparty", "port", "month",
           "tonnes", "flag", "retrieved", "note"]

# Yearly totals in million tonnes from the README table
YEARLY_MT = {2015: 66.8, 2016: 64.9, 2017: 66.8, 2018: 63.5, 2019: 66.9,
             2020: 65.7, 2021: 67.8, 2022: 57.7, 2023: 59.0, 2024: 61.1,
             2025: 64.7, 2026: 43.5}
ORE_TARIFFS = ["26011100", "26011200"]


@pytest.fixture(scope="module")
def sa():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str,
                              "counterparty": str})


@pytest.fixture(scope="module")
def raw():
    files = sorted(RAW_DIR.glob("20*.xls"))   
    frames = []
    for f in files:
        book = io.BytesIO(f.read_bytes())
        frames.append(pd.read_excel(book, sheet_name="Report",
                                    engine="openpyxl", dtype=str,
                                    keep_default_na=False))
    data = pd.concat(frames, ignore_index=True)
    data['kg'] = data['StatisticalQuantity'].astype(float)
    data['ym'] = data['YearMonth'].astype(int)
    return data


def year_rows(sa, year):
    return sa[sa["month"].str[:4] == str(year)]


# Shape of the file

def test_columns_and_constants(sa):
    assert list(sa.columns) == COLUMNS
    assert (sa["source"] == "south_africa_exporter").all()
    assert (sa["exporter"] == "ZAF").all()
    assert (sa["retrieved"] == "2026-10-09").all()
    assert (sa["flag"] == "").all()   # nothing outside says a month is wrong


def test_no_duplicate_keys(sa):
    key = ["source", "exporter", "counterparty", "port", "month",
           "retrieved"]
    assert not sa.duplicated(key).any()



def test_no_invented_months(sa):
    assert sa["month"].min() == "2015-01-01"
    assert sa["month"].max() == "2026-08-01"
    assert sa["month"].nunique() == 11 * 12 + 8


# Weights

def test_yearly_totals(sa):
    for year in YEARLY_MT:
        total = year_rows(sa, year)["tonnes"].sum() / 1e6
        assert total == pytest.approx(YEARLY_MT[year], abs=0.06), year


# Filter (no foreign origin)

def test_foreign_origin(raw, sa):
    is_ore = raw["Tariff"].isin(ORE_TARIFFS)
    foreign = raw[is_ore & (raw["CountryOfOrigin"] != "ZA")]
    assert len(foreign) > 0
    ore_all_origins = raw[is_ore]["kg"].sum() / 1000
    removed = ore_all_origins - sa["tonnes"].sum()
    assert removed == pytest.approx(foreign["kg"].sum() / 1000)


# Destinations

def test_no_unmapped_destination(sa):
    countries = pd.read_csv(COUNTRIES, dtype=str, keep_default_na=False)
    assert set(sa["counterparty"]) <= set(countries["iso3"])
    assert (sa["counterparty"].str.len() == 3).all()


def test_mozambique(sa):
    # declared destination with a transit note
    moz = sa[sa["counterparty"] == "MOZ"]
    assert len(moz) > 0
    assert moz["note"].str.contains("transit").all()
    moz_2025 = year_rows(moz, 2025)["tonnes"].sum() / 1e6
    assert moz_2025 == pytest.approx(7.0, abs=0.05)   # README: 7.0 Mt in 2025
    others = sa[sa["counterparty"] != "MOZ"]
    assert not others["note"].str.contains("transit").any()


# Provisional months

def test_provisional(sa):
    is_2026 = sa["month"].str[:4] == "2026"
    assert sa.loc[is_2026, "note"].str.contains("provisional").all()
    assert not sa.loc[~is_2026, "note"].str.contains("provisional").any()
