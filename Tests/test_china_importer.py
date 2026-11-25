"""
Known-number tests for Processed/by_source/china_importer.csv

"""
from pathlib import Path
import pandas as pd
import pytest

CSV = Path(__file__).resolve().parents[1] / "Processed" / "by_source" / \
    "china_importer.csv"

# (exporter, year): million tonnes, the partner rows of the source file
# added up by year
YEARLY_MT = {("BRA", 2019): 227.5, ("BRA", 2024): 273.0,
             ("AUS", 2019): 664.4, ("AUS", 2024): 742.5,
             ("ZAF", 2024): 38.2, ("PER", 2019): 15.0, ("PER", 2024): 22.4,
             ("LBR", 2019): 1.1}


@pytest.fixture(scope="module")
def china():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str})


def tonnes(china, exporter, month):
    row = china[(china["exporter"] == exporter) & (china["month"] == month)]
    assert len(row) == 1
    return row["tonnes"].iloc[0]


def test_january_2021_brazil(china):
    jan = tonnes(china, "BRA", "2021-01-01") / 1e6
    assert jan == pytest.approx(19.1, abs=0.05), jan


def test_yearly_totals(china):
    for key in YEARLY_MT:
        exporter, year = key
        rows = china[(china["exporter"] == exporter) &
                     (china["month"].str[:4] == str(year))]
        total = rows["tonnes"].sum() / 1e6
        assert total == pytest.approx(YEARLY_MT[key], abs=0.05), key


def test_unlisted_month(china):
    # China listed no Peruvian ore for July 2020, and no Liberian ore for
    # November 2024 so these are real zeros zeros
    assert tonnes(china, "PER", "2020-07-01") == 0
    assert tonnes(china, "LBR", "2024-11-01") == 0


def test_notes(china):
    estimated = china[(china["exporter"] == "PER") &
                      (china["month"] == "2022-02-01")]
    assert "estimated" in estimated["note"].iloc[0]
    no_weight = china[(china["exporter"] == "LBR") &
                      (china["month"] == "2021-12-01")]
    assert "missing" in no_weight["note"].iloc[0]