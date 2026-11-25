"""Known-number tests for Processed/by_source/peru_exporter.csv
"""
from pathlib import Path
import openpyxl
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]   
CSV = ROOT / "Processed" / "by_source" / "peru_exporter.csv"
RAW_DIR = ROOT / "Data" / "Export" / "Peru"
VOLUME = next((RAW_DIR / "by volume").glob("*.xlsx"))

COLUMNS = ["source", "exporter", "counterparty", "port", "month",
           "tonnes", "flag", "retrieved", "note"]

SPANISH = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Ago": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12}


@pytest.fixture(scope="module")
def peru():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str,
                              "counterparty": str})


@pytest.fixture(scope="module")
def raw_volume():
    # first two rows are the series code and tilte
    ws = openpyxl.load_workbook(VOLUME, data_only=True)["Mensuales"]
    rows = list(ws.iter_rows(values_only=True))
    series = []
    for label, value in rows[2:]:
        series.append((label, value))
    return series


def tonnes_by_month(peru):
    t = peru.copy()
    t["month"] = pd.to_datetime(t["month"])
    t["tonnes"] = pd.to_numeric(t["tonnes"])   # blank become NaN
    return t.set_index("month")["tonnes"]


# Shape 

def test_columns_and_constants(peru):
    assert list(peru.columns) == COLUMNS
    assert (peru["source"] == "peru_exporter").all()
    assert (peru["exporter"] == "PER").all()
    assert (peru["counterparty"] == "ALL").all()
    assert (peru["port"] == "").all() # source has no port either
    assert (peru["retrieved"] == "2026-10-09").all()


# Known numbers

def test_every_month(peru, raw_volume):
    t = tonnes_by_month(peru)
    for label, value in raw_volume:
        month = pd.Timestamp(year=2000 + int(label[3:]),
                             month=SPANISH[label[:3]], day=1)
        if month == pd.Timestamp("2020-05-01"):
            continue   # checked below
        assert t[month] == pytest.approx((value * 1_000_000), rel=1e-9), label


# May 2020 is a zero in the source

def test_may_2020(peru):
    row = peru[peru["month"] == "2020-05-01"].iloc[0]
    assert row["tonnes"] == ""   
    assert row["flag"] == "missing"
    assert "PortWatch" in row["note"] and "China" in row["note"]


def test_other_months(peru):
    others = peru[peru["month"] != "2020-05-01"]
    assert (others["flag"] == "").all()
    assert (pd.to_numeric(others["tonnes"]) > 0).all()


# 2026 is provisional

def test_provisional(peru):
    is_2026 = peru["month"].str[:4] == "2026"
    assert peru.loc[is_2026, "note"].str.contains("provisional").all()
    assert is_2026.sum() == 8
    assert not peru.loc[~is_2026, "note"].str.contains("provisional").any()
