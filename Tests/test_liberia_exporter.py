"""
Known-number tests for Processed/by_source/liberia_exporter.csv
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "Processed" / "by_source" / "liberia_exporter.csv"
RAW_DIR = ROOT / "Data" / "Export" / "Liberia"
MONTHLY = RAW_DIR / "liberia_iron_ore_monthly.csv"

COLUMNS = ["source", "exporter", "counterparty", "port", "month",
           "tonnes", "flag", "retrieved", "note"]

# Yearly totals in million tonnes, December 2025 is blank so it is left out.
# 2026 = Jan-Aug.
YEARLY_MT = {2019: 4.824, 2020: 5.178, 2021: 3.771, 2022: 4.578,
             2023: 3.589, 2024: 3.500, 2025: 8.362, 2026: 9.702}


@pytest.fixture(scope="module")
def liberia():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str,
                              "counterparty": str})


@pytest.fixture(scope="module")
def printed():
    a = pd.read_csv(MONTHLY, dtype={"month": str})
    return a[a["measure"] == "export_volume_tonnes"].set_index("month")


def tonnes_by_month(liberia):
    t = liberia.copy()
    t["month"] = pd.to_datetime(t["month"])
    t["tonnes"] = pd.to_numeric(t["tonnes"])   # blank cells become NaN
    return t.set_index("month")["tonnes"]


def row_for(liberia, month):
    return liberia[liberia["month"] == month + "-01"].iloc[0]


# Shape of the file

def test_shape(liberia):
    assert list(liberia.columns) == COLUMNS
    assert (liberia["source"] == "liberia_exporter").all()
    assert (liberia["exporter"] == "LBR").all()
    # total only, the bank gives no destination
    assert (liberia["counterparty"] == "ALL").all()
    assert (liberia["port"] == "").all()
    assert (liberia["retrieved"] == "2026-11-09").all()
    # Jan 2019 to Aug 2026 with no gap, and only December 2025 is blank
    expected = pd.date_range("2019-01-01", "2026-08-01", freq="MS")
    assert list(pd.to_datetime(liberia["month"])) == list(expected)
    t = tonnes_by_month(liberia)
    assert list(t.index[t.isna()]) == [pd.Timestamp("2025-12-01")]


# Known numbers

def test_yearly_totals(liberia):
    t = tonnes_by_month(liberia)
    for year in YEARLY_MT:
        total = t[str(year)].sum() / 1e6
        assert total == pytest.approx(YEARLY_MT[year], abs=0.006), year


def test_printed_figures(liberia, printed):
    # every month with tonnes matches a figure some issue printed, within 5 t
    # (October 2022 is printed as 461.84, in thousands of tonnes)
    t = tonnes_by_month(liberia).dropna()
    assert len(t) == 91
    for month, tonnes in t.items():
        key = month.strftime("%Y-%m")
        assert key in printed.index, f"{key}: no printed figure"
        figures = []
        for piece in printed.loc[key, "all_prints"].split(" ; "):
            figures.append(float(piece.split(":")[1].rstrip("†±≠*")))
        close = False
        for p in figures:
            if abs(tonnes - p) <= 5.0 or abs(tonnes - p * 1000) <= 5.0:
                close = True
        assert close, f"{key}: {tonnes} not among printed {figures}"


# The months where issues disagree

def test_changed_months(liberia):
    # October 2022: the Nov and Dec 2022 issues print 461.84 (thousands of
    # tonnes). 400,492.14 is September 2022 repeated in the Oct 2023 issue
    october = row_for(liberia, "2022-10")
    assert float(october["tonnes"]) == pytest.approx(461_840.0)
    assert october["flag"] == "unit_slip_corrected"
    assert 25.46e6 / float(october["tonnes"]) == pytest.approx(55.1, abs=0.1)
    # November 2022: 314,465.62 is November 2021 repeated in the Nov 2023 issue
    november = row_for(liberia, "2022-11")
    assert float(november["tonnes"]) == pytest.approx(432_793.13, abs=0.05)
    assert "314465.62" in november["note"]
    # July 2023 was blank before the older issues were read
    july = row_for(liberia, "2023-07")
    assert float(july["tonnes"]) == pytest.approx(180_099.54, abs=0.05)
    assert july["flag"] == ""
    # May 2021 was first printed as 392,193 t, then as 165,268 t twice
    may = row_for(liberia, "2021-05")
    assert float(may["tonnes"]) == pytest.approx(165_267.88, abs=0.05)
    assert "392192.58" in may["note"]


# The blank month

def test_december_2025(liberia, printed):
    row = row_for(liberia, "2025-12")
    assert row["tonnes"] == ""   # blank means not available, never zero
    assert row["flag"] == "known_source_error"
    assert "PortWatch" in row["note"] and "EU" in row["note"]
    # the bank's figure is still in the monthly file, not corrected
    assert printed.loc["2025-12", "agreed_value"] == pytest.approx(4_977_124.0)


# Flags

def test_flags(liberia):
    single = liberia[liberia["flag"] == "single_source"]
    assert list(single["month"]) == ["2024-06-01", "2026-08-01"]
    assert (liberia["flag"] == "missing").sum() == 0
    odd = ["single_source", "known_source_error", "unit_slip_corrected"]
    others = liberia[~liberia["flag"].isin(odd)]
    assert len(others) == 92 - 4
    assert (others["flag"] == "").all()
    assert (pd.to_numeric(others["tonnes"]) > 0).all()
