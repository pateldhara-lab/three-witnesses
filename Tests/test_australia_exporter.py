"""
Known-number tests for Processed/by_source/australia_exporter.csv
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "Processed" / "by_source" / "australia_exporter.csv"
LOOKUPS = ROOT / "Lookups"

COLUMNS = ["source", "exporter", "counterparty", "port", "month",
           "tonnes", "flag", "retrieved", "note"]
HEDLAND, DAMPIER = "port955", "port276"


@pytest.fixture(scope="module")
def australia():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str,
                              "counterparty": str})


def check(condition, message):
    """Fails the test with the message when the condition is false."""
    if not condition:
        pytest.fail(message)


def check_month(australia, port, month, expected):
    """Fails unless the port has exactly one row for the month with this
    many tonnes."""
    row = australia[(australia["port"] == port)
                    & (australia["month"] == month)]
    check(len(row) == 1, f"{port} {month}: {len(row)} rows")
    got = float(row["tonnes"].iloc[0])
    check(got == expected, f"{port} {month}: {got:,.0f}, not {expected:,}")


# Shape of the file

def test_columns_and_constants(australia):
    check(list(australia.columns) == COLUMNS,
          f"columns are {list(australia.columns)}")
    check((australia["source"] == "australia_exporter").all(),
          "a row has the wrong source")
    check((australia["exporter"] == "AUS").all(),
          "a row has the wrong exporter")
    # Dampier gives no destinations, so both ports are totals
    check((australia["counterparty"] == "ALL").all(),
          "a row has a destination")
    check((australia["flag"] == "").all(), "a row has a flag")


def test_clean_file_rules(australia):
    key = ["source", "exporter", "counterparty", "port", "month", "retrieved"]
    check(not australia.duplicated(key).any(), "duplicate rows")
    check(australia["month"].str.endswith("-01").all(),
          "a month is not the first of the month")
    check((pd.to_numeric(australia["tonnes"]) >= 0).all(),
          "negative tonnes")
    ports = pd.read_csv(LOOKUPS / "ports.csv")["portid"]
    unknown = set(australia["port"]) - set(ports)
    check(len(unknown) == 0, f"ports not in ports.csv: {unknown}")


def test_every_month(australia):
    # a missing month would shrink Hedland + Dampier in the table
    months = pd.date_range("2019-01-01", "2026-08-01", freq="MS")
    expected = list(months.strftime("%Y-%m-%d"))
    for port in [HEDLAND, DAMPIER]:
        got = list(australia[australia["port"] == port]["month"]) 
        check(got == expected, f"{port}: {len(got)} months, not 92")


# known numbers read off the printed PDFs
def test_hedland_known_months(australia):
    # Total row, iron ore column, no domestic row
    check_month(australia, HEDLAND, "2025-06-01", 54_584_776)
    # 2024-05: the Total column is on page 3 but the iron ore column is not
    check_month(australia, HEDLAND, "2024-05-01", 51_940_591)


def test_hedland_domestic_row(australia):
    # 36,386,165 in the Total row less 416,533 on the Australia row
    check_month(australia, HEDLAND, "2019-03-01", 35_969_632)
    note = australia[(australia["port"] == HEDLAND)
                     & (australia["month"] == "2019-03-01")]["note"].iloc[0]
    check("416,533" in note, f"the note says: {note}")


def test_hedland_year(australia):
    h = australia[australia["port"] == HEDLAND]
    total = h[h["month"].str.startswith("2025")]["tonnes"].sum() / 1e6
    check(total == pytest.approx(575.0, abs=0.05),
          f"2025 is {total:.2f} Mt, not 575.0")


def test_dampier_known_months(australia):
    check_month(australia, DAMPIER, "2023-07-01", 11_613_348)
    # 2019-01 printed with a gap in the digits (9 ,888,000)
    check_month(australia, DAMPIER, "2019-01-01", 9_888_000)


def test_dampier_financial_years(australia):
    d = australia[australia["port"] == DAMPIER].set_index("month")["tonnes"]
    # TOTALS row of each financial year
    fy2324 = d["2023-07-01":"2024-06-01"].sum()
    fy2021 = d["2020-07-01":"2021-06-01"].sum()
    check(fy2324 == 144_899_955, f"FY2023-24 is {fy2324:,.0f}")
    check(fy2021 == 136_599_227, f"FY2020-21 is {fy2021:,.0f}")


