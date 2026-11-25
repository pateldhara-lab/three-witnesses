"""
Known-number tests for Processed/by_source/brazil_exporter.csv
"""
from pathlib import Path
import pandas as pd
import pytest

CSV = Path(__file__).resolve().parents[1] / "Processed" / "by_source" / \
    "brazil_exporter.csv"

# Yearly totals in million tonnes (from the README)
# 2026 is January to September only
YEARLY_MT = {2016: 374.0, 2017: 383.5, 2018: 389.8, 2019: 351.0,
             2020: 341.5, 2021: 357.3, 2022: 344.0, 2023: 378.1,
             2024: 388.7, 2025: 416.3, 2026: 297.8}


@pytest.fixture(scope="module")
def brazil():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str})


def test_no_invented_months(brazil):
    assert brazil["month"].min() == "2016-01-01"
    assert brazil["month"].max() == "2026-09-01"


def test_yearly_totals(brazil):
    for year in YEARLY_MT:
        rows = brazil[brazil["month"].str[:4] == str(year)]
        total = rows["tonnes"].sum() / 1e6
        assert total == pytest.approx(YEARLY_MT[year], abs=0.05), year


def test_zero_weight_row(brazil):
    # check the zero row for Canada is kept 
    row = brazil[(brazil["month"] == "2024-02-01") &
                 (brazil["counterparty"] == "CAN")]
    assert len(row) == 1
    assert row["tonnes"].iloc[0] == 0