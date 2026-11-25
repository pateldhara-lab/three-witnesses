"""Known-number tests for Processed/by_source/eu_importer.csv
"""
from pathlib import Path
import pandas as pd
import pytest

CSV = Path(__file__).resolve().parents[1] / "Processed" / "by_source" / \
    "eu_importer.csv"

# (exporter, year): million tonnes
YEARLY_MT = {("BRA", 2016): 44.83, ("BRA", 2019): 31.99,
             ("BRA", 2024): 16.57, ("BRA", 2025): 13.23,
             ("BRA", 2026): 7.28, ("ZAF", 2024): 8.89,
             ("LBR", 2024): 2.71, ("LBR", 2025): 8.02, ("PER", 2025): 0.08}


@pytest.fixture(scope="module")
def eu():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str})


def tonnes(eu, exporter, month):
    row = eu[(eu["exporter"] == exporter) & (eu["month"] == month)]
    assert len(row) == 1
    return row["tonnes"].iloc[0]


def test_size(eu):
    assert len(eu) == 5 * 127, len(eu)
    assert eu["month"].nunique() == 127
    assert eu["exporter"].nunique() == 5
    assert eu["month"].min() == "2016-01-01"
    assert eu["month"].max() == "2026-07-01"


def test_dec_2024(eu):
    bra = tonnes(eu, "BRA", "2024-12-01")
    lbr = tonnes(eu, "LBR", "2024-12-01")
    zaf = tonnes(eu, "ZAF", "2024-12-01")
    assert bra == pytest.approx(871_549.3, abs=1), bra   # 679,645.7 + 191,903.6
    assert lbr == pytest.approx(459_538.9, abs=1), lbr
    assert zaf == pytest.approx(253_320.4, abs=1), zaf


def test_january_2021_brazil(eu):
    jan = tonnes(eu, "BRA", "2021-01-01") / 1e6
    jan_rows = eu[eu["month"] == "2021-01-01"]
    assert jan == pytest.approx(1.5, abs=0.05), jan


def test_yearly_totals(eu):
    for key in YEARLY_MT:
        exporter, year = key
        rows = eu[(eu["exporter"] == exporter) &
                  (eu["month"].str[:4] == str(year))]
        total = rows["tonnes"].sum() / 1e6
        assert total == pytest.approx(YEARLY_MT[key], abs=0.01), key


def test_unlisted_month(eu):
    # Peru never apears in the EU file for 2016
    assert tonnes(eu, "PER", "2016-01-01") == 0