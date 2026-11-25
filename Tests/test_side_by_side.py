"""Tests for Processed/side_by_side.csv 
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]   
PROCESSED = ROOT / "Processed"
BY_SOURCE = PROCESSED / "by_source"
PORTS = ROOT / "Lookups" / "ports.csv"

COLUMNS = ["country", "month", "exporter_t", "ship_t", "ship_extra_ports_t",
           "china_raw_t", "eu_raw_t", "lag_china", "lag_eu",
           "china_aligned_t", "eu_aligned_t", "importer_aligned_total_t",
           "flag"]
# Million tonnes, December 2025 left out (see test_liberia_exporter.py)
LIBERIA_YEARLY_MT = {2019: 4.824, 2020: 5.178, 2021: 3.771, 2022: 4.578,
                     2023: 3.589, 2024: 3.500, 2025: 8.362, 2026: 9.702}
LIBERIA_SINGLE_SOURCE = ["2024-06", "2026-08"]


@pytest.fixture(scope="module")
def table():
    t = pd.read_csv(PROCESSED / "side_by_side.csv", parse_dates=["month"])
    t["flag"] = t["flag"].astype("string").fillna("")
    return t


@pytest.fixture(scope="module")
def ship_clean():
    s = pd.read_csv(BY_SOURCE / "portwatch_ship.csv",
                    keep_default_na=False, dtype=str)
    s["month"] = pd.to_datetime(s["month"])
    s["tonnes"] = s["tonnes"].astype(float)
    return s


@pytest.fixture(scope="module")
def coverage():
    return pd.read_csv(PROCESSED / "coverage_by_year.csv")


@pytest.fixture(scope="module")
def steadiness():
    return pd.read_csv(PROCESSED / "steadiness_test.csv")


@pytest.fixture(scope="module")
def lag_fit():
    return pd.read_csv(PROCESSED / "lag_fit.csv")


def block(table, country):
    b = table[table["country"] == country].set_index("month")
    if b.empty:
        pytest.skip(f"{country} is not in the table")
    return b


# Shape

def test_columns(table):
    assert list(table.columns) == COLUMNS


def test_one_row_a_month(table):
    assert not table.duplicated(["country", "month"]).any()
    assert (table["month"].dt.day == 1).all()


def test_brazil_months(table):
    assert table["country"].iloc[0] == "BRA"
    brazil = table[table["country"] == "BRA"]["month"]
    expected = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
    assert list(brazil) == list(expected)
    # Brazil rows first, then South Africa, Peru, Liberia, Australia
    countries = list(table["country"].drop_duplicates())
    order = []
    for c in ["BRA", "ZAF", "PER", "LBR", "AUS"]:
        if c in countries:
            order.append(c)
    assert countries == order


# Blanks are not zeros

def test_blank_not_zero(table):
    b = block(table, "BRA")
    # China's customs data stop stop in Dec 2024
    assert b.loc["2025-01-01":, "china_raw_t"].isna().all()
    # EU data stop in Jul 2026
    assert b.loc["2026-08-01":, "eu_raw_t"].isna().all()
    # only Australia has extra ports
    assert b["ship_extra_ports_t"].isna().all()
    columns = ["exporter_t", "ship_t", "china_raw_t", "eu_raw_t"]
    assert (b[columns].min() >= 0).all()   # min skips blanks


# Known numbers (Brazil, January 2021)

def test_brazil_january_2021(table):
    row = block(table, "BRA").loc["2021-01-01"]
    assert row["exporter_t"] / 1e6 == pytest.approx(28.93, abs=0.005)
    assert row["ship_t"] / 1e6 == pytest.approx(26.0, abs=0.05)
    assert row["china_raw_t"] / 1e6 == pytest.approx(19.1, abs=0.05)
    assert row["eu_raw_t"] / 1e6 == pytest.approx(1.5, abs=0.05)


# The lag shift

def test_aligned(table):
    pairs = [("china_raw_t", "china_aligned_t", "lag_china"),
             ("eu_raw_t", "eu_aligned_t", "lag_eu")]
    for raw_col, aligned_col, lag_col in pairs:
        for country, g in table.groupby("country", sort=False):
            g = g.set_index("month")
            lag = int(g[lag_col].iloc[0])
            # a gap-free monthly index, so shifting rows is shifting months
            expected = g[raw_col].shift(-lag)
            pd.testing.assert_series_equal(g[aligned_col], expected,
                                           check_names = False)


def test_importer_total(table):
    # China plus EU, blank unless both exist
    both = table["china_aligned_t"].notna() & table["eu_aligned_t"].notna()
    total = table["china_aligned_t"] + table["eu_aligned_t"]
    pd.testing.assert_series_equal(table["importer_aligned_total_t"],
                                   total.where(both), check_names=False)
    assert table.loc[~both, "importer_aligned_total_t"].isna().all()


# ship total

def test_ship_total(table, ship_clean):
    ports = pd.read_csv(PORTS, dtype=str, keep_default_na=False)
    for country in table["country"].unique():
        feeds = ports[(ports["country_iso3"] == country) &
                      (ports["in_country_ship_total"] == "yes") &
                      (ports["feeds_table_column"] == "ship_t")]
        ids = feeds["portid"]
        rows = ship_clean[ship_clean["port"].isin(ids) &
                          (ship_clean["exporter"] == country)]
        by_port = rows.pivot_table(index="month", columns="port",
                                   values="tonnes", aggfunc="sum")
        expected = by_port.sum(axis=1)
        got = table[table["country"] == country].set_index("month")["ship_t"]
        got = got.dropna()
        pd.testing.assert_series_equal(got, expected.reindex(got.index),
                                       check_names=False)


# Coverage, steadiness, lag fit

def test_brazil_ship_cover(table, coverage):
    b = block(table, "BRA")
    both = b[["exporter_t", "ship_t"]].dropna()
    ship_share = both["ship_t"].sum() / both["exporter_t"].sum()
    # ship tonnes are 97% of exporter tonnes, Jan 2019 - Sep 2026
    assert ship_share == pytest.approx(0.967, abs=0.002)
    levels = both["exporter_t"].corr(both["ship_t"])
    assert levels == pytest.approx(0.85, abs=0.01)
    both_t = b[["exporter_t", "ship_t"]]
    pct = (both_t / both_t.shift(12) - 1).dropna()
    # year-over-year PERCENT changes
    changes = pct["exporter_t"].corr(pct["ship_t"])
    assert changes == pytest.approx(0.67, abs=0.01)
    w = b.loc["2019":"2024"]
    imports = (w["china_raw_t"] + w["eu_raw_t"]).sum()
    # China plus EU are 73% of exporter tonnes
    assert imports / w["exporter_t"].sum() == pytest.approx(0.73, abs=0.005)
    bra = coverage[coverage["country"] == "BRA"].set_index("year")
    ship = bra["ship_cover_pct"]
    # Yearly ship / exporter (the drift)
    expected = {2019: 98.1, 2020: 101.6, 2021: 98.0, 2022: 98.0,
                2023: 98.2, 2024: 96.2, 2025: 91.7, 2026: 91.9}
    for year in expected:
        assert ship[year] == pytest.approx(expected[year], abs = 0.1), year


def test_steadiness_range(coverage, steadiness):
    for _, r in steadiness.iterrows():
        is_country = coverage["country"] == r["country"]
        w = coverage[is_country & coverage["year"].between(2019, 2024)]
        w = w[r["series"] + "_pct"]
        spread = w.max() - w.min()
        assert r["range_points"] == pytest.approx(spread, abs=1e-3)
        assert (r["passes_7_point_range"] == "yes") == (spread <= 7.0)


def test_table_lags(table, lag_fit):
    assert set(lag_fit["lag"]) == {0, 1, 2, 3}
    for (country, importer), g in lag_fit.groupby(["country", "importer"]):
        best = int(g.loc[g["corr_yoy"].idxmax(), "lag"])
        col = "lag_eu"
        if importer == "CHN":
            col = "lag_china"
        assert int(table[table["country"] == country][col].iloc[0]) == best
        # every lag is judged on the same months
        assert (g["n_months_yoy"] == g["n_months_yoy"].iloc[0]).all()


# Peru (third exporter)

def test_peru_months(table):
    b = block(table, "PER")
    expected = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
    assert list(b.index) == list(expected)
    blank = list(b.index[b["exporter_t"].isna()])
    # the zero treated as missing, and September 2026 not yet published
    assert blank == [pd.Timestamp("2020-05-01"), pd.Timestamp("2026-09-01")]
    assert b.loc["2020-05-01", "flag"] == "missing"


# Liberia (fourth exporter)

def test_liberia_yearly_totals(table):
    b = block(table, "LBR")
    for year in LIBERIA_YEARLY_MT:
        total = b.loc[str(year), "exporter_t"].sum() / 1e6
        assert total == pytest.approx(LIBERIA_YEARLY_MT[year], abs=0.006), year


def test_liberia_months(table):
    b = block(table, "LBR")
    expected = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
    assert list(b.index) == list(expected)
    # December 2025 is the bank's error, and September 2026 is not out yet
    blank = list(b.index[b["exporter_t"].isna()])
    assert blank == [pd.Timestamp("2025-12-01"), pd.Timestamp("2026-09-01")]
    # the months we used to hold are the same, except two old copies
    assert b.loc["2022-10-01", "exporter_t"] == pytest.approx(461_840.0)
    assert b.loc["2022-11-01", "exporter_t"] == pytest.approx(432_793.13)
    assert b.loc["2022-12-01", "exporter_t"] == pytest.approx(398_643.71)
    assert b.loc["2023-07-01", "exporter_t"] == pytest.approx(180_099.54)
    assert b.loc["2026-06-01", "exporter_t"] == pytest.approx(1_053_218.0)


def test_liberia_flags(table):
    b = block(table, "LBR")
    assert b.loc["2022-10-01", "flag"] == "unit_slip_corrected"
    assert b.loc["2023-07-01", "flag"] == ""
    assert b.loc["2025-12-01", "flag"] == "known_source_error"
    assert not (b["flag"] == "missing").any()
    single = b.index[b["flag"] == "single_source"].strftime("%Y-%m")
    assert sorted(single) == LIBERIA_SINGLE_SOURCE
    assert (b["flag"] == "").sum() == 93 - 4


def test_liberia_steadiness(steadiness):
    st = steadiness[steadiness["country"] == "LBR"]
    # 2019 to 2024
    assert (st["years_used"] == 6).all()
    assert (st["passes_60_floor"] == "yes").all()
    assert (st["passes_7_point_range"] == "no").all()


def test_liberia_lags(table, lag_fit):
    b = block(table, "LBR")
    # the fit now uses 69 months (Jan 2019 - Sep 2024), 57 year-over-year
    lbr = lag_fit[lag_fit["country"] == "LBR"]
    assert (lbr["n_months_yoy"] == 57).all()
    assert b["lag_eu"].iloc[0] == 1
    eu = lbr[(lbr["importer"] == "EU27") & (lbr["lag"] == 1)]
    assert eu["corr_yoy"].iloc[0] == pytest.approx(0.74, abs=0.01)
    # China's customs data barely follow Liberia's exports at any lag
    china = lbr[lbr["importer"] == "CHN"]
    assert china["corr_yoy"].max() < 0.1


# Australia (fifth exporter)

def check(condition, message):
    """Fails the test with the message when the condition is false."""
    if not condition:
        pytest.fail(message)


def test_australia_months(table):
    b = block(table, "AUS")
    expected = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
    check(list(b.index) == list(expected),
          f"{len(b)} months, not {len(expected)}")
    # Pilbara Ports has nothing for September 2026 yet
    blank = list(b.index[b["exporter_t"].isna()])
    check(blank == [pd.Timestamp("2026-09-01")], f"blank months: {blank}")


def test_australia_both_ports(table):
    b = block(table, "AUS")
    clean = pd.read_csv(BY_SOURCE / "australia_exporter.csv")
    for month in ["2019-03-01", "2025-06-01", "2026-08-01"]:
        rows = clean[clean["month"] == month]
        check(len(rows) == 2, f"{month}: {len(rows)} port rows")  # 2 ports
        total = rows["tonnes"].sum()
        got = b.loc[month, "exporter_t"]
        check(got == pytest.approx(total),
              f"{month}: table {got:,.0f}, ports {total:,.0f}")


def test_australia_cape_lambert(table):
    b = block(table, "AUS")
    # PortWatch sees Cape Lambert, the Pilbara Ports record does not
    check(b["ship_extra_ports_t"].notna().all(), "a month has no ships")
    share = (b["ship_extra_ports_t"] / b["exporter_t"]).dropna()
    # its size next to Hedland + Dampier slides from 27% to 22%
    first, last = share["2019"].mean(), share["2025"].mean()
    check(first == pytest.approx(0.268, abs=0.005), f"2019 is {first:.3f}")
    check(last == pytest.approx(0.225, abs=0.005), f"2025 is {last:.3f}")


def test_australia_ship_cover(coverage):
    c = coverage[coverage["country"] == "AUS"]
    check(len(c) == 8, f"{len(c)} years, not 8")
    cover = list(c["ship_cover_pct"].round(1))
    check(c["ship_cover_pct"].between(90, 98).all(), f"ship cover: {cover}")


def test_australia_steadiness(steadiness):
    st = steadiness[steadiness["country"] == "AUS"]
    years = list(st["years_used"])
    check((st["years_used"] == 6).all(), f"years used: {years}")
    check((st["passes_60_floor"] == "yes").all(), "fails the 60% floor")
    check((st["passes_7_point_range"] == "yes").all(),
          "fails the 7-point range")
