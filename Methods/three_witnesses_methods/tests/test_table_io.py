import numpy as np
import pandas as pd
import pytest

from fake_data import make_fake_table, SimSpec
from table_io import (REQUIRED_COLUMNS, load_table, country_yoy, run_by_country)


def test_fake_table_has_exactly_bens_13_columns_in_order():
    t = make_fake_table(countries=("BRA",))
    assert tuple(t.columns) == REQUIRED_COLUMNS
    assert len(REQUIRED_COLUMNS) == 13


def test_loader_roundtrip_via_csv(tmp_path):
    p = tmp_path / "t.csv"
    make_fake_table().to_csv(p, index=False)
    df = load_table(p)
    assert set(df["country"]) == {"BRA", "AUS", "ZAF", "PER", "LBR"}
    assert df.attrs["warnings"] == []


def test_missing_column_gives_plain_error():
    t = make_fake_table(countries=("BRA",)).drop(columns=["ship_t"])
    with pytest.raises(ValueError, match="missing columns"):
        load_table(t)


def test_month_must_be_first_of_month_and_no_duplicates():
    t = make_fake_table(countries=("BRA",))
    bad = t.copy(); bad.loc[0, "month"] = pd.Timestamp("2019-01-15")
    with pytest.raises(ValueError, match="first of the month"):
        load_table(bad)
    with pytest.raises(ValueError, match="duplicate"):
        load_table(pd.concat([t, t.iloc[[0]]]))


def test_unknown_flag_rejected():
    t = make_fake_table(countries=("BRA",)); t.loc[3, "flag"] = "weird"
    with pytest.raises(ValueError, match="unknown flag"):
        load_table(t)


def test_total_not_equal_sum_is_warned():
    t = make_fake_table(countries=("BRA",)); t.loc[5, "importer_aligned_total_t"] += 50
    assert any("!=" in w for w in load_table(t).attrs["warnings"])


def test_full_window_gives_60_yoy_rows():
    ci = country_yoy(load_table(make_fake_table()), "BRA")
    assert ci.yoy.shape == (60, 3) and ci.n_excluded_by_flag == 0


def test_flagged_month_removes_it_and_the_same_month_next_year():
    t = make_fake_table(flags={("BRA", "2021-03-01"): "known_source_error"})
    ci = country_yoy(load_table(t), "BRA")
    assert ci.n_excluded_by_flag == 1
    assert ci.yoy.shape[0] == 58                      # lost Mar 2021 and Mar 2022
    assert pd.Timestamp("2021-03-01") not in ci.months
    assert pd.Timestamp("2022-03-01") not in ci.months


def test_blank_month_cannot_shift_the_year_gap():
    # a deleted month must not make 'a year earlier' point 13 months back
    t = make_fake_table(blank={("BRA", "2020-06-01")})
    df = load_table(t[t["month"] != pd.Timestamp("2020-06-01")])   # row physically absent
    ci = country_yoy(df, "BRA")
    assert ci.yoy.shape[0] == 58


def test_missing_importer_is_not_treated_as_zero():
    t = make_fake_table(countries=("BRA",))
    t.loc[30, ["china_aligned_t", "importer_aligned_total_t"]] = np.nan
    ci = country_yoy(load_table(t), "BRA")
    assert ci.yoy.shape[0] == 58
    assert np.isfinite(ci.yoy).all()


def test_window_default_is_2019_to_2024():
    t = make_fake_table(countries=("BRA",), spec=SimSpec(n_months=84))   # 2019-2025 present
    ci = country_yoy(load_table(t), "BRA")
    assert ci.months.min() == pd.Timestamp("2020-01-01")
    assert ci.months.max() == pd.Timestamp("2024-12-01")


def test_run_by_country_and_too_little_data_is_explained():
    t = make_fake_table(countries=("BRA", "LBR"))
    t.loc[(t["country"] == "LBR") & (t["month"] < "2023-01-01"),
          ["exporter_t", "importer_aligned_total_t", "ship_t"]] = np.nan
    res = run_by_country(load_table(t))
    assert res["BRA"][1] is not None and res["BRA"][1].names == ("exporter", "importer", "ships")
    assert res["LBR"][1] is None and "not run" in res["LBR"][2]


def test_steady_importer_share_does_not_break_the_method():
    # importer sees ~72% of the ore (as in the real table); noise stays recoverable
    spec = SimSpec(n_months=72, scales=(1.0, 0.72, 0.95))
    t = make_fake_table(countries=("BRA",), spec=spec, seed=7)
    ci = country_yoy(load_table(t), "BRA")
    assert ci.yoy.shape == (60, 3)
