import numpy as np
import pandas as pd

import eda
from fake_data import make_fake_table, SimSpec
from table_io import load_table


def table(**kw):
    return load_table(make_fake_table(**kw))


def test_coverage_counts_missing_months():
    t = make_fake_table(countries=("BRA",), blank={("BRA", "2020-06-01"), ("BRA", "2020-07-01")})
    cov = eda.coverage(load_table(t)).iloc[0]
    assert cov["all_three_present"] == 70
    assert cov["longest_gap_months"] == 2


def test_sanity_finds_planted_unit_slip_and_zero():
    t = make_fake_table(countries=("BRA",))
    t.loc[30, "exporter_t"] *= 10          # unit slip
    t.loc[45, "ship_t"] = 0.0              # false zero
    s = eda.sanity(load_table(t))
    kinds = set(zip(s["record"], s["kind"]))
    assert ("exporter", "far above neighbours") in kinds
    assert ("ships", "zero") in kinds


def test_clean_fake_data_has_no_suspicious_months():
    assert len(eda.sanity(table(countries=("BRA", "AUS")))) == 0


def test_steady_share_passes_and_drifting_share_fails_the_F2_rule():
    spec = SimSpec(scales=(1.0, 0.75, 0.95), noise_levels=(0.05, 0.05, 0.05))
    t = make_fake_table(countries=("BRA",), spec=spec, seed=3)
    steady = eda.share_summary(load_table(t))
    assert steady[steady.record == "importer"]["passes_F2_rule"].iloc[0]
    ramp = np.linspace(1.0, 0.75, len(t))                  # share slides ~25%
    for col in ("china_aligned_t", "eu_aligned_t", "importer_aligned_total_t"):
        t[col] = t[col] * ramp
    drift = eda.share_summary(load_table(t))
    row = drift[drift.record == "importer"].iloc[0]
    assert not row["passes_F2_rule"]
    assert row["range_points"] > 7


def test_lag_scan_recovers_the_built_in_lags():
    spec = SimSpec(noise_levels=(0.05, 0.05, 0.05))
    t = load_table(make_fake_table(countries=("BRA",), spec=spec, seed=5, lag_china=2, lag_eu=0))
    best = eda.best_lags(eda.lag_scan(t), t).set_index("buyer")
    assert best.loc["china", "best_lag_yoy"] == 2
    assert best.loc["eu", "best_lag_yoy"] == 0
    assert best.loc["china", "lag_used_in_table"] == 2


def test_seasonality_is_large_on_levels_and_small_on_yoy():
    s = eda.seasonality(table(countries=("AUS",)))
    assert (s["seasonal_share_levels"] > s["seasonal_share_yoy"]).all()
    assert (s["seasonal_share_yoy"] < 0.15).all()


def test_error_behaviour_sees_persistence_and_wild_months():
    t = make_fake_table(countries=("BRA",), seed=9)
    t.loc[20, "exporter_t"] *= 4                           # one wild month
    e = eda.error_behaviour(load_table(t))
    assert (e["excess_kurtosis"] > 3).any()
    assert (e["share_extreme"] > 0).any()


def test_odd_one_out_names_the_planted_record():
    t = make_fake_table(countries=("BRA",), seed=11)
    t.loc[40, "exporter_t"] *= 3
    o = eda.odd_one_out(load_table(t))
    assert ((o["odd_record"] == "exporter") & (o["month"] == pd.Timestamp("2022-05-01").date())).any()


def test_catalog_has_one_row_per_country_with_sizes():
    c = eda.catalog(table())
    assert sorted(c["country"]) == ["AUS", "BRA", "LBR", "PER", "ZAF"]
    for col in ("usable_months_all_three", "importer_mean_share", "ships_share_range_pts",
                "importer_ratio_sd", "suspicious_months", "odd_one_out_months"):
        assert col in c.columns


def test_eda_survives_a_country_with_almost_no_data():
    t = make_fake_table(countries=("BRA", "LBR"))
    t.loc[(t["country"] == "LBR") & (t["month"] < "2024-06-01"),
          ["exporter_t", "ship_t", "china_aligned_t", "eu_aligned_t", "importer_aligned_total_t", "china_raw_t", "eu_raw_t"]] = np.nan
    df = load_table(t)
    eda.catalog(df); eda.lag_scan(df); eda.error_behaviour(df); eda.odd_one_out(df)   # no crash


def test_disagreement_ratio_tracks_planted_noise():
    easy = eda.disagreement_vs_signal(table(countries=("BRA",), spec=SimSpec(n_months=400, noise_levels=(0.1, 0.25, 0.4)), seed=2))
    hard = eda.disagreement_vs_signal(table(countries=("BRA",), spec=SimSpec(n_months=400, noise_levels=(1.0, 1.0, 1.0)), seed=2))
    # exporter 10% + importer 25% disagree by about sqrt(.01+.0625) = 0.27; at 100% each, about 1.41
    assert 0.15 < easy["importer_vs_exporter_ratio"].iloc[0] < 0.5
    assert 1.1 < hard["importer_vs_exporter_ratio"].iloc[0] < 1.8


def test_equal_noise_equivalent_recovers_equal_noise():
    t = table(countries=("BRA",), spec=SimSpec(n_months=600, noise_levels=(1.0, 1.0, 1.0)), seed=4)
    yc = eda.yoy_correlations(t).iloc[0]
    assert 0.8 < yc["equal_noise_equivalent"] < 1.25
    easy = eda.yoy_correlations(table(countries=("BRA",), seed=4)).iloc[0]
    assert easy["mean_pairwise_corr"] > 0.85


def test_aggregation_effect_is_near_independent_value_for_independent_noise():
    t = table(countries=("BRA",), spec=SimSpec(n_months=1800, noise_levels=(0.3, 0.3, 0.3), scales=(1, 1, 1)), seed=6)
    a = eda.aggregation_effect(t)
    assert (a["quarterly_over_monthly"].between(0.45, 0.75)).all()


def test_biggest_disagreements_puts_the_planted_month_first():
    t = make_fake_table(countries=("BRA",), seed=12)
    t.loc[50, "exporter_t"] *= 2.5
    b = eda.biggest_disagreements(load_table(t))
    assert b.iloc[0]["odd_record"] == "exporter"
    assert b.iloc[0]["month"] in (pd.Timestamp("2023-03-01").date(), pd.Timestamp("2024-03-01").date())


def test_single_source_months_are_kept_by_default():
    from table_io import country_yoy
    t = make_fake_table(countries=("LBR",), seed=1)
    t.loc[30:33, "flag"] = "single_source"
    df = load_table(t)
    kept = country_yoy(df, "LBR")
    dropped = country_yoy(df, "LBR", exclude_flags=("missing", "known_source_error", "single_source"))
    assert kept.n_excluded_by_flag == 0 and kept.n_yoy_rows == 60
    assert dropped.n_excluded_by_flag == 4 and dropped.n_yoy_rows < kept.n_yoy_rows
