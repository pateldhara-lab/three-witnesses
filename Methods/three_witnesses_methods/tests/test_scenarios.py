import numpy as np

from fake_data import SimSpec, simulate
from scenarios import (Scenario, simulate_scenario, run_scenario, summarise,
                       build_scenarios, TOL)


def test_no_damage_matches_baseline_generator():
    sc = Scenario("plain", "x")
    yoy, planted, _ = simulate_scenario(sc, np.random.default_rng(5))
    sim = simulate(SimSpec(), rng=np.random.default_rng(5))
    assert np.allclose(yoy, sim.records_yoy)
    assert np.allclose(planted, sim.planted)


def test_drift_ramp_hits_the_end_of_the_window():
    # with zero noise, the last month of a record that drifts -10% is 0.9 x truth
    sc = Scenario("d", "x", noise_levels=(0, 0, 0), drift=(0, 0, -0.10))
    rng = np.random.default_rng(1)
    yoy, _, true_yoy = simulate_scenario(sc, rng)
    assert np.allclose(yoy[:, 0], true_yoy)            # undrifted record = truth exactly
    assert not np.allclose(yoy[:, 2], true_yoy)        # drifted record is not


def test_dropped_month_costs_exactly_two_yoy_rows_in_all_records():
    sc = Scenario("drop", "x", drop_months=(30,))
    yoy, _, _ = simulate_scenario(sc, np.random.default_rng(2))
    bad = np.where(np.isnan(yoy).all(axis=1))[0]
    assert list(bad) == [30 - 12, 30]                  # that month and the same month a year later
    assert np.isfinite(yoy[~np.isnan(yoy).all(axis=1)]).all()


def test_gap_blanks_one_record_only():
    sc = Scenario("gap", "x", gap_months=(("importer", 20),))
    yoy, _, _ = simulate_scenario(sc, np.random.default_rng(3))
    assert np.isnan(yoy[[8, 20], 1]).all()
    assert np.isfinite(yoy[:, [0, 2]]).all()


def test_zero_and_x10_spikes_change_only_the_named_record_and_month():
    base, _, _ = simulate_scenario(Scenario("b", "x"), np.random.default_rng(4))
    z, _, _ = simulate_scenario(Scenario("z", "x", spikes=(("importer", 40, "zero"),)), np.random.default_rng(4))
    t, _, _ = simulate_scenario(Scenario("t", "x", spikes=(("exporter", 40, "x10"),)), np.random.default_rng(4))
    for other, col in ((z, 1), (t, 0)):
        diff = ~np.isclose(other, base)
        assert set(np.where(diff.any(axis=1))[0]) == {28, 40}     # month 40 and 40-12
        assert set(np.where(diff.any(axis=0))[0]) == {col}


def test_short_window_has_14_yoy_rows():
    yoy, _, _ = simulate_scenario(Scenario("s", "x", n_months=26), np.random.default_rng(6))
    assert yoy.shape == (14, 3)


def test_run_is_reproducible_and_counts_rows():
    sc = Scenario("r", "x", drop_months=(30,))
    a = run_scenario(sc, 4, seed=9, n_boot=30)
    b = run_scenario(sc, 4, seed=9, n_boot=30)
    assert np.array_equal(a["rel"], b["rel"], equal_nan=True)
    assert (a["n_rows"] == 58).all()


def test_too_few_rows_is_reported_not_crashed():
    sc = Scenario("tiny", "x", n_months=18)             # 6 yoy rows
    res = run_scenario(sc, 3, seed=1, n_boot=20)
    assert res["failed"].all()
    s = summarise(res)
    assert s["failed_share"] == 1.0


def test_summarise_keys_and_scenario_list_is_sane():
    sc = Scenario("k", "x", noise_levels=(0.5, 0.8, 1.6))
    s = summarise(run_scenario(sc, 6, seed=2, n_boot=40))
    for k in ("coverage", "bias", "wrong_call", "right_call", "holds", "checks", "flag_share"):
        assert k in s
    names = [x.name for x in build_scenarios()]
    assert len(names) == len(set(names)) == 18
    assert TOL["coverage_min"] == 0.80


def test_shift_janfeb_scales_and_extra_cargo_touch_only_what_they_name():
    base, _, _ = simulate_scenario(Scenario("b", "x"), np.random.default_rng(7))
    s, _, _ = simulate_scenario(Scenario("s", "x", shift=(0, 2, 0)), np.random.default_rng(7))
    assert np.isfinite(s[:, [0, 2]]).all() and np.allclose(s[:, [0, 2]], base[:, [0, 2]])
    assert np.isnan(s[:, 1]).sum() == 2                               # the first 2 months have no value

    j, _, _ = simulate_scenario(Scenario("j", "x", merge_janfeb=("importer",)), np.random.default_rng(7))
    assert np.isfinite(j[:, [0, 2]]).all()
    assert np.isnan(j[:, 1]).sum() == 5                               # blank Januaries cost 5 yoy rows; the doubled Februaries stay in

    x, _, _ = simulate_scenario(Scenario("x", "x", extra_cargo=0.5), np.random.default_rng(7))
    assert np.allclose(x[:, :2], base[:, :2]) and not np.allclose(x[:, 2], base[:, 2])

    sc, _, _ = simulate_scenario(Scenario("sc", "x", noise_levels=(0, 0, 0), scales=(1, 2, 1)),
                                 np.random.default_rng(7))
    assert np.allclose(sc[:, 1], 2 * sc[:, 0])                        # importer sees twice the exporter
