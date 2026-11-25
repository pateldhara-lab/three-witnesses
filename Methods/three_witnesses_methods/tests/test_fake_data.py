import numpy as np
import pytest

from fake_data import SimSpec, simulate, realized_noise_levels
from triple_collocation import triple_collocation


def test_shapes_for_real_sample_size():
    sim = simulate(SimSpec(), seed=0)
    assert sim.records_levels.shape == (72, 3)
    assert sim.records_yoy.shape == (60, 3)   # 72 months -> 60 after yoy


def test_same_seed_same_data_different_seed_different_data():
    a, b, c = simulate(seed=5), simulate(seed=5), simulate(seed=6)
    np.testing.assert_array_equal(a.records_levels, b.records_levels)
    assert not np.array_equal(a.records_levels, c.records_levels)


def test_planted_levels_are_what_the_data_contain():
    spec = SimSpec(n_months=200_000, noise_levels=(0.10, 0.25, 0.40))
    sim = simulate(spec, seed=1)
    np.testing.assert_allclose(realized_noise_levels(sim), [0.10, 0.25, 0.40], rtol=0.03)


def test_scales_do_not_change_the_planted_meaning():
    spec = SimSpec(n_months=200_000, scales=(1.0, 0.7, 0.9), offsets=(0, 50, -20))
    sim = simulate(spec, seed=2)
    np.testing.assert_allclose(realized_noise_levels(sim), sim.planted, rtol=0.03)


def test_end_to_end_recovery_with_lots_of_data():
    # High noise levels so the check is sharp; the 10/25/40 question at n=60
    # is what the simulation study (not this unit test) answers.
    spec = SimSpec(n_months=100_000, noise_levels=(0.3, 0.5, 0.7), scales=(1.0, 0.72, 0.9))
    sim = simulate(spec, seed=3)
    res = triple_collocation(sim.records_yoy)
    assert res.ok
    np.testing.assert_allclose(res.rel_noise, [0.3, 0.5, 0.7], rtol=0.05)


def test_error_corr_creates_correlated_noise_only_in_the_chosen_pair():
    spec = SimSpec(n_months=100_000, noise_levels=(0.3, 0.3, 0.3), error_corr=0.5)
    sim = simulate(spec, seed=4)
    C = np.corrcoef(sim.noise_levels_added, rowvar=False)
    assert C[0, 1] == pytest.approx(0.5, abs=0.02)
    assert abs(C[0, 2]) < 0.02 and abs(C[1, 2]) < 0.02


def test_shared_mistakes_bias_tc_in_the_expected_direction():
    # exporter & importer share mistakes -> TC makes them look BETTER than they
    # are and the independent record (ships) look WORSE.
    spec = SimSpec(n_months=200_000, noise_levels=(0.4, 0.4, 0.4), error_corr=0.5)
    res = triple_collocation(simulate(spec, seed=5).records_yoy)
    assert res.rel_noise[0] < 0.4 and res.rel_noise[1] < 0.4
    assert res.rel_noise[2] > 0.4


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        simulate(SimSpec(n_months=10))
    with pytest.raises(ValueError):
        simulate(SimSpec(error_corr=1.0))
    with pytest.raises(ValueError):
        simulate(SimSpec(noise_levels=(-0.1, 0.2, 0.3)))
