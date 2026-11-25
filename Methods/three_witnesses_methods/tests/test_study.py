import numpy as np
import pytest

from fake_data import SimSpec, simulate
from study import (block_bootstrap_indices, cov_batch, tc_batch, bootstrap_ranges,
                   run_study, evaluate_bar, DEFAULT_BAR)
from triple_collocation import triple_collocation


def test_block_indices_shape_range_and_contiguous_blocks():
    rng = np.random.default_rng(0)
    idx = block_bootstrap_indices(60, 6, 200, rng)
    assert idx.shape == (200, 60)
    assert idx.min() >= 0 and idx.max() <= 59
    first_block = idx[:, :6]                       # each block = 6 consecutive rows
    assert (np.diff(first_block, axis=1) == 1).all()


def test_block_indices_uneven_length_is_trimmed_and_bad_block_rejected():
    rng = np.random.default_rng(1)
    assert block_bootstrap_indices(58, 6, 10, rng).shape == (10, 58)
    with pytest.raises(ValueError):
        block_bootstrap_indices(10, 11, 5, rng)


def test_batch_formula_equals_single_dataset_formula():
    for s in range(5):
        X = simulate(SimSpec(), seed=s).records_yoy
        single = triple_collocation(X)
        noise, signal = tc_batch(cov_batch(X[None, :, :]))
        np.testing.assert_allclose(noise[0], single.noise_var, rtol=1e-9)
        np.testing.assert_allclose(signal[0], single.signal_var, rtol=1e-9)


def test_bootstrap_range_is_narrow_and_contains_truth_with_lots_of_data():
    sim = simulate(SimSpec(n_months=3000, noise_levels=(0.3, 0.5, 0.7)), seed=2)
    lo, hi, _, unusable = bootstrap_ranges(sim.records_yoy, 300, 6, np.random.default_rng(3))
    truth = np.array([0.3, 0.5, 0.7])
    assert ((lo <= truth) & (truth <= hi)).all()
    assert ((hi - lo) < 0.15).all()
    assert (unusable == 0).all()


def test_study_is_reproducible_from_its_seed():
    a = run_study(4, 11, SimSpec(), n_boot=30)
    b = run_study(4, 11, SimSpec(), n_boot=30)
    np.testing.assert_array_equal(a["noise_var"], b["noise_var"])
    np.testing.assert_array_equal(a["lo"], b["lo"])
    c = run_study(4, 12, SimSpec(), n_boot=30)
    assert not np.array_equal(a["noise_var"], c["noise_var"])


def _fake_results(ships_noisiest_share, covered_share):
    """Hand-built study results, to test the scoring logic only."""
    R = 100
    planted = np.array([0.10, 0.25, 0.40])
    nv = np.tile([0.01, 0.06, 0.16], (R, 1)).astype(float)
    n_bad = int(round((1 - ships_noisiest_share) * R))
    nv[:n_bad] = [0.01, 0.16, 0.06]                      # ships NOT noisiest in these runs
    lo = np.tile(planted - 0.05, (R, 1))
    hi = np.tile(planted + 0.05, (R, 1))
    lo[:, 0], hi[:, 0] = 0.0, 0.15                       # exporter range tops out below ships'
    lo[:, 2], hi[:, 2] = 0.30, 0.50
    n_out = int(round((1 - covered_share) * R))
    lo[:n_out, 1], hi[:n_out, 1] = 0.5, 0.6              # importer range misses planted
    return {"planted": planted, "noise_var": nv, "lo": lo, "hi": hi,
            "flagged": np.zeros(R, bool), "reasons": [()] * R}


def test_scoring_marks_pass_and_fail_correctly():
    rows, _ = evaluate_bar(_fake_results(0.95, 0.90), DEFAULT_BAR)
    byitem = {r[0]: r[4] for r in rows}
    assert byitem[1] and byitem["3a"] and byitem[4] and byitem[5]
    rows, _ = evaluate_bar(_fake_results(0.50, 0.50), DEFAULT_BAR)
    byitem = {r[0]: r[4] for r in rows}
    assert not byitem[1] and not byitem["3a"]


def test_flag_share_threshold_applies():
    res = _fake_results(0.95, 0.90)
    res["flagged"][:30] = True
    res["reasons"] = [("negative_noise_variance:exporter",)] * 30 + [()] * 70
    rows, d = evaluate_bar(res, DEFAULT_BAR)
    assert not {r[0]: r[4] for r in rows}[5]
    assert d["exporter_only_share"] == pytest.approx(0.30)
