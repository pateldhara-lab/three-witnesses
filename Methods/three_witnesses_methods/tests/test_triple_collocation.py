import numpy as np
import pytest

from triple_collocation import (triple_collocation, triple_collocation_from_cov,
                                yoy_changes)


def exact_cov(a, var_truth, noise_var):
    """Population covariance of records = a_i*T + noise_i (independent noise)."""
    a = np.asarray(a, float)
    C = np.outer(a, a) * var_truth
    C[np.diag_indices(3)] += noise_var
    return C


def test_formula_is_exact_on_population_covariance():
    # scales differ on purpose: an importer that sees 70%, ships at 90%
    C = exact_cov([1.0, 0.7, 0.9], var_truth=4.0, noise_var=[0.25, 1.0, 2.0])
    res = triple_collocation_from_cov(C)
    assert res.ok
    np.testing.assert_allclose(res.noise_var, [0.25, 1.0, 2.0])
    np.testing.assert_allclose(res.signal_var, np.array([1.0, 0.49, 0.81]) * 4.0)
    expected_rel = np.sqrt(np.array([0.25, 1.0, 2.0]) / (np.array([1.0, 0.49, 0.81]) * 4.0))
    np.testing.assert_allclose(res.rel_noise, expected_rel)


def test_matches_week1_plan_wording():
    # noise = total - (cov_ij * cov_ik) / cov_jk
    C = exact_cov([1, 1, 1], 1.0, [0.1, 0.2, 0.3])
    res = triple_collocation_from_cov(C)
    manual = C[0, 0] - C[0, 1] * C[0, 2] / C[1, 2]
    assert res.noise_var[0] == pytest.approx(manual)


def test_recovers_planted_noise_with_lots_of_data():
    rng = np.random.default_rng(1)
    n = 400_000
    truth = rng.standard_normal(n)
    levels = np.array([0.3, 0.5, 0.7])
    X = truth[:, None] + rng.standard_normal((n, 3)) * levels
    res = triple_collocation(X)
    assert res.ok
    np.testing.assert_allclose(res.rel_noise, levels, rtol=0.03)


def test_scale_and_offset_do_not_change_relative_noise():
    rng = np.random.default_rng(2)
    n = 50_000
    truth = rng.standard_normal(n)
    X = truth[:, None] + rng.standard_normal((n, 3)) * [0.3, 0.5, 0.7]
    base = triple_collocation(X).rel_noise
    Y = X.copy()
    Y[:, 1] = Y[:, 1] * 0.7 + 12.0   # steady 70% share plus an offset
    Y[:, 2] = Y[:, 2] * 3.0 - 5.0
    np.testing.assert_allclose(triple_collocation(Y).rel_noise, base, rtol=1e-9)


def test_unrelated_records_are_flagged_not_hidden():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 3))        # no shared truth at all
    res = triple_collocation(X)
    assert res.status == "assumptions_broken"
    assert len(res.reasons) > 0


def test_negative_noise_variance_is_reported():
    # record 0 is *more* tightly tied to 1 and 2 than they are to each other
    C = np.array([[1.0, 0.9, 0.9],
                  [0.9, 1.0, 0.5],
                  [0.9, 0.5, 1.0]])
    res = triple_collocation_from_cov(C)
    assert res.status == "assumptions_broken"
    assert any(r.startswith("negative_noise_variance") for r in res.reasons)
    assert res.noise_var.min() < 0
    assert np.isnan(res.rel_noise[res.noise_var < 0]).all()


def test_nan_rows_are_dropped_and_counted():
    rng = np.random.default_rng(4)
    truth = rng.standard_normal(200)
    X = truth[:, None] + rng.standard_normal((200, 3)) * 0.4
    X[5, 1] = np.nan
    X[17, 0] = np.nan
    res = triple_collocation(X)
    assert res.n_dropped == 2 and res.n_used == 198


def test_input_validation():
    with pytest.raises(ValueError):
        triple_collocation(np.zeros((50, 2)))
    with pytest.raises(ValueError):
        triple_collocation(np.zeros((5, 3)))


def test_yoy_changes_shape_and_values():
    x = np.arange(72, dtype=float)
    d = yoy_changes(x)
    assert d.shape == (60,)
    assert (d == 12).all()
    with pytest.raises(ValueError):
        yoy_changes(np.arange(12))
