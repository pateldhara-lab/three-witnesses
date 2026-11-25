import numpy as np
import pandas as pd

from fake_data import make_fake_table
from table_io import load_table
from stage2_plant import (plant_errors, score_flags, reference_rule, run_once, RECORD_COLS, KINDS,
                          FIT_END)


def table():
    return load_table(make_fake_table(seed=3))


def test_planting_counts_kinds_and_changes_only_planted_cells():
    df = table()
    out, truth = plant_errors(df, np.random.default_rng(1), n_per_kind=3)
    assert set(truth.kind) == set(KINDS)
    assert (truth[truth.kind != "janfeb"].groupby("kind").size() == 3).all()
    cols = list(RECORD_COLS.values())
    changed = ~np.isclose(out[cols].to_numpy(float), df[cols].to_numpy(float), equal_nan=True)
    changed_cm = set(map(tuple, out.loc[changed.any(axis=1), ["country", "month"]].to_numpy()))
    planted_cm = set(map(tuple, truth[["country", "month"]].to_numpy()))
    assert changed_cm == planted_cm


def test_planting_stays_in_test_years_and_never_adjacent():
    df = table()
    _, truth = plant_errors(df, np.random.default_rng(2), n_per_kind=3)
    assert (truth.month > FIT_END).all()
    for c, g in truth[truth.kind != "janfeb"].groupby("country"):
        m = sorted(g.month)
        assert all((b - a).days > 31 for a, b in zip(m, m[1:]))


def test_score_flags_exact_on_a_toy_case():
    df = pd.DataFrame({"country": "X", "month": pd.date_range("2023-01-01", periods=10, freq="MS")})
    truth = pd.DataFrame({"country": ["X", "X"], "month": pd.to_datetime(["2023-02-01", "2023-05-01"]),
                          "record": ["importer"] * 2, "kind": ["zero", "x10"]})
    flagged = {("X", pd.Timestamp("2023-02-01")), ("X", pd.Timestamp("2023-07-01"))}
    s = score_flags(flagged, truth, df)
    assert s["catch_rate"] == 0.5 and s["n_planted"] == 2 and s["n_clean"] == 8
    assert s["false_alarm_rate"] == 1 / 8
    assert s["catch_by_kind"]["zero"] == 1.0 and s["catch_by_kind"]["x10"] == 0.0


def test_reference_rule_runs_end_to_end_and_catches_big_slips():
    df = table()
    s, truth = run_once(df, reference_rule, np.random.default_rng(4), n_per_kind=3)
    assert 0 <= s["false_alarm_rate"] <= 0.5
    big = {k: v for k, v in s["catch_by_kind"].items() if k in ("x10", "unit_swap", "zero")}
    assert np.nanmean(list(big.values())) > 0.5      # the stand-in rule sees large slips on clean fake data


def test_run_is_reproducible():
    df = table()
    a, _ = run_once(df, reference_rule, np.random.default_rng(5))
    b, _ = run_once(df, reference_rule, np.random.default_rng(5))
    assert a == b
