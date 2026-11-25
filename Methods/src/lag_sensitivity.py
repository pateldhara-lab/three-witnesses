"""Lag sensitivity on the real table: how much does each country's answer move when the
importer lags are set differently?

Why: the scenario study (G1, G2) showed that an importer record that is 1 month off
already pushes its noise estimate about 66% too high, and 3 months off about 225% too
high.  The lags in Ben's table were fitted on weakly-identified correlations (EDA table
4), so before trusting any real-country result we need to see how much it moves.

How the table builds the importer record (checked: reproduces columns 10-12 exactly):
    china_aligned(month) = china_raw(month + lag_china)      (raw value from `lag` months later)
    eu_aligned(month)    = eu_raw(month + lag_eu)
    importer_aligned_total = china_aligned + eu_aligned      (blank if either is blank)
This module rebuilds that for ANY pair of lags, then runs triple collocation exactly as
table_io.country_yoy / run_by_country do (same window, same flag rule, same yoy).

EXPLORATORY: real data, labelled illustration, not a result (the scenario bar is for the
fake-data study).  The question and the reading rule are fixed in the script header
before it was run:  an estimate is called STABLE if the importer noise/signal moves by
no more than 25% (the scenario bar's bias limit) across lags one month either side of
the table's lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from table_io import TC_COLUMNS, TC_NAMES, DEFAULT_EXCLUDE_FLAGS, _tokens
from triple_collocation import triple_collocation, yoy_changes
from study import bootstrap_ranges


def importer_series(d: pd.DataFrame, lag_china: int, lag_eu: int) -> pd.Series:
    """Importer total on the full monthly calendar of `d` (a one-country table indexed by month)."""
    cal = pd.date_range(d.index.min(), d.index.max(), freq="MS")
    d = d.reindex(cal)
    china = d["china_raw_t"].shift(-int(lag_china))
    eu = d["eu_raw_t"].shift(-int(lag_eu))
    return china + eu                       # NaN if either buyer is blank, as in the table


def run_lags(df: pd.DataFrame, country: str, lag_china: int, lag_eu: int,
             start="2019-01-01", end="2024-12-01", n_boot=500, block_len=6, seed=20261110,
             exclude_flags=DEFAULT_EXCLUDE_FLAGS):
    d = df[df["country"] == country.upper()].set_index("month").sort_index()
    imp_full = importer_series(d, lag_china, lag_eu)
    cal = pd.date_range(start, end, freq="MS")
    vals = pd.DataFrame({
        "exporter_t": d["exporter_t"].reindex(cal),
        "importer": imp_full.reindex(cal),
        "ship_t": d["ship_t"].reindex(cal)}).to_numpy(float).copy()
    flags = d["flag"].astype("object").fillna("").reindex(cal).fillna("")
    bad = flags.map(lambda f: any(t in exclude_flags for t in _tokens(f))).to_numpy()
    vals[bad, :] = np.nan
    yoy = yoy_changes(vals)
    yoy = yoy[np.isfinite(yoy).all(axis=1)]
    row = {"country": country.upper(), "lag_china": lag_china, "lag_eu": lag_eu, "n_rows": len(yoy)}
    for nm in TC_NAMES:
        row[f"{nm}_rel"] = row[f"{nm}_lo"] = row[f"{nm}_hi"] = np.nan
    row.update(corr_imp_exp=np.nan, corr_ships_exp=np.nan, corr_imp_ships=np.nan, reasons="")
    if len(yoy) < 24:
        row.update(status="too_few_rows")
        return row
    res = triple_collocation(yoy, TC_NAMES)
    lo, hi, _, _ = bootstrap_ranges(yoy, n_boot, block_len, np.random.default_rng(seed))
    c = np.corrcoef(yoy, rowvar=False)
    row.update(status=res.status, reasons=";".join(res.reasons),
               corr_imp_exp=c[1, 0], corr_ships_exp=c[2, 0], corr_imp_ships=c[1, 2])
    for i, nm in enumerate(TC_NAMES):
        row[f"{nm}_rel"], row[f"{nm}_lo"], row[f"{nm}_hi"] = res.rel_noise[i], lo[i], hi[i]
    return row


def grid(df, country: str, china_lags, eu_lags, **kw) -> pd.DataFrame:
    return pd.DataFrame([run_lags(df, country, lc, le, **kw) for lc in china_lags for le in eu_lags])


def stable_verdict(rows: pd.DataFrame, base_china: int, base_eu: int, limit: float = 0.25):
    """Importer noise/signal at the table's lags vs lags 1 month either side (both buyers moved together)."""
    base = rows[(rows.lag_china == base_china) & (rows.lag_eu == base_eu)]
    if base.empty or not np.isfinite(base.importer_rel.iloc[0]):
        return "no_base_estimate", np.nan
    b = base.importer_rel.iloc[0]
    near = rows[(rows.lag_china == base_china + 1) & (rows.lag_eu == base_eu + 1) |
                (rows.lag_china == max(base_china - 1, 0)) & (rows.lag_eu == max(base_eu - 1, 0))]
    near = near[np.isfinite(near.importer_rel)]
    if near.empty:
        return "no_neighbour_estimate", np.nan
    worst = float(np.max(np.abs(near.importer_rel / b - 1)))
    return ("STABLE" if worst <= limit else "NOT STABLE"), worst
