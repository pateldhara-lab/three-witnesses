"""Reads Ben's side-by-side table and feeds it to triple collocation.

The table layout (agreed Oct 2026; one row per country and month):

     1 country                  BRA / AUS / ZAF / PER / LBR
     2 month                    first of the month the ore LEFT
     3 exporter_t               exporter's own tonnes            -> witness 1
     4 ship_t                   PortWatch tonnes, same ports     -> witness 3
     5 ship_extra_ports_t       ore-port tonnes the exporter record misses (Cape Lambert); not used by TC
     6 china_raw_t              China customs, month China recorded it
     7 eu_raw_t                 EU customs, as published
     8 lag_china                months China was moved back
     9 lag_eu                   months the EU was moved back
    10 china_aligned_t          china_raw_t from `lag_china` months later
    11 eu_aligned_t             same for the EU
    12 importer_aligned_total_t sum of 10 and 11                -> witness 2
    13 flag                     missing | known_source_error | single_source (blank = fine)

Triple collocation uses columns 3, 12 and 4 only.  Everything else is kept for
checks.  Extra columns (e.g. source / snapshot date) are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from triple_collocation import triple_collocation, TCResult, yoy_changes

REQUIRED_COLUMNS = (
    "country", "month", "exporter_t", "ship_t", "ship_extra_ports_t",
    "china_raw_t", "eu_raw_t", "lag_china", "lag_eu",
    "china_aligned_t", "eu_aligned_t", "importer_aligned_total_t", "flag",
)
# order matters: it is the column order given to triple collocation
TC_COLUMNS = ("exporter_t", "importer_aligned_total_t", "ship_t")
TC_NAMES = ("exporter", "importer", "ships")
# flags that mean "do not use this month".  single_source (Liberia: the figure has one printed
# copy that is unconfirmed) is KEPT by default, because the value exists; pass
# exclude_flags=("missing", "known_source_error", "single_source") for a sensitivity run.
DEFAULT_EXCLUDE_FLAGS = ("missing", "known_source_error")
VALID_FLAGS = {"missing", "known_source_error", "single_source", "unit_slip_corrected"}
NUMERIC_COLUMNS = tuple(c for c in REQUIRED_COLUMNS if c not in ("country", "month", "flag"))


def load_table(source) -> pd.DataFrame:
    """Read a CSV path or DataFrame, check it, return a clean copy.

    Raises ValueError with a plain-language message if columns are missing,
    months are not first-of-month, a country-month appears twice, or a flag has
    an unknown value.  Warns (returns in df.attrs['warnings']) when column 12
    does not equal column 10 + column 11.
    """
    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"table is missing columns: {missing}")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        raise ValueError("some `month` values could not be read as dates")
    if not (df["month"].dt.day == 1).all():
        raise ValueError("`month` must be the first of the month (e.g. 2021-01-01)")
    for c in NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["country"] = df["country"].astype(str).str.strip().str.upper()
    df["flag"] = df["flag"].astype("string").fillna("").astype(str).str.strip().str.lower()

    if df.duplicated(["country", "month"]).any():
        d = df[df.duplicated(["country", "month"], keep=False)][["country", "month"]]
        raise ValueError(f"duplicate country-month rows, e.g.\n{d.head()}")
    bad = {t for f in df["flag"] for t in _tokens(f)} - VALID_FLAGS
    if bad:
        raise ValueError(f"unknown flag values: {sorted(bad)}; allowed: {sorted(VALID_FLAGS)}")

    warnings = []
    both = df["china_aligned_t"].notna() & df["eu_aligned_t"].notna()
    diff = (df.loc[both, "china_aligned_t"] + df.loc[both, "eu_aligned_t"]
            - df.loc[both, "importer_aligned_total_t"]).abs()
    n_bad = int((diff > 1e-6 * (1 + df.loc[both, "importer_aligned_total_t"].abs())).sum())
    if n_bad:
        warnings.append(f"importer_aligned_total_t != china_aligned_t + eu_aligned_t in {n_bad} rows")
    one_missing = df["china_aligned_t"].isna() ^ df["eu_aligned_t"].isna()
    n_partial = int((one_missing & df["importer_aligned_total_t"].notna()).sum())
    if n_partial:
        warnings.append(f"{n_partial} rows have an importer total although China or EU is blank "
                        "(a total from only one buyer is not comparable)")
    df.attrs["warnings"] = warnings
    return df.sort_values(["country", "month"]).reset_index(drop=True)


def _tokens(flag: str):
    return [t.strip() for t in flag.replace(";", ",").split(",") if t.strip()]


@dataclass
class CountryInput:
    country: str
    months: pd.DatetimeIndex     # month of the later value of each yoy change
    yoy: np.ndarray              # (n, 3) in TC_COLUMNS order, complete rows only
    n_months_in_window: int
    n_excluded_by_flag: int
    n_yoy_rows: int


def country_yoy(df: pd.DataFrame, country: str, start="2019-01-01", end="2024-12-01",
                exclude_flags=DEFAULT_EXCLUDE_FLAGS, lag: int = 12) -> CountryInput:
    """Year-over-year changes for one country, inside the three-way window.

    Steps: put the country on a complete monthly calendar (a missing month
    becomes NaN, so 'a year earlier' is always truly 12 months earlier) ->
    blank out flagged months in ALL three witnesses -> yoy change -> drop rows
    with any NaN.  Dropping one month removes two yoy rows (that month and the
    same month a year later), which is the honest cost.
    """
    d = df[df["country"] == country.upper()]
    if d.empty:
        raise ValueError(f"no rows for country {country}")
    cal = pd.date_range(start, end, freq="MS")
    d = d.set_index("month").reindex(cal)
    vals = d[list(TC_COLUMNS)].copy()
    flagged = d["flag"].astype("object").fillna("").map(lambda f: any(t in exclude_flags for t in _tokens(f)))
    vals.loc[flagged, :] = np.nan
    yoy = yoy_changes(vals.to_numpy(float), lag)
    months = cal[lag:]
    keep = np.isfinite(yoy).all(axis=1)
    return CountryInput(country.upper(), months[keep], yoy[keep], len(cal),
                        int(flagged.sum()), int(keep.sum()))


def run_by_country(df: pd.DataFrame, countries=None, min_rows: int = 24, **kwargs):
    """Triple collocation for each country.  Returns {country: (CountryInput, TCResult|None, note)}.

    Countries with fewer than `min_rows` usable yoy rows are not run; the note
    says why (never silently skipped).
    """
    out = {}
    for c in (countries or sorted(df["country"].unique())):
        ci = country_yoy(df, c, **kwargs)
        if ci.n_yoy_rows < min_rows:
            out[c] = (ci, None, f"only {ci.n_yoy_rows} usable yoy rows (< {min_rows}); not run")
        else:
            out[c] = (ci, triple_collocation(ci.yoy, TC_NAMES), "")
    return out
