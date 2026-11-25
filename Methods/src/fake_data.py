"""Fake-data generator: a known 'truth' plus three records with PLANTED noise.

Because we choose the noise, we know the right answer and can check whether
triple collocation finds it (Stage 4A).

How the planted level is defined (important, write it in the report):
    planted level r_i = sd(record i's noise) / (a_i * sd(true change))
measured on YEAR-OVER-YEAR CHANGES, the scale the method is run on.  So
r = 0.25 means "this record's noise is a quarter the size of the real
year-over-year movement".  Noise is added to monthly levels as white noise; a
year-over-year difference of white noise is sqrt(2) times bigger, which the
code corrects for so the planted level really is the level on the changes.

Optional error_corr makes two records share part of their mistakes (e.g.
exporter and importer customs working from the same shipping papers).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from triple_collocation import yoy_changes

NAMES = ("exporter", "importer", "ships")


@dataclass(frozen=True)
class SimSpec:
    n_months: int = 72                          # real three-way window: Jan 2019 - Dec 2024
    noise_levels: tuple = (0.10, 0.25, 0.40)    # planted noise/signal per record
    scales: tuple = (1.0, 1.0, 1.0)             # steady share, e.g. 0.72 = sees 72%
    offsets: tuple = (0.0, 0.0, 0.0)
    error_corr: float = 0.0                     # shared mistakes between correlated_pair
    correlated_pair: tuple = (0, 1)
    base_level: float = 100.0                   # arbitrary units (think Mt per month)
    seasonal_amp: float = 10.0
    ar_phi: float = 0.8                         # persistence of the true series
    ar_sd: float = 10.0


@dataclass
class SimData:
    spec: SimSpec
    true_levels: np.ndarray         # (n,)
    records_levels: np.ndarray      # (n, 3)
    noise_levels_added: np.ndarray  # (n, 3) the actual noise drawn, level space
    true_yoy: np.ndarray            # (n-12,)
    records_yoy: np.ndarray         # (n-12, 3)  <- feed this to triple_collocation
    planted: np.ndarray             # (3,) planted noise/signal on yoy scale


def _true_series(spec: SimSpec, rng) -> np.ndarray:
    n, burn = spec.n_months, 100
    innov_sd = spec.ar_sd * np.sqrt(1 - spec.ar_phi ** 2)
    e = rng.standard_normal(n + burn) * innov_sd
    ar = np.empty(n + burn)
    ar[0] = e[0]
    for t in range(1, n + burn):
        ar[t] = spec.ar_phi * ar[t - 1] + e[t]
    ar = ar[burn:]
    season = spec.seasonal_amp * np.sin(2 * np.pi * (np.arange(n) % 12) / 12)
    return spec.base_level + season + ar


def simulate(spec: SimSpec = SimSpec(), rng=None, seed=None) -> SimData:
    """Make one fake dataset.  Pass seed (or an rng) for reproducibility."""
    if spec.n_months <= 15:
        raise ValueError("n_months must exceed 15")
    if min(spec.noise_levels) < 0:
        raise ValueError("noise levels must be >= 0")
    if not -1 < spec.error_corr < 1:
        raise ValueError("error_corr must be in (-1, 1)")
    if rng is None:
        rng = np.random.default_rng(seed)

    truth = _true_series(spec, rng)
    true_yoy = yoy_changes(truth)
    sd_true = true_yoy.std(ddof=1)

    r = np.asarray(spec.noise_levels, float)
    a = np.asarray(spec.scales, float)
    b = np.asarray(spec.offsets, float)
    sd_level = r * np.abs(a) * sd_true / np.sqrt(2.0)  # sqrt(2): yoy differencing

    z = rng.standard_normal((spec.n_months, 3))
    if spec.error_corr != 0.0:
        i, j = spec.correlated_pair
        z[:, j] = spec.error_corr * z[:, i] + np.sqrt(1 - spec.error_corr ** 2) * z[:, j]
    noise = z * sd_level

    records = a * truth[:, None] + b + noise
    return SimData(spec, truth, records, noise, true_yoy, yoy_changes(records), r.copy())


def realized_noise_levels(sim: SimData) -> np.ndarray:
    """Noise/signal actually present in this dataset on the yoy scale
    (differs slightly from planted because of random draws)."""
    a = np.abs(np.asarray(sim.spec.scales, float))
    noise_yoy = yoy_changes(sim.noise_levels_added)
    return noise_yoy.std(axis=0, ddof=1) / (a * sim.true_yoy.std(ddof=1))


def make_fake_table(countries=("BRA", "AUS", "ZAF", "PER", "LBR"), spec: SimSpec = SimSpec(),
                    seed: int = 0, lag_china: int = 1, lag_eu: int = 0,
                    china_share: float = 0.8, flags=None, blank=None):
    """A fake table in Ben's 13-column layout, to test the pipeline end to end.

    flags: {(country, 'YYYY-MM-01'): 'known_source_error'}   blank: set of (country, month)
    to leave empty in all witnesses.  Needs pandas (imported here, not at top, so
    the core generator stays numpy-only).
    """
    import pandas as pd
    flags, blank = flags or {}, blank or set()
    months = pd.date_range("2019-01-01", periods=spec.n_months, freq="MS")
    rng = np.random.default_rng(seed)
    frames = []
    for c in countries:
        sim = simulate(spec, rng=rng)
        exp, imp, shp = sim.records_levels.T
        china_al, eu_al = imp * china_share, imp * (1 - china_share)
        df = pd.DataFrame({
            "country": c, "month": months,
            "exporter_t": exp, "ship_t": shp, "ship_extra_ports_t": np.nan,
            "china_raw_t": pd.Series(china_al).shift(lag_china).to_numpy(),  # raw = aligned, later
            "eu_raw_t": pd.Series(eu_al).shift(lag_eu).to_numpy(),
            "lag_china": lag_china, "lag_eu": lag_eu,
            "china_aligned_t": china_al, "eu_aligned_t": eu_al,
            "importer_aligned_total_t": imp, "flag": "",
        })
        for (fc, fm), f in flags.items():
            if fc == c:
                df.loc[df["month"] == pd.Timestamp(fm), "flag"] = f
        for (bc, bm) in blank:
            if bc == c:
                m = df["month"] == pd.Timestamp(bm)
                df.loc[m, ["exporter_t", "ship_t", "china_aligned_t", "eu_aligned_t",
                           "importer_aligned_total_t"]] = np.nan
                df.loc[m, "flag"] = "missing"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
