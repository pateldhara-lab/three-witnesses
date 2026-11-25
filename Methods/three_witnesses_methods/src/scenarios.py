"""Realistic-error scenarios for the Stage 4A study (built from the EDA error catalog).

The first study (study.py / RECOVERY_BAR.md) was the BASELINE: independent mistakes, a
steady share, a full 72 months, noise 10/25/40%.  The EDA of the real table
(results/eda/) showed the real records are far noisier and messier than that:
  * year-over-year correlations of 0.36-0.80 (the baseline fake data had ~0.95),
  * Liberia: only 26 months, 14 usable year-over-year rows,
  * ship share drifting (ZAF +7 pts, LBR -23 pts, PER -19 pts) and importer share
    drifting (LBR -11, PER -13),
  * a false zero (Peru importer, June 2020), months missing (BRA/AUS/ZAF/PER),
  * shared mistakes between exporter and importer customs are plausible (same papers).

A scenario = the baseline generator (fake_data.simulate) plus a list of damages applied
afterwards.  Planted noise is always the INDEPENDENT noise on the year-over-year scale,
exactly as in the baseline; drift, spikes and gaps are extra damage the method has to
survive, and the study reports how far the estimates move.

Damages (all optional, all in SCENARIO below):
  drift        per record, total change in the record's scale over the window, e.g.
               -0.09 = the record sees 9% less of the truth at the end than at the start
               (linear ramp).  Models a drifting share.
  spikes       list of (record, month_index, kind) with kind 'zero' or 'x10'
               (false zero, unit slip).  Month index counts from 0.
  drop_months  month indices blanked in ALL three records, as the pipeline does for months
               flagged 'missing' / 'known_source_error'.  Costs two yoy rows per month.
  gap_months   {record: [month indices]} months blank in ONE record only (unflagged gap).
  scales       steady share each record sees (e.g. importer 1.46 = sees 146% of the exporter).
  shift        per record, months by which the record is wrongly timed (positive = recorded
               that many months late), e.g. an importer lag set at 3 months when the voyage is ~3 weeks.
  merge_janfeb records whose January is blank and whose February holds Jan+Feb, every year (China file).
  extra_cargo  relative size of a persistent non-ore component added to the ships record only
               (coal, grain in mixed ports): sd of the extra = this fraction of the sd of the true levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from fake_data import SimSpec, simulate, NAMES
from triple_collocation import triple_collocation, yoy_changes
from study import bootstrap_ranges

REC = {n: i for i, n in enumerate(NAMES)}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    noise_levels: tuple = (0.10, 0.25, 0.40)   # independent noise/signal on yoy scale
    n_months: int = 72
    error_corr: float = 0.0
    correlated_pair: tuple = (0, 1)
    drift: tuple = (0.0, 0.0, 0.0)             # exporter, importer, ships
    spikes: tuple = ()                         # ((record, month_index, 'zero'|'x10'), ...)
    drop_months: tuple = ()
    gap_months: tuple = ()                     # ((record, month_index), ...)
    scales: tuple = (1.0, 1.0, 1.0)
    shift: tuple = (0, 0, 0)
    merge_janfeb: tuple = ()                   # record names, e.g. ("importer",)
    extra_cargo: float = 0.0


def simulate_scenario(sc: Scenario, rng):
    """One fake dataset with the scenario's damages.

    Returns (yoy (n,3) with NaN where unusable, planted (3,), true_yoy).
    """
    spec = SimSpec(n_months=sc.n_months, noise_levels=sc.noise_levels, scales=sc.scales,
                   error_corr=sc.error_corr, correlated_pair=sc.correlated_pair)
    sim = simulate(spec, rng=rng)
    n = sc.n_months
    ramp = 1.0 + np.outer(np.linspace(0.0, 1.0, n), np.asarray(sc.drift, float))   # (n,3)
    a = np.asarray(sc.scales, float)
    rec = sim.records_levels + (ramp - 1.0) * a * sim.true_levels[:, None]   # drift on the truth part
    if sc.extra_cargo > 0:
        phi = spec.ar_phi
        e = rng.standard_normal(n + 100) * np.sqrt(1 - phi ** 2)
        ar = np.empty(n + 100); ar[0] = e[0]
        for t in range(1, n + 100):
            ar[t] = phi * ar[t - 1] + e[t]
        rec[:, 2] += ar[100:] * sc.extra_cargo * sim.true_levels.std(ddof=1)
    for i, k in enumerate(sc.shift):                       # wrong timing: record arrives k months late
        if k:
            rec[:, i] = np.concatenate([np.full(k, np.nan), rec[:-k, i]]) if k > 0 else \
                        np.concatenate([rec[-k:, i], np.full(-k, np.nan)])
    for r in sc.merge_janfeb:
        i = REC[r]
        for y in range(0, n - 1, 12):                      # month index 0 is January
            rec[y + 1, i] = rec[y, i] + rec[y + 1, i]
            rec[y, i] = np.nan
    for r, m, kind in sc.spikes:
        i = REC[r] if isinstance(r, str) else r
        rec[m, i] = 0.0 if kind == "zero" else rec[m, i] * 10.0
    for r, m in sc.gap_months:
        rec[m, REC[r] if isinstance(r, str) else r] = np.nan
    for m in sc.drop_months:
        rec[m, :] = np.nan
    return yoy_changes(rec), sim.planted, sim.true_yoy


def run_scenario(sc: Scenario, n_runs: int, seed: int, n_boot: int = 500,
                 block_len: int = 6, level: float = 0.90) -> dict:
    """Many fake datasets for one scenario; everything determined by `seed`."""
    rng = np.random.default_rng(seed)
    R = n_runs
    out = {"scenario": sc, "planted": np.asarray(sc.noise_levels, float), "seed": seed,
           "n_runs": R, "n_boot": n_boot, "block_len": block_len, "level": level,
           "rel": np.full((R, 3), np.nan), "noise_var": np.full((R, 3), np.nan),
           "lo": np.full((R, 3), np.nan), "hi": np.full((R, 3), np.nan),
           "flagged": np.zeros(R, bool), "reasons": [], "n_rows": np.zeros(R, int),
           "failed": np.zeros(R, bool)}
    for r in range(R):
        yoy, _, _ = simulate_scenario(sc, rng)
        keep = np.isfinite(yoy).all(axis=1)
        out["n_rows"][r] = int(keep.sum())
        x = yoy[keep]
        try:
            res = triple_collocation(x, NAMES)
        except ValueError:                      # fewer than 8 usable rows
            out["failed"][r] = True
            out["reasons"].append(("too_few_rows",))
            continue
        if x.shape[0] < block_len:
            out["failed"][r] = True
            out["reasons"].append(("too_few_rows",))
            continue
        lo, hi, _, _ = bootstrap_ranges(x, n_boot, block_len, rng, level)
        out["noise_var"][r], out["rel"][r] = res.noise_var, res.rel_noise
        out["lo"][r], out["hi"][r] = lo, hi
        out["flagged"][r] = not res.ok
        out["reasons"].append(res.reasons)
    return out


# ---------- the questions the project actually asks ----------
# Pre-specified in RECOVERY_BAR_scenarios.md.
TOL = {
    "coverage_min": 0.80,       # importer and ships 90% ranges contain the planted value
    "bias_max": 0.25,           # |median estimate / planted - 1| for importer and ships
    "wrong_call_max": 0.05,     # share of runs that confidently call the WRONG record noisier
}


def summarise(res: dict, tol: dict = TOL) -> dict:
    """Numbers reported for one scenario, plus the 'holds' verdict against TOL."""
    p = res["planted"]
    ok = ~res["failed"]
    lo, hi, rel, nv = res["lo"][ok], res["hi"][ok], res["rel"][ok], res["noise_var"][ok]
    n = max(ok.sum(), 1)
    has = np.isfinite(lo) & np.isfinite(hi)
    cover = np.array([np.mean((lo[has[:, i], i] <= p[i]) & (p[i] <= hi[has[:, i], i]))
                      if has[:, i].any() else np.nan for i in range(3)])
    med = np.array([np.nanmedian(rel[:, i]) if np.isfinite(rel[:, i]).any() else np.nan
                    for i in range(3)])
    undefined = 1 - np.isfinite(rel).mean(axis=0)
    bias = med / p - 1
    # failure-sentence call: ships' range sits entirely above BOTH customs ranges
    with np.errstate(invalid="ignore"):
        ships_worse = (lo[:, 2] > hi[:, 0]) & (lo[:, 2] > hi[:, 1])
        ships_better = (hi[:, 2] < lo[:, 0]) & (hi[:, 2] < lo[:, 1])
    truly_worse = bool(p[2] > max(p[0], p[1]) * 1.0)
    truly_better = bool(p[2] < min(p[0], p[1]) * 1.0)
    wrong = ships_better if truly_worse else (ships_worse if truly_better else (ships_worse | ships_better))
    right = ships_worse if truly_worse else (ships_better if truly_better else np.zeros(len(wrong), bool))
    ships_top = float(np.mean(np.argmax(np.nan_to_num(nv, nan=-1e9), axis=1) == 2))
    s = {
        "name": res["scenario"].name, "runs": int(res["n_runs"]), "failed_share": float(res["failed"].mean()),
        "median_rows": int(np.median(res["n_rows"])),
        "planted": p, "median_est": med, "bias": bias, "undefined": undefined,
        "coverage": cover, "flag_share": float(res["flagged"].mean()),
        "ships_worse_call": float(np.mean(ships_worse)), "ships_better_call": float(np.mean(ships_better)),
        "ships_truth": "worse" if truly_worse else ("better" if truly_better else "mixed"),
        "right_call": float(np.mean(right)), "wrong_call": float(np.mean(wrong)),
        "ships_top_rank": ships_top,
        "median_range": [(float(np.nanmedian(lo[:, i])), float(np.nanmedian(hi[:, i]))) for i in range(3)],
    }
    checks = {
        "coverage_importer": bool(cover[1] >= tol["coverage_min"]),
        "coverage_ships": bool(cover[2] >= tol["coverage_min"]),
        "bias_importer": bool(abs(bias[1]) <= tol["bias_max"]),
        "bias_ships": bool(abs(bias[2]) <= tol["bias_max"]),
        "wrong_call": bool(s["wrong_call"] <= tol["wrong_call_max"]),
    }
    s["checks"] = checks
    s["holds"] = all(checks.values())
    return s


# ---------- the scenario list (fixed before any scenario run; see the bar file) ----------
def build_scenarios() -> list[Scenario]:
    base = (0.50, 0.80, 1.60)       # "ships clearly worst" at a realistic scale
    drift9 = -0.09                  # ZAF/BRA-sized drift of the ship share
    drift24 = -0.24                 # LBR/PER-sized drift
    L = []
    # A: realistic noise levels, independent errors, steady share, 72 months
    L.append(Scenario("A1_zaf_like", "noise like ZAF: ships about as noisy as importer", (0.75, 1.55, 1.60)))
    L.append(Scenario("A2_bra_like", "noise like BRA: ships clearly the cleanest", (1.25, 1.25, 0.20)))
    L.append(Scenario("A3_all_equal", "all three equally and highly noisy", (1.00, 1.00, 1.00)))
    L.append(Scenario("A4_ships_worst", "ships clearly noisiest at a realistic scale", base))
    # B: short window (Liberia: 26 months -> 14 yoy rows)
    L.append(Scenario("B1_short_window", "26 months (Liberia-like), ships worst", base, n_months=26))
    # C: drifting share
    L.append(Scenario("C1_ships_drift_9", "ship share drifts 9% over the window", base, drift=(0, 0, drift9)))
    L.append(Scenario("C2_ships_drift_24", "ship share drifts 24% over the window", base, drift=(0, 0, drift24)))
    # D: shared mistakes between exporter and importer customs
    L.append(Scenario("D1_shared_025", "exporter and importer share 25% of their mistakes", base, error_corr=0.25))
    L.append(Scenario("D2_shared_050", "exporter and importer share 50% of their mistakes", base, error_corr=0.50))
    # E: wild months
    L.append(Scenario("E1_false_zero", "one importer month falsely zero, NOT flagged (Peru 2020)",
                      base, spikes=(("importer", 41, "zero"),)))
    L.append(Scenario("E2_unit_slip", "one exporter month x10, NOT flagged",
                      base, spikes=(("exporter", 41, "x10"),)))
    L.append(Scenario("E3_zero_flagged", "same false zero but caught and dropped in all records",
                      base, spikes=(("importer", 41, "zero"),), drop_months=(41,)))
    # F: everything at once, as the real table is
    L.append(Scenario("F1_combined", "ships drift 9% + 25% shared mistakes + one dropped month + 2 importer gaps",
                      base, drift=(0, 0, drift9), error_corr=0.25, drop_months=(41,),
                      gap_months=(("importer", 10), ("importer", 55))))
    # G: wrong timing (importer lag set too long; EU 3 months vs a ~3-week voyage)
    L.append(Scenario("G1_timing_1", "importer recorded 1 month late", base, shift=(0, 1, 0)))
    L.append(Scenario("G2_timing_3", "importer recorded 3 months late", base, shift=(0, 3, 0)))
    # H: China's January and February merged in the importer record
    L.append(Scenario("H1_janfeb_merge", "importer: Jan blank, Feb = Jan+Feb, every year",
                      base, merge_janfeb=("importer",)))
    # I: non-ore cargo in the ship tonnes (mixed ports)
    L.append(Scenario("I1_nonore_cargo", "ships carry a persistent non-ore component, half the ore's own variability",
                      base, extra_cargo=0.5))
    # J: steady shares far from 100% (importer sees 146% of exporter, ships 77%)
    L.append(Scenario("J1_steady_shares", "steady shares 100% / 146% / 77% (PER importer, ZAF ships)",
                      base, scales=(1.0, 1.46, 0.77)))
    return L
