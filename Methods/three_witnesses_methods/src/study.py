"""Stage 4A simulation study: does triple collocation recover planted noise?

For each of N fake datasets (see fake_data.py) this
  1. runs triple collocation (point estimates, status, reasons),
  2. builds 90% ranges with a moving-block bootstrap (keeps time order),
  3. records whether the planted value is inside each range,
and then scores everything against the bar written in RECOVERY_BAR.md.

Planted value = the noise/signal level on the year-over-year scale (the quantity
`rel_noise` estimates).  Ranking uses the raw estimated noise variance, so a
negative (impossible) value counts as the lowest, cleanest, and is not discarded.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

from fake_data import SimSpec, simulate, NAMES
from triple_collocation import triple_collocation


# ---------- block bootstrap ----------
def block_bootstrap_indices(n: int, block_len: int, n_boot: int, rng) -> np.ndarray:
    """Moving-block bootstrap row indices, shape (n_boot, n).

    Each resample glues together random runs of `block_len` consecutive rows, so
    neighbouring months stay neighbours.
    """
    if not 1 <= block_len <= n:
        raise ValueError("block_len must be between 1 and n")
    n_blocks = math.ceil(n / block_len)
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_len)).reshape(n_boot, -1)
    return idx[:, :n]


def cov_batch(Xb: np.ndarray) -> np.ndarray:
    """Covariance matrices for a batch of datasets: (B, n, 3) -> (B, 3, 3)."""
    n = Xb.shape[1]
    Xc = Xb - Xb.mean(axis=1, keepdims=True)
    return np.einsum("bni,bnj->bij", Xc, Xc) / (n - 1)


def tc_batch(C: np.ndarray):
    """Triple collocation on a batch of covariance matrices.

    Returns (noise_var, signal_var), each (B, 3).  Same formula as
    triple_collocation.triple_collocation_from_cov, vectorised.
    """
    B = C.shape[0]
    noise, signal = np.empty((B, 3)), np.empty((B, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(3):
            j, k = [m for m in range(3) if m != i]
            signal[:, i] = C[:, i, j] * C[:, i, k] / C[:, j, k]
            noise[:, i] = C[:, i, i] - signal[:, i]
    return noise, signal


def bootstrap_ranges(data: np.ndarray, n_boot: int, block_len: int, rng, level: float = 0.90):
    """Percentile range for each record's noise/signal.

    Replicates with non-positive shared signal are unusable and left out of the range
    (their share is returned).  Replicates with negative noise variance count as 0
    noise (the closest possible value); their share is returned too.
    Returns lo, hi, share_negative, share_unusable  (each length 3).
    """
    data = np.asarray(data, float)
    idx = block_bootstrap_indices(data.shape[0], block_len, n_boot, rng)
    noise, signal = tc_batch(cov_batch(data[idx]))
    usable = np.isfinite(signal) & (signal > 0) & np.isfinite(noise)
    rel = np.full(noise.shape, np.nan)
    rel[usable] = np.sqrt(np.clip(noise[usable], 0, None) / signal[usable])
    a = (1 - level) / 2 * 100
    lo, hi = np.full(3, np.nan), np.full(3, np.nan)
    for i in range(3):
        col = rel[:, i][np.isfinite(rel[:, i])]
        if col.size >= 20:
            lo[i], hi[i] = np.percentile(col, [a, 100 - a])
    return lo, hi, (usable & (noise < 0)).mean(axis=0), (~usable).mean(axis=0)


# ---------- the study ----------
def run_study(n_runs: int, seed: int, spec: SimSpec = SimSpec(), n_boot: int = 500,
              block_len: int = 6, level: float = 0.90) -> dict:
    """Run the whole study.  Everything is determined by `seed`."""
    rng = np.random.default_rng(seed)
    R = n_runs
    out = {
        "planted": np.asarray(spec.noise_levels, float),
        "noise_var": np.empty((R, 3)), "rel": np.empty((R, 3)),
        "lo": np.empty((R, 3)), "hi": np.empty((R, 3)),
        "boot_share_negative": np.empty((R, 3)),
        "flagged": np.zeros(R, bool), "reasons": [],
        "n_runs": R, "n_boot": n_boot, "block_len": block_len, "level": level,
        "seed": seed, "n_months": spec.n_months,
    }
    for r in range(R):
        sim = simulate(spec, rng=rng)
        res = triple_collocation(sim.records_yoy, NAMES)
        lo, hi, sneg, _ = bootstrap_ranges(sim.records_yoy, n_boot, block_len, rng, level)
        out["noise_var"][r], out["rel"][r] = res.noise_var, res.rel_noise
        out["lo"][r], out["hi"][r], out["boot_share_negative"][r] = lo, hi, sneg
        out["flagged"][r] = not res.ok
        out["reasons"].append(res.reasons)
    return out


def reason_category(reason: str) -> str:
    return reason.split(" ")[0]


def exporter_only(reasons) -> bool:
    return len(reasons) > 0 and all(r.startswith("negative_noise_variance:exporter") for r in reasons)


# ---------- scoring against the bar ----------
DEFAULT_BAR = {
    "ships_noisiest_min": 0.80,       # item 1
    "full_order_min": 0.60,           # item 2
    "coverage_window": (0.85, 0.95),  # item 3, importer and ships
    "coverage_exporter_min": 0.80,    # item 3, exporter
    "any_flag_max": 0.20,             # item 5
}


def evaluate_bar(res: dict, bar: dict = DEFAULT_BAR):
    """Score a study against the bar.  Returns (rows, details).

    rows: list of (item, description, observed, requirement, passed).
    """
    nv, planted = res["noise_var"], res["planted"]
    ships_noisiest = float(np.mean(np.argmax(nv, axis=1) == 2))
    full_order = float(np.mean((nv[:, 0] < nv[:, 1]) & (nv[:, 1] < nv[:, 2])))
    has_ci = np.isfinite(res["lo"]) & np.isfinite(res["hi"])
    cover = np.array([
        np.mean((res["lo"][has_ci[:, i], i] <= planted[i]) & (planted[i] <= res["hi"][has_ci[:, i], i]))
        if has_ci[:, i].any() else np.nan for i in range(3)])
    med_hi_exp = float(np.nanmedian(res["hi"][:, 0]))
    med_lo_ships = float(np.nanmedian(res["lo"][:, 2]))
    any_flag = float(res["flagged"].mean())
    lo_w, hi_w = bar["coverage_window"]

    rows = [
        (1, "ships identified as noisiest", ships_noisiest,
         f">= {bar['ships_noisiest_min']:.0%}", ships_noisiest >= bar["ships_noisiest_min"]),
        (2, "full order exporter < importer < ships", full_order,
         f">= {bar['full_order_min']:.0%}", full_order >= bar["full_order_min"]),
        ("3a", "coverage of 90% range: importer", cover[1],
         f"{lo_w:.0%}-{hi_w:.0%}", bool(lo_w <= cover[1] <= hi_w)),
        ("3b", "coverage of 90% range: ships", cover[2],
         f"{lo_w:.0%}-{hi_w:.0%}", bool(lo_w <= cover[2] <= hi_w)),
        ("3c", "coverage of 90% range: exporter", cover[0],
         f">= {bar['coverage_exporter_min']:.0%}", bool(cover[0] >= bar["coverage_exporter_min"])),
        (4, "median exporter range top is below median ships range bottom",
         f"{med_hi_exp:.3f} vs {med_lo_ships:.3f}", "exporter top < ships bottom",
         bool(med_hi_exp < med_lo_ships)),
        (5, "share of runs with any flag", any_flag,
         f"<= {bar['any_flag_max']:.0%}", any_flag <= bar["any_flag_max"]),
    ]
    details = {
        "coverage": cover, "ships_noisiest": ships_noisiest, "full_order": full_order,
        "reason_counts": Counter(reason_category(x) for rs in res["reasons"] for x in rs),
        "exporter_only_share": float(np.mean([exporter_only(rs) for rs in res["reasons"]])),
        "flag_share": any_flag,
    }
    return rows, details


def format_report(res: dict, bar: dict = DEFAULT_BAR, git_hash: str | None = None) -> str:
    rows, d = evaluate_bar(res, bar)
    L = []
    L.append("=" * 72)
    L.append("STAGE 4A SIMULATION STUDY: does triple collocation recover planted noise?")
    L.append("=" * 72)
    L.append(f"runs = {res['n_runs']}   months = {res['n_months']} "
             f"(-> {res['n_months'] - 12} year-over-year rows)   seed = {res['seed']}")
    L.append(f"bootstrap: {res['n_boot']} resamples, blocks of {res['block_len']} months, "
             f"{res['level']:.0%} ranges")
    L.append(f"code version (git): {git_hash or 'not a git repo'}")
    L.append(f"planted noise/signal: " + ", ".join(f"{n} {p:.0%}" for n, p in zip(NAMES, res["planted"])))
    L.append("")
    L.append("What the method estimated (point estimates, runs where defined):")
    L.append(f"  {'record':<10}{'planted':>8}{'median':>9}{'5th-95th pct':>16}{'undefined*':>12}"
             f"{'median 90% range':>22}")
    for i, nm in enumerate(NAMES):
        v = res["rel"][:, i]
        ok = np.isfinite(v)
        p5, p95 = (np.percentile(v[ok], [5, 95]) if ok.any() else (np.nan, np.nan))
        L.append(f"  {nm:<10}{res['planted'][i]:>8.2f}{(np.median(v[ok]) if ok.any() else np.nan):>9.3f}"
                 f"{p5:>9.3f} -{p95:>6.3f}{(1 - ok.mean()):>11.0%}"
                 f"{np.nanmedian(res['lo'][:, i]):>14.3f} -{np.nanmedian(res['hi'][:, i]):>6.3f}")
    L.append("  * 'undefined' = negative noise variance, reported not hidden.")
    L.append("")
    L.append("Scored against the bar (RECOVERY_BAR.md):")
    for item, desc, obs, req, ok in rows:
        o = f"{obs:.1%}" if isinstance(obs, float) else str(obs)
        L.append(f"  [{'PASS' if ok else 'FAIL'}] item {item}: {desc}: observed {o}, required {req}")
    L.append("")
    L.append(f"Overall: {'BAR MET' if all(r[4] for r in rows) else 'BAR NOT MET (see fallback in RECOVERY_BAR.md)'}")
    L.append("")
    L.append(f"Flagged runs (any cause): {d['flag_share']:.1%}; exporter-only "
             f"('too clean to measure'): {d['exporter_only_share']:.1%}")
    if d["reason_counts"]:
        L.append("  by cause: " + "; ".join(f"{k} x{v}" for k, v in d["reason_counts"].most_common()))
    return "\n".join(L)
