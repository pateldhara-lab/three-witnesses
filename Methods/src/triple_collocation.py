"""Triple collocation (TC) for the Three Witnesses project.

THE IDEA IN ONE PARAGRAPH
Three records (exporter, importer, ships) each measure the same hidden "true"
movement, plus their own mistakes.  Model for each record i:

    record_i = a_i * truth + b_i + noise_i

where a_i is a steady scale (e.g. an importer that always sees 70% of the ore),
b_i a steady offset, and noise_i the record's own error.  If the three noise
terms are unrelated to each other, then the covariance between two records
contains ONLY the shared truth.  That gives, for record i with the other two
called j and k:

    shared (signal) variance of i  =  C_ij * C_ik / C_jk
    noise variance of i            =  C_ii - C_ij * C_ik / C_jk

(C is the covariance matrix of the three records.)  This is the formula quoted
in the Week 1 plan: "total spread minus (shared movement with the second times
shared movement with the third) divided by shared movement of second and third".

WHAT WE REPORT
Scale-free numbers, so a record that sees only 70% of the ore is not punished:
    rel_noise        = noise sd / signal sd   (0.25 means "noise is 25% the size
                       of the real movement"; this is what we plant in
                       simulations as 10% / 25% / 40%)
    corr_with_truth  = correlation between the record and the hidden truth

IMPOSSIBLE ANSWERS ("assumptions broken")
The method can return a negative noise variance, which is impossible.  We never
hide this.  The result carries status="assumptions_broken" and a list of
reasons.  NOTE: with only ~60 points this can also happen by sampling bad luck
even when the assumptions hold; the simulation measures how often.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

DEFAULT_NAMES = ("exporter", "importer", "ships")
# If two records correlate less than this, the formula divides by a near-zero
# number and the answers are not trustworthy.
MIN_PAIR_CORR = 0.10


@dataclass(frozen=True)
class TCResult:
    names: tuple
    n_used: int
    n_dropped: int
    noise_var: np.ndarray        # in each record's own units squared; can be negative
    signal_var: np.ndarray       # a_i^2 * var(truth); can be negative if broken
    rel_noise: np.ndarray        # sqrt(noise_var/signal_var); NaN if impossible
    corr_with_truth: np.ndarray  # sqrt(signal_var/total_var); NaN if impossible
    status: str                  # "ok" or "assumptions_broken"
    reasons: tuple

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> str:
        lines = [f"status: {self.status}  (n used = {self.n_used}, dropped = {self.n_dropped})"]
        for i, nm in enumerate(self.names):
            lines.append(
                f"  {nm:<9} noise/signal = {self.rel_noise[i]:.3f}   "
                f"corr with truth = {self.corr_with_truth[i]:.3f}"
            )
        if self.reasons:
            lines.append("  reasons: " + "; ".join(self.reasons))
        return "\n".join(lines)


def yoy_changes(levels, lag: int = 12) -> np.ndarray:
    """Year-over-year change: value minus the value `lag` rows earlier.

    Removes the shared seasonal pattern (which would make records look more
    alike than they are).  A 72-month series becomes 60 rows.  NaNs propagate.
    """
    a = np.asarray(levels, dtype=float)
    if a.shape[0] <= lag:
        raise ValueError(f"need more than {lag} rows, got {a.shape[0]}")
    return a[lag:] - a[:-lag]


def triple_collocation_from_cov(cov, names=DEFAULT_NAMES, n_used=0, n_dropped=0,
                                min_pair_corr: float = MIN_PAIR_CORR) -> TCResult:
    """Core calculation from a 3x3 covariance matrix (used by tests, bootstrap)."""
    C = np.asarray(cov, dtype=float)
    if C.shape != (3, 3):
        raise ValueError("cov must be 3x3")
    noise = np.empty(3)
    signal = np.empty(3)
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        with np.errstate(divide="ignore", invalid="ignore"):
            signal[i] = C[i, j] * C[i, k] / C[j, k]
        noise[i] = C[i, i] - signal[i]

    reasons = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = C[a, b] / np.sqrt(C[a, a] * C[b, b])
        if not np.isfinite(r) or r < min_pair_corr:
            reasons.append(f"weak_link:{names[a]}-{names[b]} (corr={r:.2f})")
    for i in range(3):
        if not np.isfinite(noise[i]) or noise[i] < 0:
            reasons.append(f"negative_noise_variance:{names[i]}")
        if not np.isfinite(signal[i]) or signal[i] <= 0:
            reasons.append(f"nonpositive_signal:{names[i]}")

    good = np.isfinite(noise) & np.isfinite(signal) & (noise >= 0) & (signal > 0)
    rel = np.full(3, np.nan)
    corr = np.full(3, np.nan)
    rel[good] = np.sqrt(noise[good] / signal[good])
    corr[good] = np.sqrt(signal[good] / np.diag(C)[good])
    return TCResult(tuple(names), int(n_used), int(n_dropped), noise, signal, rel, corr,
                    "assumptions_broken" if reasons else "ok", tuple(reasons))


def triple_collocation(data, names=DEFAULT_NAMES,
                       min_pair_corr: float = MIN_PAIR_CORR) -> TCResult:
    """Run TC on an (n, 3) array: columns = the three records, rows = time.

    Rows with any NaN are dropped (and counted in n_dropped).  Feed it
    year-over-year changes, not raw levels (see yoy_changes).
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"data must have shape (n, 3), got {x.shape}")
    keep = np.isfinite(x).all(axis=1)
    n_dropped = int((~keep).sum())
    x = x[keep]
    if x.shape[0] < 8:
        raise ValueError(f"need at least 8 complete rows, got {x.shape[0]}")
    C = np.cov(x, rowvar=False, ddof=1)
    return triple_collocation_from_cov(C, names, x.shape[0], n_dropped, min_pair_corr)
