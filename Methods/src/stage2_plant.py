"""Stage 2 (Flag) test harness: plant known errors in the table, count what a flag rule catches.

The slide's pass test: plant fake errors in real data, then count how many the flags catch
and how often they raise a false alarm.  Rules are fitted on 2019-2022 and tested on later years.

The error kinds come from the EDA error catalog and the scenario study, not from guesswork:
    x10         a month multiplied by 10            (unit slip; scenario E2)
    zero        a month set to 0                    (false zero; Peru importer June 2020; scenario E1)
    unit_swap   a month multiplied by 1000          (tonnes vs kilograms)
    drop        a month made blank, unflagged       (missing month)
    janfeb      January blank, February = Jan + Feb (China's merged months; scenario H1)

THIS FILE DOES NOT CONTAIN BEN'S RULES.  `reference_rule` is a deliberately simple stand-in so the
harness can be tested end to end.  Ben's rules plug in through the same interface:
    rule(table_df, fit_months) -> set of (country, month) flagged
Flags are per country-month (the slide: "flag that month as suspicious").

Pass numbers must be agreed and committed BEFORE the harness is run on the real table
(see STAGE2_BAR_template.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RECORD_COLS = {"exporter": "exporter_t", "importer": "importer_aligned_total_t", "ships": "ship_t"}
KINDS = ("x10", "zero", "unit_swap", "drop", "janfeb")
FIT_END = pd.Timestamp("2022-12-01")
TEST_START, TEST_END = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-01")


def plant_errors(df: pd.DataFrame, rng, n_per_kind: int = 3, kinds=KINDS, months=None):
    """Return (damaged copy, truth table).  `months` = (first, last) to plant inside (default: the test years).

    Rules: only months where all three records exist; at most one error per country-month and no
    two planted months of one country adjacent (so one error cannot hide in another's neighbours);
    janfeb is planted on a January/February pair; the error goes in one randomly chosen record.
    """
    lo, hi = months or (TEST_START, TEST_END)
    out = df.copy()
    truth, used = [], set()
    ok = out.dropna(subset=list(RECORD_COLS.values()))
    ok = ok[(ok.month >= lo) & (ok.month <= hi)]
    for kind in kinds:
        planted = 0
        pool = ok[ok.month.dt.month == 1] if kind == "janfeb" else ok
        for _ in range(500):
            if planted == n_per_kind:
                break
            row = pool.iloc[int(rng.integers(len(pool)))]
            c, m = row["country"], row["month"]
            near = {(c, m + pd.DateOffset(months=k)) for k in (-1, 0, 1)}
            if kind == "janfeb":
                near |= {(c, m + pd.DateOffset(months=2))}
            if near & used:
                continue
            rec = list(RECORD_COLS)[int(rng.integers(3))]
            col = RECORD_COLS[rec]
            i = out.index[(out.country == c) & (out.month == m)][0]
            if kind == "x10":
                out.loc[i, col] *= 10
            elif kind == "zero":
                out.loc[i, col] = 0.0
            elif kind == "unit_swap":
                out.loc[i, col] *= 1000
            elif kind == "drop":
                out.loc[i, col] = np.nan
            elif kind == "janfeb":
                j = out.index[(out.country == c) & (out.month == m + pd.DateOffset(months=1))]
                if len(j) == 0 or not np.isfinite(out.loc[j[0], col]):
                    continue
                out.loc[j[0], col] = out.loc[i, col] + out.loc[j[0], col]
                out.loc[i, col] = np.nan
                truth.append((c, m + pd.DateOffset(months=1), rec, kind))   # both months are damaged
            used |= near
            truth.append((c, m, rec, kind))
            planted += 1
    return out, pd.DataFrame(truth, columns=["country", "month", "record", "kind"])


def score_flags(flagged: set, truth: pd.DataFrame, df: pd.DataFrame, months=None) -> dict:
    """Catch rate (per kind and overall) and false-alarm rate over a test window.

    catch rate        = planted country-months that were flagged / planted country-months
    false-alarm rate  = flagged country-months with NO planted error / clean country-months
    A country-month counts once however many records are damaged in it.
    """
    lo, hi = months or (TEST_START, TEST_END)
    cm = df[(df.month >= lo) & (df.month <= hi)][["country", "month"]].drop_duplicates()
    allm = set(map(tuple, cm.to_numpy()))
    bad = set(map(tuple, truth[["country", "month"]].drop_duplicates().to_numpy())) & allm
    clean = allm - bad
    flagged = set(flagged) & allm
    by_kind = {}
    for k, g in truth.groupby("kind"):
        s = set(map(tuple, g[["country", "month"]].to_numpy())) & allm
        by_kind[k] = (len(s & flagged) / len(s)) if s else np.nan
    return {"n_planted": len(bad), "n_clean": len(clean),
            "catch_rate": len(bad & flagged) / len(bad) if bad else np.nan,
            "false_alarm_rate": len(clean & flagged) / len(clean) if clean else np.nan,
            "catch_by_kind": by_kind}


def reference_rule(df: pd.DataFrame, fit_months, k: float = None, target_false_alarm: float = 0.05):
    """A SIMPLE stand-in rule (not Ben's): flag a month if any record's log-ratio to the other records,
    or any record being zero/blank, is far from that country's typical value.

    Fitted on the fit months only: the typical value is the fit-period median, the spread is the
    fit-period MAD, and the cut-off is the smallest k that keeps the fit-period false-alarm rate at or
    below `target_false_alarm`.  Returns the set of flagged (country, month) over ALL months.
    """
    d = df.copy()
    cols = list(RECORD_COLS.values())
    zs = pd.DataFrame(index=d.index)
    for c in d.country.unique():
        m = d.country == c
        fit = m & d.month.isin(fit_months)
        feats = {}
        for a, b in (("exporter", "importer"), ("exporter", "ships"), ("importer", "ships")):
            x = np.log(d.loc[m, RECORD_COLS[a]].where(d.loc[m, RECORD_COLS[a]] > 0)
                       / d.loc[m, RECORD_COLS[b]].where(d.loc[m, RECORD_COLS[b]] > 0))
            f = x[fit[m]]
            med, mad = f.median(), 1.4826 * (f - f.median()).abs().median()
            feats[f"{a}/{b}"] = ((x - med) / mad) if mad and np.isfinite(mad) else x * np.nan
        z = pd.DataFrame(feats)
        zs.loc[m, "z"] = z.abs().max(axis=1).fillna(np.inf)     # a zero/blank record makes a ratio NaN -> flagged
    has_all = d[cols].notna().all(axis=1)
    fit_rows = d.month.isin(fit_months) & has_all
    if k is None:
        cand = zs.loc[fit_rows, "z"].replace(np.inf, np.nan).dropna()
        k = float(np.quantile(cand, 1 - target_false_alarm)) if len(cand) else 3.0
    flagged = d[(zs["z"] > k)]
    # a month that is blank in some record is only flagged if the other records exist (an unflagged gap)
    return set(map(tuple, flagged[["country", "month"]].to_numpy()))


def run_once(df, rule, rng, n_per_kind=3):
    """Fit on 2019-2022, plant in 2023-2024, flag, score.  `rule(df, fit_months)` returns a set."""
    fit_months = set(df.loc[df.month <= FIT_END, "month"])
    damaged, truth = plant_errors(df, rng, n_per_kind)
    return score_flags(rule(damaged, fit_months), truth, damaged), truth
