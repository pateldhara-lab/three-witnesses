"""Exploratory data analysis (EDA) of the side-by-side table.

Purpose: find out what kinds of errors the REAL data contain, and how big they
are, so that the fake data used to test triple collocation can imitate them.
Every function takes the validated table from table_io.load_table and returns a
pandas DataFrame, so each result can be read, saved or charted.

Names used below
    exporter  = exporter_t                      (witness 1)
    importer  = importer_aligned_total_t        (witness 2)
    ships     = ship_t                          (witness 3)
    window    = Jan 2019 - Dec 2024 (decision C2)

These are DESCRIPTIONS of the data, not results about which record is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = ("2019-01-01", "2024-12-01")
RECORDS = {"exporter": "exporter_t", "importer": "importer_aligned_total_t", "ships": "ship_t"}
SHARE_MIN = 0.60       # decision F2: a partial record must cover at least 60% ...
SHARE_MAX_RANGE = 0.07 # ... and its annual average may move at most ~7 points


# ---------- helpers ----------
def _calendar(window=WINDOW):
    return pd.date_range(window[0], window[1], freq="MS")


def _country(df, country, window=WINDOW):
    """One country on a full monthly calendar (missing months become NaN)."""
    d = df[df["country"] == country].set_index("month").reindex(_calendar(window))
    return d


def _bad_flag(f):
    toks = [t.strip() for t in str(f).replace(";", ",").split(",") if t.strip()]
    return any(t in ("missing", "known_source_error") for t in toks)


def _yoy(s: pd.Series, lag=12):
    return s - s.shift(lag)


def _excess_kurtosis(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 8 or x.std() == 0:
        return np.nan
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3.0)


def _autocorr1(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    if ok.sum() < 8:
        return np.nan
    a, b = x[1:][ok], x[:-1][ok]
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------- 1. coverage ----------
def coverage(df, window=WINDOW) -> pd.DataFrame:
    """Which months exist, per country and record, inside the window."""
    rows = []
    cal = _calendar(window)
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        all_rows = df[df["country"] == c]
        present = {k: d[col].notna() for k, col in RECORDS.items()}
        three = pd.concat(present, axis=1).all(axis=1)
        gaps = (~three).astype(int)
        longest = int((gaps * (gaps.groupby((gaps == 0).cumsum()).cumsum())).max())
        rows.append({
            "country": c,
            "table_first_month": all_rows["month"].min().date(),
            "table_last_month": all_rows["month"].max().date(),
            "window_months": len(cal),
            **{f"{k}_present": int(v.sum()) for k, v in present.items()},
            "all_three_present": int(three.sum()),
            "longest_gap_months": longest,
            "flagged_months": int(all_rows["flag"].map(_bad_flag).sum()),
        })
    return pd.DataFrame(rows)


# ---------- 2. sanity: zeros, wild months, possible unit slips ----------
def sanity(df, window=WINDOW, low=0.2, high=5.0) -> pd.DataFrame:
    """Months that look wrong on their own: zeros, negatives, or a value far
    from its neighbours (below `low` or above `high` times the centred 13-month
    median).  A ratio near 10 or 0.1 suggests a unit slip."""
    out = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        for k, col in RECORDS.items():
            s = d[col]
            med = s.rolling(13, center=True, min_periods=6).median()
            for m, v in s.items():
                if pd.isna(v):
                    continue
                kind = None
                if v < 0:
                    kind = "negative"
                elif v == 0:
                    kind = "zero"
                elif pd.notna(med.loc[m]) and med.loc[m] > 0:
                    r = v / med.loc[m]
                    if r < low:
                        kind = "far below neighbours"
                    elif r > high:
                        kind = "far above neighbours"
                if kind:
                    out.append({"country": c, "record": k, "month": m.date(), "value": v,
                                "neighbour_median": med.loc[m], "kind": kind})
    cols = ["country", "record", "month", "value", "neighbour_median", "kind"]
    return pd.DataFrame(out, columns=cols)


# ---------- 3. ratios: steady or drifting share? ----------
def ratios_by_year(df, window=WINDOW) -> pd.DataFrame:
    """Importer total and ship total as a share of the exporter's tonnes, per year."""
    rows = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        for k in ("importer", "ships"):
            both = d[[RECORDS[k], RECORDS["exporter"]]].dropna()
            both = both[both[RECORDS["exporter"]] > 0]
            if both.empty:
                continue
            yearly = both.groupby(both.index.year).sum()
            share = yearly[RECORDS[k]] / yearly[RECORDS["exporter"]]
            for y, v in share.items():
                rows.append({"country": c, "record": k, "year": int(y), "share": float(v),
                             "months": int((both.index.year == y).sum())})
    return pd.DataFrame(rows, columns=["country", "record", "year", "share", "months"])


def share_summary(df, window=WINDOW) -> pd.DataFrame:
    """Average share, how far the yearly share moves, and the F2 steady-share rule."""
    yr = ratios_by_year(df, window)
    rows = []
    for (c, k), g in yr.groupby(["country", "record"]):
        rng = g["share"].max() - g["share"].min()
        first, last = g.sort_values("year").iloc[0], g.sort_values("year").iloc[-1]
        rows.append({
            "country": c, "record": k, "years": len(g),
            "mean_share": g["share"].mean(),
            "min_year_share": g["share"].min(), "max_year_share": g["share"].max(),
            "range_points": rng * 100,
            "change_first_to_last_points": (last["share"] - first["share"]) * 100,
            "passes_F2_rule": bool(g["share"].min() >= SHARE_MIN and rng <= SHARE_MAX_RANGE),
        })
    return pd.DataFrame(rows, columns=["country", "record", "years", "mean_share", "min_year_share", "max_year_share", "range_points", "change_first_to_last_points", "passes_F2_rule"])


# ---------- 4. timing: which lag fits best? ----------
def lag_scan(df, window=WINDOW, max_lag=3) -> pd.DataFrame:
    """Correlation between the exporter's month t and a buyer's RAW figure at month
    t+lag, for lag 0..max_lag, on levels and on year-over-year changes.  The year-over-year
    column is the fairer one (seasons are shared by all records and inflate levels)."""
    rows = []
    cal = pd.date_range(window[0], pd.Timestamp(window[1]) + pd.DateOffset(months=max_lag), freq="MS")
    for c in sorted(df["country"].unique()):
        d = df[df["country"] == c].set_index("month").reindex(cal)
        exp = d["exporter_t"]
        for buyer, col in (("china", "china_raw_t"), ("eu", "eu_raw_t")):
            raw = d[col]
            if raw.notna().sum() < 12:
                continue
            for lag in range(max_lag + 1):
                x = exp.iloc[: len(cal) - max_lag]
                y = raw.shift(-lag).iloc[: len(cal) - max_lag]
                lv = pd.concat([x, y], axis=1).dropna()
                yo = pd.concat([_yoy(x), _yoy(y)], axis=1).dropna()
                rows.append({
                    "country": c, "buyer": buyer, "lag_months": lag,
                    "corr_levels": lv.corr().iloc[0, 1] if len(lv) >= 8 else np.nan,
                    "corr_yoy": yo.corr().iloc[0, 1] if len(yo) >= 8 else np.nan,
                    "n_levels": len(lv), "n_yoy": len(yo),
                })
    return pd.DataFrame(rows, columns=["country", "buyer", "lag_months", "corr_levels", "corr_yoy", "n_levels", "n_yoy"])


def best_lags(lagdf: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Best lag by year-over-year correlation, beside the lag Ben's table used."""
    rows = []
    for (c, b), g in lagdf.groupby(["country", "buyer"]):
        g = g.dropna(subset=["corr_yoy"])
        if g.empty:
            continue
        best = g.loc[g["corr_yoy"].idxmax()]
        used = df.loc[df["country"] == c, "lag_china" if b == "china" else "lag_eu"].dropna()
        rows.append({"country": c, "buyer": b, "best_lag_yoy": int(best["lag_months"]),
                     "best_corr_yoy": best["corr_yoy"],
                     "corr_yoy_spread_across_lags": g["corr_yoy"].max() - g["corr_yoy"].min(),
                     "lag_used_in_table": (int(used.mode().iloc[0]) if len(used) else np.nan)})
    return pd.DataFrame(rows)


# ---------- 5. seasons ----------
def seasonality(df, window=WINDOW) -> pd.DataFrame:
    """Share of each record's variation explained by calendar month, on levels and on
    year-over-year changes.  A large value on levels and a small one on changes is what
    taking year-over-year changes is meant to achieve."""
    rows = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        for k, col in RECORDS.items():
            s = d[col].dropna()
            rows.append({"country": c, "record": k,
                         "seasonal_share_levels": _month_r2(s),
                         "seasonal_share_yoy": _month_r2(_yoy(d[col]).dropna())})
    return pd.DataFrame(rows)


def _month_r2(s: pd.Series):
    if len(s) < 24:
        return np.nan
    tot = ((s - s.mean()) ** 2).sum()
    if tot == 0:
        return np.nan
    fit = s.groupby(s.index.month).transform("mean")
    return float(1 - ((s - fit) ** 2).sum() / tot)


# ---------- 6. how the errors behave ----------
def error_behaviour(df, window=WINDOW) -> pd.DataFrame:
    """Log of (record / exporter), month by month.  Real movement cancels out of a ratio,
    so what moves is mostly mistakes and slice differences.  Reported per country:
      sd            typical size of the month-to-month wobble (log points, about % / 100)
      lag1_autocorr  do wobbles persist from one month to the next? (0 = no, near 1 = yes)
      excess_kurtosis  heavy tails: 0 is a bell curve, large positive means occasional wild months
      share_extreme    share of months more than 3 robust standard deviations from typical
    plus corr_importer_vs_ships_ratio: both ratios share the exporter in the bottom, so a
    large positive value points to noise in the exporter record."""
    rows = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        d = d[~d["flag"].fillna("").map(_bad_flag)]
        exp = d["exporter_t"].where(d["exporter_t"] > 0)
        lr = {}
        for k in ("importer", "ships"):
            v = d[RECORDS[k]].where(d[RECORDS[k]] > 0)
            lr[k] = np.log(v / exp)
        for k, s in lr.items():
            x = (s - s.mean()).to_numpy()
            ok = np.isfinite(x)
            if ok.sum() < 12:
                continue
            mad = np.median(np.abs(x[ok] - np.median(x[ok]))) * 1.4826
            rows.append({
                "country": c, "record": k, "months_used": int(ok.sum()),
                "sd": float(np.nanstd(x)),
                "lag1_autocorr": _autocorr1(x),
                "excess_kurtosis": _excess_kurtosis(x),
                "share_extreme": float((np.abs(x[ok] - np.median(x[ok])) > 3 * mad).mean()) if mad > 0 else np.nan,
                "corr_importer_vs_ships_ratio": np.nan,
            })
        both = pd.concat([lr["importer"], lr["ships"]], axis=1).dropna()
        if len(both) >= 12:
            cc = float(both.corr().iloc[0, 1])
            for r in rows:
                if r["country"] == c:
                    r["corr_importer_vs_ships_ratio"] = cc
    return pd.DataFrame(rows, columns=["country", "record", "months_used", "sd", "lag1_autocorr", "excess_kurtosis", "share_extreme", "corr_importer_vs_ships_ratio"])


# ---------- 7. odd one out (preview of Stage 2) ----------
def odd_one_out(df, window=WINDOW, threshold=3.0) -> pd.DataFrame:
    """Month by month, which record's year-over-year change disagrees with the other two?
    Each change is standardised (robustly) within its country; a record is 'odd' when it sits
    more than `threshold` standard units from the average of the other two."""
    out = []
    for c in sorted(df["country"].unique()):
        full = _country(df, c, window)
        z = {}
        for k, col in RECORDS.items():
            ch = _yoy(full[col])
            if ch.notna().sum() < 8:
                z[k] = ch * np.nan
                continue
            mad = np.nanmedian(np.abs(ch - np.nanmedian(ch))) * 1.4826
            z[k] = (ch - np.nanmedian(ch)) / mad if mad and mad > 0 else ch * np.nan
        zz = pd.DataFrame(z)
        for k in RECORDS:
            others = [o for o in RECORDS if o != k]
            dev = zz[k] - zz[others].mean(axis=1)
            for m, v in dev.items():
                if np.isfinite(v) and abs(v) > threshold and zz.loc[m].notna().all():
                    out.append({"country": c, "month": m.date(), "odd_record": k, "deviation_units": float(v)})
    return pd.DataFrame(out, columns=["country", "month", "odd_record", "deviation_units"])


# ---------- error catalog: one row per country, sizes for the fake-data generator ----------
def catalog(df, window=WINDOW) -> pd.DataFrame:
    """The measured sizes the fake data should imitate, one row per country."""
    cov = coverage(df, window).set_index("country")
    sh = share_summary(df, window)
    eb = error_behaviour(df, window)
    san = sanity(df, window)
    odd = odd_one_out(df, window)
    rows = []
    for c in cov.index:
        row = {"country": c,
               "usable_months_all_three": int(cov.loc[c, "all_three_present"]),
               "longest_gap_months": int(cov.loc[c, "longest_gap_months"])}
        for k in ("importer", "ships"):
            g = sh[(sh.country == c) & (sh.record == k)]
            row[f"{k}_mean_share"] = float(g["mean_share"].iloc[0]) if len(g) else np.nan
            row[f"{k}_share_range_pts"] = float(g["range_points"].iloc[0]) if len(g) else np.nan
            row[f"{k}_share_drift_first_to_last_pts"] = float(g["change_first_to_last_points"].iloc[0]) if len(g) else np.nan
            row[f"{k}_passes_F2"] = bool(g["passes_F2_rule"].iloc[0]) if len(g) else np.nan
            e = eb[(eb.country == c) & (eb.record == k)]
            row[f"{k}_ratio_sd"] = float(e["sd"].iloc[0]) if len(e) else np.nan
            row[f"{k}_ratio_lag1_autocorr"] = float(e["lag1_autocorr"].iloc[0]) if len(e) else np.nan
            row[f"{k}_ratio_excess_kurtosis"] = float(e["excess_kurtosis"].iloc[0]) if len(e) else np.nan
        row["suspicious_months"] = int((san["country"] == c).sum())
        row["odd_one_out_months"] = int((odd["country"] == c).sum())
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- 9. how big is disagreement compared with real movement? ----------
def disagreement_vs_signal(df, window=WINDOW) -> pd.DataFrame:
    """On year-over-year LOG changes (about % change): the typical size of each record's
    movement, and how far apart each PAIR of records is, as a multiple of the exporter's
    movement (a stand-in for the real signal).

    If two records had independent noise of sizes a and b (as a fraction of the real movement),
    the pair would disagree by about sqrt(a^2 + b^2).  So these ratios say how noisy the real
    data are, WITHOUT splitting the noise between records (that is what triple collocation
    does later).  The exporter is used as the signal proxy and contains its own noise, so the
    true signal is a little smaller and these ratios are, if anything, too low."""
    rows = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        d = d[~d["flag"].fillna("").map(_bad_flag)]
        L = {k: np.log(d[col].where(d[col] > 0)) for k, col in RECORDS.items()}
        Y = pd.DataFrame({k: _yoy(v) for k, v in L.items()}).dropna()
        if len(Y) < 10:
            continue
        sd = Y.std(ddof=1)
        pair = lambda a, b: float((Y[a] - Y[b]).std(ddof=1))
        sig = float(sd["exporter"])
        rows.append({
            "country": c, "yoy_rows": len(Y), "signal_sd_exporter_yoy_log": sig,
            "importer_vs_exporter_ratio": pair("importer", "exporter") / sig,
            "ships_vs_exporter_ratio": pair("ships", "exporter") / sig,
            "importer_vs_ships_ratio": pair("importer", "ships") / sig,
        })
    return pd.DataFrame(rows, columns=["country", "yoy_rows", "signal_sd_exporter_yoy_log",
                                       "importer_vs_exporter_ratio", "ships_vs_exporter_ratio", "importer_vs_ships_ratio"])


# ---------- 10. does cargo shift between neighbouring months? ----------
def aggregation_effect(df, window=WINDOW) -> pd.DataFrame:
    """Size of the wobble in log(record / exporter) for single months versus 3-month blocks.
    If the mistakes were independent month to month, adding up three months would shrink the
    wobble to about 0.58 of its size (1 / sqrt 3).  If it shrinks MUCH more, cargo is being
    booked in a neighbouring month by one record and not the other (a timing shift), which
    cancels out when months are added up."""
    rows = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        d = d[~d["flag"].fillna("").map(_bad_flag)]
        for k in ("importer", "ships"):
            x = d[[RECORDS[k], "exporter_t"]].dropna()
            x = x[(x > 0).all(axis=1)]
            if len(x) < 18:
                continue
            monthly = np.log(x[RECORDS[k]] / x["exporter_t"])
            full = d[[RECORDS[k], "exporter_t"]].where(d[[RECORDS[k], "exporter_t"]] > 0)
            q = full.resample("QS").agg(lambda s: s.sum() if s.notna().all() and len(s) == 3 else np.nan)
            q = q.dropna()
            if len(q) < 6:
                continue
            quarterly = np.log(q[RECORDS[k]] / q["exporter_t"])
            sm, sq = float(monthly.std(ddof=1)), float(quarterly.std(ddof=1))
            rows.append({"country": c, "record": k, "sd_monthly": sm, "sd_quarterly": sq,
                         "quarterly_over_monthly": sq / sm, "expected_if_independent": 1 / np.sqrt(3)})
    return pd.DataFrame(rows, columns=["country", "record", "sd_monthly", "sd_quarterly",
                                       "quarterly_over_monthly", "expected_if_independent"])


def biggest_disagreements(df, window=WINDOW, n=4) -> pd.DataFrame:
    """The n months per country where one record's year-over-year change sits furthest from the
    other two (standardised).  Unlike odd_one_out there is no cut-off, so something always shows."""
    out = []
    for c in sorted(df["country"].unique()):
        full = _country(df, c, window)
        z = {}
        for k, col in RECORDS.items():
            ch = _yoy(full[col])
            if ch.notna().sum() < 8:
                z[k] = ch * np.nan
                continue
            mad = np.nanmedian(np.abs(ch - np.nanmedian(ch))) * 1.4826
            z[k] = (ch - np.nanmedian(ch)) / mad if mad and mad > 0 else ch * np.nan
        zz = pd.DataFrame(z)
        zz = zz[zz.notna().all(axis=1)]
        recs = []
        for k in RECORDS:
            others = [o for o in RECORDS if o != k]
            dev = zz[k] - zz[others].mean(axis=1)
            for m, v in dev.items():
                recs.append({"country": c, "month": m.date(), "odd_record": k, "deviation_units": float(v)})
        r = pd.DataFrame(recs)
        if len(r):
            out.append(r.reindex(r["deviation_units"].abs().sort_values(ascending=False).index).head(n))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["country", "month", "odd_record", "deviation_units"])

# ---------- 11. how alike are the three records on year-over-year changes? ----------
def yoy_correlations(df, window=WINDOW) -> pd.DataFrame:
    """Pairwise correlation of year-over-year tonne changes, per country.

    In the fake-data study the three records were almost perfectly alike (correlations about
    0.95).  Real records are not.  `equal_noise_equivalent` converts the average correlation
    into the noise/signal each record would need if all three were equally noisy and their
    mistakes unrelated: correlation = 1 / (1 + n^2), so n = sqrt(1/corr - 1).  It is a rough
    way to size the fake data, NOT an estimate of any one record's noise."""
    rows = []
    for c in sorted(df["country"].unique()):
        d = _country(df, c, window)
        d = d[~d["flag"].fillna("").map(_bad_flag)]
        Y = pd.DataFrame({k: _yoy(d[col]) for k, col in RECORDS.items()}).dropna()
        if len(Y) < 10:
            continue
        cc = Y.corr()
        pairs = [cc.loc["importer", "exporter"], cc.loc["ships", "exporter"], cc.loc["importer", "ships"]]
        mean = float(np.mean(pairs))
        rows.append({"country": c, "yoy_rows": len(Y),
                     "corr_importer_exporter": float(pairs[0]),
                     "corr_ships_exporter": float(pairs[1]),
                     "corr_importer_ships": float(pairs[2]),
                     "mean_pairwise_corr": mean,
                     "equal_noise_equivalent": float(np.sqrt(1 / mean - 1)) if mean > 0 else np.nan})
    return pd.DataFrame(rows, columns=["country", "yoy_rows", "corr_importer_exporter", "corr_ships_exporter",
                                       "corr_importer_ships", "mean_pairwise_corr", "equal_noise_equivalent"])
