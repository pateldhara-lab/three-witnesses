"""Run triple collocation on the REAL side-by-side table (the run behind Report 2, Section 3).

    python scripts/run_real_data.py                      # uses data/side_by_side.csv
    python scripts/run_real_data.py path/to/table.csv    # or any other table
    python scripts/run_real_data.py --min-rows 14        # also run countries with 14-23 rows

For each country: year-over-year changes of exporter, importers, ships; triple collocation;
90% ranges from a moving-block bootstrap (6-month blocks, 500 resamples, fixed seed);
steadiness labels (60% floor, 7-point range).  Writes to results/real_data/:
    real_tc_results.csv   real_tc_report.txt   real_tc_figure.png
Exploratory: read the ranges, not just the numbers.
"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from table_io import load_table, country_yoy, TC_NAMES
from triple_collocation import triple_collocation
from study import bootstrap_ranges

ap = argparse.ArgumentParser()
ap.add_argument("table", nargs="?", default=os.path.join(HERE, "..", "data", "side_by_side.csv"))
ap.add_argument("--min-rows", type=int, default=24)
ap.add_argument("--seed", type=int, default=20261124)
a = ap.parse_args()
if not os.path.exists(a.table):
    sys.exit(f"Table not found: {a.table}\nPut Ben's final side_by_side.csv in the data/ folder (see README).")

df = load_table(a.table)
out_dir = os.path.join(HERE, "..", "results", "real_data"); os.makedirs(out_dir, exist_ok=True)
NAMES = {"AUS": "Australia", "BRA": "Brazil", "PER": "Peru", "ZAF": "South Africa", "LBR": "Liberia"}

def steadiness(country):
    d = df[(df.country == country) & (df.month >= "2019-01-01") & (df.month <= "2024-12-01")].copy()
    d["y"] = d.month.dt.year; res = {}
    for col, k in (("importer_aligned_total_t", "importer"), ("ship_t", "ships")):
        m = d.dropna(subset=["exporter_t", col]); g = m.groupby("y")[["exporter_t", col]].sum()
        p = g[col] / g["exporter_t"] * 100
        res[k] = (float(p.min()), float(p.max() - p.min()))
    ok = all(v[0] >= 60 and v[1] <= 7 for v in res.values())
    return res, ok

rows, lines = [], []
for w in df.attrs["warnings"]: lines.append("WARNING: " + w)
lines += ["REAL-DATA TRIPLE COLLOCATION  (table: %s)" % os.path.basename(a.table), "noise / signal, lower = cleaner; 90% range in brackets", ""]
for c in sorted(df.country.unique()):
    ci = country_yoy(df, c); st, steady = steadiness(c)
    row = {"country": c, "rows": ci.n_yoy_rows, "steady_60_and_7pt": steady,
           "importer_share_min": st["importer"][0], "importer_share_range": st["importer"][1],
           "ship_share_min": st["ships"][0], "ship_share_range": st["ships"][1]}
    if ci.n_yoy_rows < a.min_rows:
        row["status"] = f"not run (< {a.min_rows} rows)"; lines.append(f"{c}: {ci.n_yoy_rows} rows - {row['status']}\n")
    else:
        res = triple_collocation(ci.yoy, TC_NAMES)
        lo, hi, sneg, _ = bootstrap_ranges(ci.yoy, 500, 6, np.random.default_rng(a.seed))
        corr = np.corrcoef(ci.yoy.T)
        row.update(status=("ok" if res.ok else "assumptions_broken"), reasons=";".join(res.reasons),
                   corr_exp_imp=corr[0, 1], corr_exp_ship=corr[0, 2], corr_imp_ship=corr[1, 2])
        lines.append(f"{c} ({NAMES.get(c, c)}): {ci.n_yoy_rows} rows, status {row['status']}")
        for i, n in enumerate(TC_NAMES):
            v = res.rel_noise[i]; row[n] = float(v) if np.isfinite(v) else np.nan
            row[n + "_lo"], row[n + "_hi"], row[n + "_undefined_share"] = float(lo[i]), float(hi[i]), float(sneg[i])
            lines.append(f"   {n:9} {'undefined' if not np.isfinite(v) else f'{v:.2f}':>9}   ({lo[i]:.2f} - {hi[i]:.2f})   undefined in {sneg[i]*100:.0f}% of resamples")
        if res.reasons: lines.append("   reasons: " + "; ".join(res.reasons))
        lines.append(f"   steadiness: importer share min {st['importer'][0]:.0f}% range {st['importer'][1]:.1f} pts; "
                     f"ships min {st['ships'][0]:.0f}% range {st['ships'][1]:.1f} pts -> {'steady' if steady else 'NOT steady'}\n")
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "real_tc_results.csv"), index=False)
open(os.path.join(out_dir, "real_tc_report.txt"), "w").write("\n".join(lines))
print("\n".join(lines))

# figure: one panel per country, dot = estimate, line = 90% range, dashed = noise equals signal
run = [r for r in rows if "exporter" in r]
fig, axs = plt.subplots(1, len(run), figsize=(2.2 * len(run), 3), sharey=True)
axs = np.atleast_1d(axs); cols = ["#2B6CB0", "#2F855A", "#DD6B20"]
for ax, r in zip(axs, run):
    ax.axhline(1, color="k", lw=0.6, ls="--")
    for i, n in enumerate(TC_NAMES):
        ax.vlines(i, r[n + "_lo"], r[n + "_hi"], color=cols[i], lw=2)
        if np.isfinite(r[n]): ax.plot(i, r[n], "o", color=cols[i])
        else: ax.plot(i, 0, "o", mfc="white", mec=cols[i])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Exp.", "Imp.", "Ships"], fontsize=8)
    ax.set_title(f"{NAMES.get(r['country'], r['country'])}\n({r['rows']} rows)", fontsize=9)
axs[0].set_ylabel("noise / signal (90% range)"); axs[0].set_ylim(-0.1, 7)
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "real_tc_figure.png"), dpi=200)
print("Saved to results/real_data/")
