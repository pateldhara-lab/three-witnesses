"""Exploratory data analysis of Ben's side-by-side table.

    python scripts/run_eda.py path/to/side_by_side.csv
    python scripts/run_eda.py --fake            (practice run on made-up data)

Writes results/eda/: one CSV per analysis, eda_report.txt, and charts (PNG) if
matplotlib is installed.  Everything printed here DESCRIBES the data; none of it is
a result about which record is right.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from table_io import load_table
import eda

if len(sys.argv) != 2:
    sys.exit(__doc__)
if sys.argv[1] == "--fake":
    from fake_data import make_fake_table
    df = load_table(make_fake_table())
    label = "FAKE DATA (practice run)"
else:
    df = load_table(sys.argv[1])
    label = sys.argv[1]

root = os.path.join(os.path.dirname(__file__), "..")
out = os.path.join(root, "results", "eda" + ("_fake" if sys.argv[1] == "--fake" else ""))
os.makedirs(out, exist_ok=True)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
fmt = lambda d: d.round(3).to_string(index=False) if len(d) else "(none)"

lagdf = eda.lag_scan(df)
results = {
    "1_coverage": eda.coverage(df),
    "2_suspicious_months": eda.sanity(df),
    "3_share_by_year": eda.ratios_by_year(df),
    "3_share_summary": eda.share_summary(df),
    "4_lag_scan": lagdf,
    "4_best_lags": eda.best_lags(lagdf, df),
    "5_seasonality": eda.seasonality(df),
    "6_error_behaviour": eda.error_behaviour(df),
    "7_odd_one_out": eda.odd_one_out(df),
    "7b_biggest_disagreements": eda.biggest_disagreements(df),
    "8_error_catalog": eda.catalog(df),
    "9_disagreement_vs_signal": eda.disagreement_vs_signal(df),
    "10_timing_aggregation": eda.aggregation_effect(df),
    "11_yoy_correlations": eda.yoy_correlations(df),
}
notes = {
    "1_coverage": "Months present per record inside Jan 2019 - Dec 2024, and the longest run of months where any record is missing.",
    "2_suspicious_months": "Zeros, negatives, or values far from their neighbours (ratio near 10 or 0.1 hints at a unit slip).",
    "3_share_summary": "Importer and ship totals as a share of the exporter's tonnes. passes_F2_rule = at least 60% every year AND yearly share moves at most 7 points.",
    "4_best_lags": "Best lag by year-over-year correlation beside the lag used in the table. A small spread means the lag is weakly identified.",
    "5_seasonality": "Share of variation explained by calendar month. Should be large on levels and small on year-over-year changes.",
    "6_error_behaviour": "Wobble of log(record/exporter): size (sd), persistence (lag1_autocorr), wild months (excess_kurtosis, share_extreme).",
    "7_odd_one_out": "PREVIEW for Stage 2: months where one record's year-over-year change disagrees with the other two.",
    "7b_biggest_disagreements": "The 4 months per country where one record sits furthest from the other two (no cut-off).",
    "8_error_catalog": "One row per country: the measured sizes the fake data should imitate.",
    "9_disagreement_vs_signal": "How far apart each PAIR of records is, as a multiple of the exporter's year-over-year movement. Independent noise a and b gives about sqrt(a^2+b^2). The fake-data study had 0.27 to 0.47.",
    "10_timing_aggregation": "Wobble of log(record/exporter) in single months vs 3-month blocks. Near 0.58 = independent; well below = cargo shifting between neighbouring months.",
    "11_yoy_correlations": "How alike the records are on year-over-year changes. The fake-data study had about 0.95. equal_noise_equivalent = noise/signal each record would need if all three were equally noisy.",
}

lines = [f"EDA of {label}", "=" * 72, ""]
for name, d in results.items():
    d.to_csv(os.path.join(out, name + ".csv"), index=False)
    if name in ("3_share_by_year", "4_lag_scan"):
        continue
    lines += [f"--- {name} ---", notes.get(name, ""), fmt(d), ""]
report = "\n".join(lines)
with open(os.path.join(out, "eda_report.txt"), "w") as f:
    f.write(report)
print(report)

# ---------- charts (optional) ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib not installed: skipping charts (pip install matplotlib)")
    sys.exit(0)

NAVY, TEAL, YEL, RED = "#10243E", "#1F8A8C", "#FFD35C", "#B8432F"
countries = sorted(df["country"].unique())

# A. coverage timeline
cal = eda._calendar()
fig, ax = plt.subplots(figsize=(9, 0.5 * len(countries) * 3 + 1.2))
yl, labels = 0, []
for c in countries:
    d = eda._country(df, c)
    for k, col in eda.RECORDS.items():
        ok = d[col].notna().to_numpy()
        ax.scatter(np.arange(len(cal))[ok], np.full(ok.sum(), yl), s=9, color=NAVY, marker="s")
        ax.scatter(np.arange(len(cal))[~ok], np.full((~ok).sum(), yl), s=9, color=RED, marker="s")
        labels.append(f"{c} {k}"); yl += 1
    yl += 0.6
ax.set_yticks([i for i, l in enumerate(labels)]); ax.set_yticklabels(labels, fontsize=7)
ax.set_xticks(np.arange(0, len(cal), 12)); ax.set_xticklabels([str(cal[i].year) for i in range(0, len(cal), 12)])
ax.invert_yaxis(); ax.set_title("Which months exist (navy) and which are missing (red)", loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(out, "chart_coverage.png"), dpi=160); plt.close()

# B. share over time
yr = results["3_share_by_year"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, k in zip(axs, ("importer", "ships")):
    for c in countries:
        g = yr[(yr.country == c) & (yr.record == k)].sort_values("year")
        if len(g):
            ax.plot(g["year"], g["share"] * 100, marker="o", label=c)
    ax.axhline(60, color=RED, ls="--", lw=1); ax.set_title(f"{k} total as % of exporter's tonnes", fontsize=9, loc="left")
    ax.set_xlabel("year")
axs[0].set_ylabel("% of exporter"); axs[1].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(out, "chart_shares.png"), dpi=160); plt.close()

# C. lag scan
ld = results["4_lag_scan"]
if len(ld):
    pairs = ld[["country", "buyer"]].drop_duplicates().to_numpy().tolist()
    fig, axs = plt.subplots(1, len(pairs), figsize=(2.3 * len(pairs), 2.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (c, b) in zip(axs, pairs):
        g = ld[(ld.country == c) & (ld.buyer == b)]
        ax.bar(g["lag_months"], g["corr_yoy"], color=TEAL)
        ax.set_title(f"{c} {b}", fontsize=8); ax.set_xticks(range(4)); ax.set_xlabel("lag (months)", fontsize=7)
    axs[0].set_ylabel("year-over-year corr", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "chart_lags.png"), dpi=160); plt.close()
# D. how alike are the records? real data vs the fake-data study
yc = results["11_yoy_correlations"]
dv = results["9_disagreement_vs_signal"]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
x = np.arange(len(yc)); w = 0.26
for i, (col, lab, colr) in enumerate((("corr_importer_exporter", "importer-exporter", NAVY),
                                       ("corr_ships_exporter", "ships-exporter", TEAL),
                                       ("corr_importer_ships", "importer-ships", YEL))):
    axs[0].bar(x + (i - 1) * w, yc[col], w, label=lab, color=colr)
axs[0].axhline(0.95, color=RED, ls="--", lw=1); axs[0].text(len(yc) - 0.5, 0.96, "fake-data study", color=RED, fontsize=7, ha="right", va="bottom")
axs[0].set_xticks(x); axs[0].set_xticklabels([f"{c}\n({n} rows)" for c, n in zip(yc["country"], yc["yoy_rows"])], fontsize=7)
axs[0].set_ylim(0, 1.1); axs[0].set_title("How alike the records are (year-over-year)", fontsize=9, loc="left"); axs[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
x = np.arange(len(dv))
for i, (col, lab, colr) in enumerate((("importer_vs_exporter_ratio", "importer-exporter", NAVY),
                                       ("ships_vs_exporter_ratio", "ships-exporter", TEAL),
                                       ("importer_vs_ships_ratio", "importer-ships", YEL))):
    axs[1].bar(x + (i - 1) * w, dv[col], w, label=lab, color=colr)
axs[1].axhspan(0.27, 0.47, color=RED, alpha=0.15, label="fake-data study (0.27 to 0.47)")
axs[1].set_xticks(x); axs[1].set_xticklabels(dv["country"], fontsize=8)
axs[1].set_title("Pair disagreement, as a multiple of real movement", fontsize=9, loc="left"); axs[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(out, "chart_noise_scale.png"), dpi=160); plt.close()
print(f"\nSaved tables, eda_report.txt and charts in {os.path.abspath(out)}")
