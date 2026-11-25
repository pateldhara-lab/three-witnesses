"""Report 2, Figure 3 and Table 3 - scale by year and noise on real records.

Needs the methods code, so it runs once Methods/src is in place. Reads
Processed/side_by_side.csv, coverage_by_year.csv and steadiness_test.csv.
Writes to Charts/:
  report2_table3_noise.csv          noise-to-signal ratios with 90% intervals
  report2_fig3_scale_and_noise.png  coverage by year and the noise ratios
Run from the repo folder:  python EDA/EDA_7.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Processed"
CHARTS = ROOT / "Charts"
CHARTS.mkdir(exist_ok=True)

# Let's borrow the methods code (triple collocation and the bootstrap)
sys.path.insert(0, str(ROOT / "Methods" / "src"))
import table_io
from triple_collocation import triple_collocation
from study import bootstrap_ranges

COUNTRIES = ["AUS", "BRA", "LBR", "PER", "ZAF"]
NAMES = {"AUS": "Australia", "BRA": "Brazil", "LBR": "Liberia",
         "PER": "Peru", "ZAF": "South Africa"}
NAVY, GREY = "#0E2841", "#5F6B76"
EXPORTER, EXPORTER_INK = "#5A97E0", "#2773CE"
IMPORTER, IMPORTER_INK = "#84E291", "#1F852E"
SHIP, SHIP_INK = "#FFD347", "#8F6D00"
RESAMPLES, BLOCK, SEED = 500, 6, 20261123

installed = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Aptos", "Inter", "Liberation Sans", "DejaVu Sans"]:
    if name in installed:
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#B0B7BE",
                     "axes.linewidth": 0.6, "xtick.color": GREY,
                     "ytick.color": GREY})

coverage = pd.read_csv(PROCESSED / "coverage_by_year.csv")
coverage = coverage[coverage["year"].between(2019, 2024)]
steady = pd.read_csv(PROCESSED / "steadiness_test.csv")


def is_steady(country):
    """True when ships and importers both pass the 60% floor and range."""
    rows = steady[(steady["country"] == country)
                  & steady["series"].isin(["importer_cover_aligned",
                                           "ship_cover"])]
    return bool(((rows["passes_60_floor"] == "yes")
                 & (rows["passes_7_point_range"] == "yes")).all())


# Part 1 - Table 3, triple collocation on the real records
# One Liberia month has a unit slip that we corrected, so it stays in
table_io.VALID_FLAGS = table_io.VALID_FLAGS | {"unit_slip_corrected"}
data = table_io.load_table(PROCESSED / "side_by_side.csv")
rng = np.random.default_rng(SEED)
results = {}
lines = ["country,rows,exporter,exporter_low,exporter_high,importer,"
         "importer_low,importer_high,ships,ships_low,ships_high,"
         "r_exporter_importer,r_exporter_ships,r_importer_ships,verdict"]
for country in COUNTRIES:
    one = table_io.country_yoy(data, country,
                               exclude_flags=("missing",
                                              "known_source_error"))
    estimate = triple_collocation(one.yoy).rel_noise
    low, high, negative, unused = bootstrap_ranges(one.yoy, RESAMPLES, BLOCK,
                                                   rng)
    corr = np.corrcoef(one.yoy, rowvar=False)

    # Three checks - steady scale, a finite estimate, ships and importers
    # correlated at 0.3 or more - and the verdict counts how many fail
    checks = [is_steady(country), bool(np.isfinite(estimate[0])),
              bool(corr[1, 2] >= 0.3)]
    fails = 3 - sum(checks)
    verdict = ["identified", "weak", "not identified",
               "not identified"][fails]
    results[country] = (estimate, low, high)

    # One row per country, a negative noise variance is written as undefined
    row = [country, str(one.n_yoy_rows)]
    for i in range(3):
        row.append("undefined" if not np.isfinite(estimate[i])
                   else "%.2f" % estimate[i])
        row.append("%.2f" % low[i])
        row.append("%.2f" % high[i])
    row += ["%.2f" % corr[0, 1], "%.2f" % corr[0, 2], "%.2f" % corr[1, 2],
            verdict]
    lines.append(",".join(row))
    print(country, one.n_yoy_rows, np.round(estimate, 2), verdict)
with open(CHARTS / "report2_table3_noise.csv", "w") as file:
    file.write("\n".join(lines) + "\n")

# Part 2 - Figure 3
fig = plt.figure(figsize=(6.5, 3.0))
fig.text(0.0, 0.965, "(a) Share of the exporter total seen by each record "
         "(%), by year", fontsize=8, fontweight="bold", color=NAVY)
for k in range(len(COUNTRIES)):
    country = COUNTRIES[k]
    ax = fig.add_axes([0.055 + k * 0.19, 0.58, 0.16, 0.30])
    one = coverage[coverage["country"] == country]
    ax.axhline(100, color=EXPORTER, lw=1.0, ls=":")
    ax.plot(one["year"], one["ship_cover_pct"], color=SHIP, lw=1.6,
            marker="o", ms=2.5)
    ax.plot(one["year"], one["importer_cover_aligned_pct"], color=IMPORTER,
            lw=1.6, marker="o", ms=2.5)
    ax.set_ylim(60, 170)
    ax.set_xticks([2019, 2021, 2023])
    ax.set_xticklabels(["2019", "2021", "2023"], fontsize=6.5)
    ax.tick_params(labelsize=6.5, length=2)
    if k > 0:
        ax.set_yticklabels([])
    ax.spines[["top", "right"]].set_visible(False)
    label = "steady" if is_steady(country) else "not steady"
    ax.set_title(NAMES[country] + ": " + label, fontsize=7.2, pad=3,
                 fontweight="bold",
                 color=NAVY if is_steady(country) else "#B4462B")
fig.text(0.62, 0.965, "ships", color=SHIP_INK, fontsize=7.2,
         fontweight="bold")
fig.text(0.68, 0.965, "importers (after lag)", color=IMPORTER_INK,
         fontsize=7.2, fontweight="bold")
fig.text(0.84, 0.965, "exporter = 100", color=EXPORTER_INK, fontsize=7.2,
         fontweight="bold")

fig.text(0.0, 0.46, "(b) Noise-to-signal ratio on the real records, with "
         "90% intervals (band = planted range, 0.10 to 0.40)", fontsize=8,
         fontweight="bold", color=NAVY)
ax = fig.add_axes([0.055, 0.08, 0.93, 0.29])
ax.axhspan(0.10, 0.40, color="#E6E9EC", zorder=0)
records = [(EXPORTER, "exporter"), (IMPORTER, "importer"), (SHIP, "ships")]
top = 4.0
for k in range(len(COUNTRIES)):
    estimate, low, high = results[COUNTRIES[k]]
    for i in range(3):
        x = k + (i - 1) * 0.22
        color = records[i][0]
        ax.plot([x, x], [low[i], min(high[i], top)], color=color, lw=1.6,
                zorder=2)
        if high[i] > top:
            ax.text(x + 0.04, top - 0.1, "to %.1f" % high[i], fontsize=5.8,
                    color=GREY, ha="left", va="top")
        if np.isfinite(estimate[i]):
            ax.scatter([x], [estimate[i]], s=22, color=color, ec=NAVY,
                       lw=0.5, zorder=3)
        else:
            ax.scatter([x], [0], s=22, facecolor="white", ec=color, lw=1.2,
                       zorder=3)
ax.set_xticks(range(5))
ax.set_xticklabels([NAMES[c] for c in COUNTRIES], fontsize=7)
ax.set_xlim(-0.5, 4.8)
ax.set_ylim(0, top)
ax.tick_params(labelsize=6.5, length=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#EEF0F2", lw=0.5)
fig.text(0.62, 0.395, "exporter", color=EXPORTER_INK, fontsize=7.2,
         fontweight="bold")
fig.text(0.70, 0.395, "importer", color=IMPORTER_INK, fontsize=7.2,
         fontweight="bold")
fig.text(0.78, 0.395, "ships", color=SHIP_INK, fontsize=7.2,
         fontweight="bold")
fig.text(0.84, 0.395, "open = undefined", color=GREY, fontsize=7)
fig.savefig(CHARTS / "report2_fig3_scale_and_noise.png", dpi=300)
plt.close(fig)
