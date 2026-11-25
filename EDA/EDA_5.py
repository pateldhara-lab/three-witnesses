"""Presentation 1 - the two first-look charts on the real tables.

Reads Processed/lag_fit.csv and Processed/coverage_by_year.csv.
Writes to Charts/:
  first_look_lag_levels.csv / .png   China's record against the exporter's,
                                     correlation of monthly levels by lag
  first_look_ship_share.csv / .png   ship total as % of the exporter's tonnes
Run from the repo folder:  python EDA/EDA_5.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "Charts"
CHARTS.mkdir(exist_ok=True)

COUNTRIES = ["BRA", "PER", "ZAF"]
NAMES = {"BRA": "Brazil", "PER": "Peru", "ZAF": "South Africa"}
COLORS = {"BRA": "#0E2841", "PER": "#A02B93", "ZAF": "#E97132"}
MARKERS = {"BRA": "o", "PER": "^", "ZAF": "s"}

# Part 1 - China's record against the exporter's, by lag
# The fit file has one row per country, importer and lag (0 to 3 months)
fit = pd.read_csv(ROOT / "Processed" / "lag_fit.csv")
fit = fit[(fit["importer"] == "CHN") & (fit["country"].isin(COUNTRIES))]
levels = fit.pivot(index="lag", columns="country", values="corr_levels")
levels = levels[COUNTRIES].round(2)
levels.to_csv(CHARTS / "first_look_lag_levels.csv")
print(levels.to_string())

# Part 2 - ship tonnes as a share of the exporter's own tonnes, by year
# 2026 only has part of the year, so we leave it out (same as the slide)
coverage = pd.read_csv(ROOT / "Processed" / "coverage_by_year.csv")
coverage = coverage[coverage["country"].isin(COUNTRIES)]
coverage = coverage[coverage["year"].between(2019, 2025)]
share = coverage.pivot(index="year", columns="country",
                       values="ship_cover_pct")[COUNTRIES].round(0)
share.to_csv(CHARTS / "first_look_ship_share.csv")
print(share.to_string())

# Part 3 - the charts
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for country in COUNTRIES:
    ax.plot(levels.index, levels[country], color=COLORS[country],
            marker=MARKERS[country], lw=1.8, label=NAMES[country])
ax.set_ylim(0, 1)
ax.set_xticks(range(4))
ax.set_xlabel("months China's figures are moved back")
ax.set_ylabel("correlation")
ax.set_title("China's record against the exporter's, by lag\n"
             "correlation of monthly levels, Jan 2019 to Sep 2024",
             fontsize=9, loc="left")
ax.legend(frameon=False, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, -0.4))
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(CHARTS / "first_look_lag_levels.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
for country in COUNTRIES:
    ax.plot(share.index, share[country], color=COLORS[country],
            marker=MARKERS[country], lw=1.8, label=NAMES[country])
ax.set_ylim(60, 120)
ax.set_xlabel("year")
ax.set_ylabel("% of exporter's tonnes")
ax.set_title("Ships against the exporter's own count, by year\n"
             "ship total as % of exporter's tonnes, 2026 left out",
             fontsize=9, loc="left")
ax.legend(frameon=False, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, -0.4))
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(CHARTS / "first_look_ship_share.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
