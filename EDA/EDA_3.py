"""Charts and numbers for the Q3 slides: guessing each month's official
exporter figure before it is published.

Reads Processed/side_by_side.csv. Writes to Charts/:
  guesses_gain.png      how much better the ship guess is than the best
                        simple guess, in each half of the test period
  guesses_miss.png      average miss each month of the best simple guess and
                        the ship guess, January 2022 on
  guesses_table.csv     the numbers behind both charts (every guess, every
                        country, all months and each half)
Run from the repo folder:  python EDA/EDA_3.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "Processed" / "side_by_side.csv"
CHARTS = ROOT / "Charts"
CHARTS.mkdir(exist_ok=True)

COUNTRIES = ["AUS", "BRA", "ZAF", "PER", "LBR"]
NAMES = {"AUS": "Australia", "BRA": "Brazil", "ZAF": "South Africa",
         "PER": "Peru", "LBR": "Liberia"}
TEST_START = "2022-01-01"       # guesses are scored from here on
SECOND_HALF = "2024-01-01"      # 2022-23 is the first half, 2024-26 the second
RATIO_MONTHS = 12               # ships guess looks back this many months
MIN_RATIOS = 6                  # and needs at least this many of them
BAR = 10                        # ship guess must be 10% better, each half

SHIP, SHIP_LIGHT = "#FFD347", "#FFE9A8"
GREY, NAVY, MUTED = "#9AA5B1", "#0E2841", "#5F6B76"

# Let's use the first of these fonts the computer has
installed = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Aptos", "Inter", "Liberation Sans", "DejaVu Sans"]:
    if name in installed:
        plt.rcParams["font.family"] = name
        break


# Part 1 - the table, one row per month for each country
side = pd.read_csv(TABLE)
months = sorted(side["month"].unique())


# Part 2 - three guesses of each month's official figure
# Two use only the official record, one also uses this month's ships
rows = []
for code in COUNTRIES:
    block = side[side["country"] == code].set_index("month").reindex(months)
    exporter = block["exporter_t"].tolist()
    ship = block["ship_t"].tolist()

    last_year, last_month, ship_ratio = [], [], []
    for i in range(len(months)):
        # same month last year, and last month
        last_year.append(exporter[i - 12] if i >= 12 else np.nan)
        last_month.append(exporter[i - 1] if i >= 1 else np.nan)

        # this month's ships divided by the ship / exporter ratio of the 12
        # months before this one (average of the months that have both)
        ratios = []
        for j in range(max(0, i - RATIO_MONTHS), i):
            if pd.notna(exporter[j]) and pd.notna(ship[j]) and exporter[j] > 0:
                ratios.append(ship[j] / exporter[j])
        if len(ratios) >= MIN_RATIOS and pd.notna(ship[i]):
            ship_ratio.append(ship[i] / (sum(ratios) / len(ratios)))
        else:
            ship_ratio.append(np.nan)

    # Part 3 - score the guesses, only months where all three and the
    # official figure exist
    guesses = pd.DataFrame({"month": months, "actual": exporter,
                            "last_year": last_year,
                            "last_month": last_month,
                            "ship_ratio": ship_ratio}).dropna()
    guesses = guesses[(guesses["month"] >= TEST_START)
                      & (guesses["actual"] > 0)]
    halves = [("all", guesses),
              ("2022-23", guesses[guesses["month"] < SECOND_HALF]),
              ("2024-26", guesses[guesses["month"] >= SECOND_HALF])]
    for label, part in halves:
        row = {"country": code, "period": label, "months": len(part)}
        for guess in ["last_year", "last_month", "ship_ratio"]:
            # average miss = mean absolute % error against the official figure
            miss = (part[guess] - part["actual"]).abs() / part["actual"]
            row[guess] = round(100 * miss.mean(), 1)
        rows.append(row)

table = pd.DataFrame(rows)
table["best_simple"] = table[["last_year", "last_month"]].min(axis=1)
table["best_ship"] = table["ship_ratio"]
# gain = how much smaller the ship miss is than the best simple miss
table["gain_pct"] = (100 * (1 - table["best_ship"]
                            / table["best_simple"])).round(1)
table["clears_bar"] = table["gain_pct"] >= BAR
table.to_csv(CHARTS / "guesses_table.csv", index=False)
print(table.to_string(index=False))

for label in ["2022-23", "2024-26"]:
    part = table[table["period"] == label]
    print(label, "countries that clear the %d%% bar:" % BAR,
          int(part["clears_bar"].sum()), "of", len(part))


# Part 4 - chart 1, gain over the best simple guess in each half
def label_bars(ax, bars, size=11):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                "%.0f" % bar.get_height(), ha="center", va="bottom",
                fontsize=size, color=NAVY)


def tidy(ax, ylabel):
    for edge in ["top", "right"]:
        ax.spines[edge].set_visible(False)
    ax.spines["left"].set_color("#C9D2DA")
    ax.spines["bottom"].set_color("#C9D2DA")
    ax.tick_params(labelsize=11, colors="#595959", length=0)
    ax.set_ylabel(ylabel, fontsize=11, color="#595959")
    ax.grid(axis="y", color="#E3E8EC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(range(len(COUNTRIES)))
    ax.set_xticklabels([NAMES[c] for c in COUNTRIES])


x = np.arange(len(COUNTRIES))
first = table[table["period"] == "2022-23"].set_index("country")
second = table[table["period"] == "2024-26"].set_index("country")
overall = table[table["period"] == "all"].set_index("country")

fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=250)
bars_one = ax.bar(x - 0.19, [first.loc[c, "gain_pct"] for c in COUNTRIES],
                  0.36, color=SHIP_LIGHT, label="2022-23", zorder=3)
bars_two = ax.bar(x + 0.19, [second.loc[c, "gain_pct"] for c in COUNTRIES],
                  0.36, color=SHIP, label="2024-26", zorder=3)
label_bars(ax, bars_one)
label_bars(ax, bars_two)
ax.axhline(BAR, color=NAVY, linewidth=1.4, linestyle="--", zorder=4,
           label="Bar to clear: %d%%" % BAR)
ax.set_ylim(0, 80)
tidy(ax, "Gain over the best simple guess (%)")
ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=1)
fig.tight_layout()
fig.savefig(CHARTS / "guesses_gain.png", facecolor="white")
plt.close(fig)


# Part 5 - chart 2, average miss each month over the whole test period
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=250)
bars_simple = ax.bar(x - 0.19, [overall.loc[c, "best_simple"]
                                for c in COUNTRIES],
                     0.36, color=GREY, label="Best simple guess", zorder=3)
bars_ship = ax.bar(x + 0.19, [overall.loc[c, "best_ship"]
                              for c in COUNTRIES],
                   0.36, color=SHIP, label="Ship-based guess", zorder=3)
for bars in [bars_simple, bars_ship]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                "%.1f" % bar.get_height(), ha="center", va="bottom",
                fontsize=11, color=NAVY)
ax.set_ylim(0, 35)
tidy(ax, "Average miss each month (%)")
ax.legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout()
fig.savefig(CHARTS / "guesses_miss.png", facecolor="white")
plt.close(fig)
