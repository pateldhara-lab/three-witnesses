"""Charts and numbers for the Q1 evidence slides: gaps between the records.

Reads Processed/side_by_side.csv, Processed/coverage_by_year.csv and the
clean files in Processed/by_source. Writes to Charts/:
  usual_ratio_level_drift.png   how each record compares with the exporter,
                                before and after 2022
  usual_ratio_table.csv         the numbers behind that chart
  coverage_summary.csv          how much of the exporter's tonnes each
                                witness covers, 2019 to 2024
  brazil_jan_2021.csv           clean files to one side-by-side row
Run from the repo folder:  python EDA/EDA_1.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "Processed" / "side_by_side.csv"
COVERAGE = ROOT / "Processed" / "coverage_by_year.csv"
BY_SOURCE = ROOT / "Processed" / "by_source"
CHARTS = ROOT / "Charts"
CHARTS.mkdir(exist_ok=True)

COUNTRIES = ["AUS", "BRA", "ZAF", "PER", "LBR"]
NAMES = {"AUS": "Australia", "BRA": "Brazil", "ZAF": "South Africa",
         "PER": "Peru", "LBR": "Liberia"}
FIT_QUARTERS = 16               # 2019 to 2022 is the fit window
SHUFFLES = 20000

# Palette of the deck
SHIP, SHIP_INK = "#FFD347", "#8F6D00"
IMPORTER, IMPORTER_INK = "#84E291", "#1F852E"
EXPORTER, EXPORTER_INK = "#5A97E0", "#2773CE"
NAVY, MUTED, TINT = "#0E2841", "#5F6B76", "#FFF6D6"

# Let's use the first of these fonts the computer has
installed = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Aptos", "Inter", "Liberation Sans", "DejaVu Sans"]:
    if name in installed:
        plt.rcParams["font.family"] = name
        break


# the side-by-side table
side = pd.read_csv(TABLE)
side["month"] = pd.to_datetime(side["month"])

months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")

records = {}
for code in COUNTRIES:
    block = side[side["country"] == code].set_index("month").reindex(months)
    records[code] = {"exporter": block["exporter_t"].to_numpy(),
                     "ship": block["ship_t"].to_numpy(),
                     "importer": block["importer_aligned_total_t"].to_numpy(),
                     "flag": block["flag"].to_numpy()}


def to_quarters(values):
    """Sums months into quarters, blank unless all three months are there."""
    out = []
    for q in range(len(values) // 3):
        out.append(values[q * 3:q * 3 + 3].sum())
    return np.array(out)


def ratios_for(code, grain):
    """Returns ship / exporter and importers / exporter by month or quarter."""
    exporter, ship, importer = (records[code]["exporter"],
                                records[code]["ship"],
                                records[code]["importer"])
    if grain == "quarter":
        exporter, ship, importer = (to_quarters(exporter), to_quarters(ship),
                                    to_quarters(importer))
    return ship / exporter, importer / exporter


# Part 2 - usual ratio in the fit and test years
def shuffle_p_value(fit_part, test_part):
    """How often a random split of the same quarters moves the median ratio
    as much as the real split did."""
    rng = np.random.default_rng(5500)
    pooled = np.concatenate([fit_part, test_part])
    move = abs(np.median(test_part) / np.median(fit_part) - 1)
    count = 0
    for i in range(SHUFFLES):
        shuffled = rng.permutation(pooled)
        new_fit = shuffled[:len(fit_part)]
        new_test = shuffled[len(fit_part):]
        if abs(np.median(new_test) / np.median(new_fit) - 1) >= move:
            count = count + 1
    return (count + 1) / (SHUFFLES + 1)


rows = []
for code in COUNTRIES:
    ship_ratio, importer_ratio = ratios_for(code, "quarter")
    both = ~np.isnan(ship_ratio) & ~np.isnan(importer_ratio)
    quarter = np.arange(len(ship_ratio))
    fit = both & (quarter < FIT_QUARTERS)
    test = both & (quarter >= FIT_QUARTERS)
    for record, ratio in [("ship", ship_ratio), ("importer", importer_ratio)]:
        fit_median, test_median = np.median(ratio[fit]), np.median(ratio[test])
        rows.append({"country": code, "record": record,
                     "fit_quarters": fit.sum(), "test_quarters": test.sum(),
                     "usual_2019_22": fit_median, "usual_2023_24": test_median,
                     "move_pct": 100 * (test_median / fit_median - 1),
                     "p_value": shuffle_p_value(ratio[fit], ratio[test])})
usual = pd.DataFrame(rows)
usual.round(4).to_csv(CHARTS / "usual_ratio_table.csv", index=False)
print(usual.round(3).to_string())


# Part 3 - chart 1, the usual ratio before and after 2022
fig = plt.figure(figsize=(7.7, 4.15), dpi=250)
ax = fig.add_axes([0.31, 0.075, 0.47, 0.84])
ax.set_xlim(0.6, 1.6)
ax.set_ylim(5, 0)
for edge in ["top", "right", "left"]:
    ax.spines[edge].set_visible(False)
ax.spines["bottom"].set_color("#9AA5AE")
ax.set_yticks([])
ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.tick_params(axis="x", labelsize=11, colors=MUTED, length=0)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#E3E8EC", linewidth=0.8)
ax.axvline(1.0, color="#9AA5AE", linewidth=1.3, linestyle="--", zorder=1)
ax.text(1.0, -0.12, "exporter = 1.0", ha="center", va="bottom", fontsize=10.5,
        fontweight="bold", color=MUTED)

side_axis = ax.get_yaxis_transform()    # x as a share of the plot, y as data
for i in range(len(COUNTRIES)):
    code = COUNTRIES[i]
    if i > 0:
        ax.plot([-0.64, 1.30], [i, i], color="#D5DBE0", linewidth=0.8,
                transform=side_axis, clip_on=False)
    ax.text(-0.64, i + 0.5, NAMES[code], transform=side_axis, ha="left",
            va="center", fontsize=12, fontweight="bold", color=NAVY)
    for record, lane, fill, ink, label in [
            ("ship", 0.27, SHIP, SHIP_INK, "Ships"),
            ("importer", 0.73, IMPORTER, IMPORTER_INK, "Importers")]:
        row = usual[(usual["country"] == code) & (usual["record"] == record)]
        before = float(row["usual_2019_22"].iloc[0])
        after = float(row["usual_2023_24"].iloc[0])
        real = float(row["p_value"].iloc[0]) < 0.05    # a move beyond luck
        y = i + lane
        if real:
            ax.add_patch(Rectangle((-0.30, y - 0.17), 1.60, 0.34,
                                   transform=side_axis, color=TINT,
                                   clip_on=False, zorder=0))
        ax.text(-0.27, y, label, transform=side_axis, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color=ink)
        if round(after, 2) != round(before, 2):
            ax.plot([before, after], [y, y], color=ink, linewidth=2.4,
                    zorder=2)
        ax.scatter([before], [y], s=62, facecolor="white", edgecolor=ink,
                   linewidth=1.7, zorder=3)
        ax.scatter([after], [y], s=62, facecolor=fill, edgecolor=ink,
                   linewidth=1.0, zorder=4)
        if real:
            ax.text(after + 0.03, y, "%+.0f%%" % (100 * (after / before - 1)),
                    ha="left", va="center", fontsize=11, fontweight="bold",
                    color=NAVY, zorder=5)
        if round(after, 2) != round(before, 2):
            text = "%.2f → %.2f" % (before, after)
        else:
            text = "%.2f" % after
        ax.text(1.30, y, text, transform=side_axis, ha="right", va="center",
                fontsize=11, fontweight="bold" if real else "normal",
                color=NAVY if real else "#595959")
fig.savefig(CHARTS / "usual_ratio_level_drift.png", facecolor="white")
plt.close(fig)


# Part 4 - how much of the exporter's tonnes each witness covers
coverage = pd.read_csv(COVERAGE)
coverage = coverage[(coverage["year"] >= 2019) & (coverage["year"] <= 2024)]
summary = []
for code in COUNTRIES:
    block = coverage[coverage["country"] == code]
    weight = block["exporter_t"]
    summary.append({
        "country": code,
        "importers_pct": (block["importer_cover_raw_pct"] * weight).sum()
        / weight[block["importer_cover_raw_pct"].notna()].sum(),
        "ships_pct": (block["ship_cover_pct"] * weight).sum() / weight.sum(),
        "exporter_mt_a_year": weight.mean() / 1e6})
summary = pd.DataFrame(summary)
summary.round(1).to_csv(CHARTS / "coverage_summary.csv", index=False)
print(summary.round(1).to_string())


# Part 5 - January 2021 in Brazil, from the clean files to one row
def total_for(name, **keep):
    data = pd.read_csv(BY_SOURCE / (name + ".csv"))
    data = data[(data["exporter"] == "BRA") & (data["month"] == "2021-01-01")]
    for column in keep:
        data = data[data[column] == keep[column]]
    return data["tonnes"].sum(), len(data)


case = side[(side["country"] == "BRA") & (side["month"] == "2021-01-01")]
case = case.iloc[0]
brazil_rows = [
    ("brazil_exporter", "to China", total_for("brazil_exporter",
                                               counterparty="CHN")),
    ("china_importer", "China", total_for("china_importer")),
    ("eu_importer", "EU27", total_for("eu_importer")),
    ("portwatch_ship", "Itaqui", total_for("portwatch_ship", port="port506"))]
table = pd.DataFrame([{"source": s, "what": w, "tonnes_mt": t[0] / 1e6,
                       "rows": t[1]} for s, w, t in brazil_rows])
table.round(2).to_csv(CHARTS / "brazil_jan_2021.csv", index=False)
print(table.round(2).to_string())
print("side-by-side row:", round(case["exporter_t"] / 1e6, 2),
      round(case["ship_t"] / 1e6, 2), round(case["china_raw_t"] / 1e6, 2),
      round(case["eu_raw_t"] / 1e6, 2), case["lag_china"], case["lag_eu"],
      round(case["importer_aligned_total_t"] / 1e6, 2))
