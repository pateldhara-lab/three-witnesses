"""Charts and numbers for the noise slides: how noisy is each record.

Reads Processed/side_by_side.csv. Writes to Charts/:
  noise_real_estimates.png      noise of each record in each country, as a
                                multiple of the real month-to-month movement
  noise_table.csv               the numbers behind that chart
Run from the repo folder:  python EDA/EDA_2.py
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

SHIP, SHIP_INK = "#FFD347", "#8F6D00"
IMPORTER, IMPORTER_INK = "#84E291", "#1F852E"
EXPORTER, EXPORTER_INK = "#5A97E0", "#2773CE"
NAVY, MUTED, TINT = "#0E2841", "#5F6B76", "#FFE9A8"

# Let's use the first of these fonts the computer has
installed = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Aptos", "Inter", "Liberation Sans", "DejaVu Sans"]:
    if name in installed:
        plt.rcParams["font.family"] = name
        break


# Part 1 - the table, 72 months per country (China's data stops in 2024)
side = pd.read_csv(TABLE)
side["month"] = pd.to_datetime(side["month"])
months = pd.date_range("2019-01-01", "2024-12-01", freq="MS")


# Part 2 - noise by triple collocation on year-over-year changes
def year_changes(values):
    """A month minus the same month a year earlier (72 months become 60)."""
    return values[12:] - values[:-12]


def triple_collocation(exporter, importer, ship):
    """Noise divided by signal for each record, in that order. A record gets
    nan when its noise comes out negative (an impossible answer)."""
    data = np.column_stack([exporter, importer, ship])
    data = data[~np.isnan(data).any(axis=1)]
    cov = np.cov(data, rowvar=False)
    out = []
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        signal = cov[i, j] * cov[i, k] / cov[j, k]
        noise = cov[i, i] - signal
        if noise < 0 or signal <= 0:
            out.append(np.nan)
        else:
            out.append(np.sqrt(noise / signal))
    return out, len(data)


rows = []
for code in COUNTRIES:
    block = side[side["country"] == code].set_index("month").reindex(months)
    exporter = block["exporter_t"].to_numpy().copy()
    importer = block["importer_aligned_total_t"].to_numpy().copy()
    ship = block["ship_t"].to_numpy().copy()
    # Months we know are wrong are blanked in all three records
    bad = block["flag"].isin(["missing", "known_source_error"]).to_numpy()
    exporter[bad], importer[bad], ship[bad] = np.nan, np.nan, np.nan
    estimates, used = triple_collocation(year_changes(exporter),
                                         year_changes(importer),
                                         year_changes(ship))
    rows.append({"country": code, "rows_used": used,
                 "exporter": estimates[0], "importer": estimates[1],
                 "ship": estimates[2]})
noise = pd.DataFrame(rows)
noise.round(3).to_csv(CHARTS / "noise_table.csv", index=False)
print(noise.round(2).to_string())


# Part 3 - chart 1, the dot plot of real estimates
fig = plt.figure(figsize=(7.0, 4.26), dpi=250)
ax = fig.add_axes([0.2, 0.14, 0.78, 0.84])
ax.set_xlim(0, 3.6)
ax.set_ylim(3.3, 0)
ax.axvspan(0.1, 0.4, color=TINT, zorder=0)
ax.text(0.12, 0.28, "Noise planted in the test: 0.1 to 0.4", fontsize=10,
        fontweight="bold", color=NAVY, va="center")
for edge in ["top", "right", "left"]:
    ax.spines[edge].set_visible(False)
ax.set_yticks([])
ax.set_xticks([0, 1, 2, 3])
ax.tick_params(axis="x", labelsize=11, colors="#595959")
ax.set_xlabel("Noise as a multiple of the real month-to-month movement",
              fontsize=11, color="#595959")

side_axis = ax.get_yaxis_transform()
nudge = {("importer", "ZAF"): -0.1, ("importer", "AUS"): 0.1}
where = {("ship", "AUS"): "below", ("importer", "AUS"): "below",
         ("importer", "BRA"): "below"}
lanes = [("exporter", 1.0, EXPORTER, EXPORTER_INK, "Exporter's\nown figure"),
         ("ship", 1.95, SHIP, SHIP_INK, "Ship\ntracking"),
         ("importer", 2.7, IMPORTER, IMPORTER_INK, "Importers'\ncustoms")]
for record, y, fill, ink, label in lanes:
    ax.plot([0, 3.6], [y, y], color="#E3E8EC", linewidth=1.5, zorder=1)
    ax.text(-0.03, y, label, transform=side_axis, ha="right", va="center",
            fontsize=11, fontweight="bold", color=ink, linespacing=1.1)
    shown = noise[noise[record].notna()]
    for i in range(len(shown)):
        x = float(shown[record].iloc[i])
        code = shown["country"].iloc[i]
        # a few dots sit close together, so they get their own spot
        dot_y = y + nudge.get((record, code), 0)
        place = where.get((record, code), "above")
        ax.scatter([x], [dot_y], s=95, facecolor=fill, edgecolor="white",
                   linewidth=1.4, zorder=3)
        if place == "below":
            ax.text(x, dot_y + 0.17, NAMES[code], ha="center", va="top",
                    fontsize=9.5, color=NAVY)
        else:
            lift = 0.36 if place == "higher" else 0.17
            ax.text(x, dot_y - lift, NAMES[code], ha="center", va="bottom",
                    fontsize=9.5, color=NAVY)
missing = noise[noise["exporter"].isna()]
ax.text(3.58, 1.0, "No valid estimate:\n" + "  ".join(
    NAMES[c] for c in missing["country"]), ha="right", va="center",
    fontsize=9.5, style="italic", color="#595959")
fig.savefig(CHARTS / "noise_real_estimates.png", facecolor="white")
plt.close(fig)
