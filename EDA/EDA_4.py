"""Report 2 figures that come straight from the processed tables.

Reads Processed/side_by_side.csv, coverage_by_year.csv, lag_fit.csv,
lag_table.csv and the clean files in Processed/by_source. Writes to Charts/:
  report2_fig1_pipeline_steps.png   the four pipeline steps with row counts
  report2_fig2_lag_fit.png          how the importer lag is fitted
  report2_fig4_same_flow.png        yearly ratio and lag curve, same flow
  report2_fig5_three_records.png    what each record sees, Brazil
Run from the repo folder:  python EDA/EDA_4.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Processed"
BY_SOURCE = PROCESSED / "by_source"
CHARTS = ROOT / "Charts"
CHARTS.mkdir(exist_ok=True)

WINDOW = ("2019-01-01", "2024-12-01")
COUNTRIES = ["AUS", "BRA", "LBR", "PER", "ZAF"]
NAMES = {"AUS": "Australia", "BRA": "Brazil", "LBR": "Liberia",
         "PER": "Peru", "ZAF": "South Africa", "CHN": "China",
         "EU27": "EU27"}

NAVY, GREY, LIGHT = "#0E2841", "#5F6B76", "#F1F4F6"
EXPORTER, EXPORTER_INK = "#5A97E0", "#2773CE"
IMPORTER, IMPORTER_INK = "#84E291", "#1F852E"
SHIP, SHIP_INK = "#FFD347", "#8F6D00"

# Let's use the first of these fonts the computer has
installed = {f.name for f in font_manager.fontManager.ttflist}
for name in ["Aptos", "Inter", "Liberation Sans", "DejaVu Sans"]:
    if name in installed:
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#B0B7BE",
                     "axes.linewidth": 0.6, "xtick.color": GREY,
                     "ytick.color": GREY, "axes.labelcolor": GREY})

table = pd.read_csv(PROCESSED / "side_by_side.csv", parse_dates=["month"])
table = table[(table["month"] >= WINDOW[0]) & (table["month"] <= WINDOW[1])]
coverage = pd.read_csv(PROCESSED / "coverage_by_year.csv")
coverage = coverage[coverage["year"].between(2019, 2024)]
fit = pd.read_csv(PROCESSED / "lag_fit.csv")
lags = pd.read_csv(PROCESSED / "lag_table.csv")


def box(ax, x, y, width, height, fill, edge="none", dashed=False):
    """One rounded box on an axis that runs from 0 to 1."""
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0,rounding_size=0.01",
                                fc=fill, ec=edge, lw=0.8,
                                ls="--" if dashed else "-"))


def load_clean(name):
    """One clean file as text, with a flag for the 2019 to 2024 window."""
    data = pd.read_csv(BY_SOURCE / (name + ".csv"), dtype=str,
                       keep_default_na=False)
    data["in_window"] = (data["month"] >= WINDOW[0]) \
        & (data["month"] <= WINDOW[1])
    return data


# Part 1 - Figure 1, the four pipeline steps with the row counts
# Rows for our five exporters only, the importer files hold a few more
exporter_rows = pd.concat([load_clean(name) for name in [
    "brazil_exporter", "south_africa_exporter", "peru_exporter",
    "liberia_exporter", "australia_exporter"]])
importer_rows = pd.concat([load_clean("china_importer"),
                           load_clean("eu_importer")])
importer_rows = importer_rows[importer_rows["exporter"].isin(COUNTRIES)]
ship_rows = load_clean("portwatch_ship")
ship_rows = ship_rows[ship_rows["exporter"].isin(COUNTRIES)]

# Ports that feed the ship total, the others are dropped in the link step
ports = pd.read_csv(ROOT / "Lookups" / "ports.csv", dtype=str,
                    keep_default_na=False)
total_ports = set(ports[(ports["in_country_ship_total"] == "yes")
                        & (ports["feeds_table_column"] == "ship_t")]["portid"])
ships_in_window = ship_rows[ship_rows["in_window"]]
exporter_total = len(exporter_rows)
exporter_outside = int((~exporter_rows["in_window"]).sum())
importer_total = len(importer_rows)
importer_outside = int((~importer_rows["in_window"]).sum())
ship_total = len(ship_rows)
ship_outside = int((~ship_rows["in_window"]).sum())
other_ports = int((~ships_in_window["port"].isin(total_ports)).sum())
all_three = table[["exporter_t", "ship_t",
                   "importer_aligned_total_t"]].notna().all(axis=1)
print(exporter_total, exporter_outside, importer_total, importer_outside,
      ship_total, ship_outside, other_ports, len(table), int(all_three.sum()))

# Each step: number, name, script, colour, input, operations, output
steps = [
    ("Step 1", "Cleaning", "8 loader scripts", EXPORTER,
     "8 source files as\npublished (spreadsheets,\nCSV, API rows, PDF)",
     ["1  convert units to tonnes", "2  cut calendars into months",
      "3  match names to ISO\n    codes and port IDs"],
     "8 clean files\n(CSV, nine columns)"),
    ("Step 2", "Linking", "build script", IMPORTER,
     f"clean files, five exporters:\n{exporter_total:,} exporter, "
     f"{importer_total:,}\nimporter, {ship_total:,} ship rows",
     ["1  ingest the clean files",
      f"2  filter out rows outside\n    2019-2024 ({exporter_outside:,}, "
      f"{importer_outside}, {ship_outside})\n    and {other_ports} ship rows "
      "on\n    ports outside the totals",
      "3  sum destinations and\n    named ports"],
     "one value per country,\nmonth and record"),
    ("Step 3", "Coverage", "build script", SHIP,
     "monthly values of\nthe three records",
     ["1  divide importer total and\n    ship total by exporter\n    total, "
      "per country and year",
      "2  take the yearly range\n    of each share"],
     "coverage table\n(feeds the scale check)"),
    ("Step 4", "Lag", "build script", "#0B76A0",
     "monthly exporter and\nimporter values (China,\nEU27)",
     ["1  shift each importer series\n    back by 0 to 3 months",
      "2  keep the lag with the highest\n    correlation of yearly changes",
      "3  add aligned China and EU27"],
     f"side-by-side table:\n{len(table)} country-months,\n"
     f"{int(all_three.sum())} with all three")]

fig = plt.figure(figsize=(6.5, 2.7))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Column left edge, column width and the gap between columns
left, width, gap = 0.075, 0.222, 0.012
# Bottom edge and height of the four rows
rows = {"head": (0.858, 0.130), "in": (0.637, 0.193), "op": (0.234, 0.37),
        "out": (0.005, 0.193)}
for key, label in [("in", "Input"), ("op", "Operations"),
                   ("out", "Output")]:
    y, height = rows[key]
    ax.text(0.035, y + height / 2, label, rotation=90, ha="center",
            va="center", fontsize=7.5, fontweight="bold", color=GREY)

for k in range(len(steps)):
    number, name, script, colour, step_in, operations, step_out = steps[k]
    x = left + k * (width + gap)

    # Header with the step name and the script that does it
    y, height = rows["head"]
    box(ax, x, y, width, height, NAVY)
    ax.text(x + 0.012, y + height * 0.68, number + " · " + name,
            fontsize=8.5, fontweight="bold", color="white", va="center")
    ax.text(x + 0.012, y + height * 0.27, script, fontsize=6.8,
            color="#C9D3DD", va="center")
    ax.add_patch(plt.Rectangle((x, y), 0.006, height, fc=colour))

    # Input, operations and output boxes
    y, height = rows["in"]
    box(ax, x, y, width, height, LIGHT)
    ax.text(x + 0.01, y + height / 2, step_in, fontsize=6.7, color=NAVY,
            va="center", linespacing=1.35)
    y, height = rows["op"]
    box(ax, x, y, width, height, "white", edge="#9AA3AB")
    ax.text(x + 0.01, y + height - 0.02, "\n".join(operations),
            fontsize=6.7, color=NAVY, va="top", linespacing=1.4)
    y, height = rows["out"]
    box(ax, x, y, width, height, "#F6C6AD" if k == 3 else "#DCE6F0")
    ax.text(x + 0.01, y + height / 2, step_out, fontsize=6.9, color=NAVY,
            va="center", fontweight="bold", linespacing=1.35)

    # Arrows down the column, and across to the next step
    for top, bottom in [(rows["in"][0], rows["op"][0] + rows["op"][1]),
                        (rows["op"][0], rows["out"][0] + rows["out"][1])]:
        ax.annotate("", xy=(x + width / 2, bottom + 0.002),
                    xytext=(x + width / 2, top - 0.002),
                    arrowprops=dict(arrowstyle="-|>", color="#9AA3AB",
                                    lw=0.8, mutation_scale=7))
    if k < 3:
        ax.annotate("", xy=(x + width + gap + 0.001, rows["head"][0] + 0.052),
                    xytext=(x + width - 0.001, rows["head"][0] + 0.052),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0,
                                    mutation_scale=7))
fig.savefig(CHARTS / "report2_fig1_pipeline_steps.png", dpi=300)
plt.close(fig)


# Part 2 - Figure 2, how the importer lag is fitted
# Left: Brazil's yearly changes against China's, before and after the lag
brazil_lags = lags[lags["country"] == "BRA"].set_index("importer")
lag_china = int(brazil_lags.loc["CHN", "fitted_lag"])
lag_eu = int(brazil_lags.loc["EU27", "fitted_lag"])

span = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
brazil = pd.read_csv(BY_SOURCE / "brazil_exporter.csv", parse_dates=["month"])
exporter = brazil.groupby("month")["tonnes"].sum().reindex(span)
china = pd.read_csv(BY_SOURCE / "china_importer.csv", parse_dates=["month"])
china = china[china["exporter"] == "BRA"].set_index("month")["tonnes"]
china = china.reindex(span)

# Change on the same month a year earlier, in million tonnes
exporter_change = (exporter - exporter.shift(12)) / 1e6
china_change = (china - china.shift(12)) / 1e6
china_moved = china.shift(-lag_china)
china_moved_change = (china_moved - china_moved.shift(12)) / 1e6
shown = (span >= "2021-01-01") & (span <= "2023-12-01")

fig = plt.figure(figsize=(6.5, 2.7))
ax = fig.add_axes([0.075, 0.13, 0.36, 0.70])
ax.axhline(0, color="#B0B7BE", lw=0.6)
ax.plot(span[shown], exporter_change[shown], color=EXPORTER, lw=1.5)
ax.plot(span[shown], china_change[shown], color=IMPORTER, lw=1.1, ls="--")
ax.plot(span[shown], china_moved_change[shown], color=IMPORTER_INK, lw=1.5)
ax.text(span[shown][1], 8.6, "Brazil's exports (all destinations)",
        color=EXPORTER_INK, fontsize=6.8)
ax.text(span[shown][1], -9.6, "China's imports from Brazil: as recorded "
        "(dashed)", color=IMPORTER_INK, fontsize=6.8)
ax.text(span[shown][1], -11.3, f"and shifted back {lag_china} months "
        "(solid)", color=IMPORTER_INK, fontsize=6.8)
ax.set_ylim(-12, 10)
ax.set_ylabel("change on same month a year earlier (Mt)")
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.text(0.0, 0.955, f"(a) Brazil to China: lag 0 against lag {lag_china}",
         fontsize=8.5, fontweight="bold", color=NAVY)

# Right: correlation at lags 0 to 3 for every route, the chosen lag boxed
ax = fig.add_axes([0.715, 0.13, 0.275, 0.70])
routes = lags.sort_values(["importer", "corr_yoy"], ascending=[True, False])
grid, labels = [], []
for i in range(len(routes)):
    route = routes.iloc[i]
    one_route = fit[(fit["country"] == route["country"])
                    & (fit["importer"] == route["importer"])]
    grid.append(one_route.sort_values("lag")["corr_yoy"].values)
    sailing = route["sailing_time"].replace("about ", "")
    labels.append(NAMES[route["country"]] + " → " + NAMES[route["importer"]]
                  + ", " + sailing.replace("weeks", "wk"))
grid = np.array(grid)
ax.imshow(grid, cmap="Blues", vmin=-0.2, vmax=0.8, aspect="auto")
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, "%.2f" % grid[i, j], ha="center", va="center",
                fontsize=6.2, color="white" if grid[i, j] > 0.45 else NAVY)
    chosen = int(routes.iloc[i]["fitted_lag"])
    ax.add_patch(plt.Rectangle((chosen - 0.5, i - 0.5), 1, 1, fill=False,
                               ec=NAVY, lw=1.3))
ax.set_xticks(range(4))
ax.set_xticklabels([str(k) for k in range(4)])
ax.set_xlabel("lag (months)")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=6.3)
ax.tick_params(length=0)
for edge in ax.spines.values():
    edge.set_visible(False)
fig.text(0.53, 0.955, "(b) Correlation by lag (box = chosen lag)",
         fontsize=8.5, fontweight="bold", color=NAVY)
fig.savefig(CHARTS / "report2_fig2_lag_fit.png", dpi=300)
plt.close(fig)


# Part 3 - Figure 4, same-flow comparison of exporter and importer rows
def monthly(name, **keep):
    """Tonnes per month of one clean file, keeping rows that match."""
    data = pd.read_csv(BY_SOURCE / (name + ".csv"), parse_dates=["month"])
    for column in keep:
        data = data[data[column] == keep[column]]
    return data.groupby("month")["tonnes"].sum()


brazil_all = monthly("brazil_exporter")
brazil_to_china = monthly("brazil_exporter", counterparty="CHN")
zaf_to_china = monthly("south_africa_exporter", counterparty="CHN")
china_brazil = monthly("china_importer", exporter="BRA")
china_zaf = monthly("china_importer", exporter="ZAF")
eu_brazil = monthly("eu_importer", exporter="BRA")

years = list(range(2016, 2025))


def yearly(series):
    """Tonnes added up by year, 2016 to 2024."""
    series = series[(series.index.year >= 2016) & (series.index.year <= 2024)]
    return series.groupby(series.index.year).sum().reindex(years)


# Ratio of the importer record to the exporter record, by year
aggregate = (yearly(china_brazil) + yearly(eu_brazil)) / yearly(brazil_all)
same_brazil = yearly(china_brazil) / yearly(brazil_to_china)
same_zaf = yearly(china_zaf) / yearly(zaf_to_china)
print(pd.DataFrame({"BRA aggregate": aggregate, "BRA to China": same_brazil,
                    "ZAF to China": same_zaf}).round(3).to_string())

# Correlation of monthly tonnes with China's months moved back 0 to 4
span = pd.date_range("2016-01-01", "2024-12-01", freq="MS")
fit_months = (span >= "2019-01-01") & (span <= "2024-08-01")
lag_corr = {}
for label, exporter_series, china_series in [
        ("Brazil to China", brazil_to_china, china_brazil),
        ("South Africa to China", zaf_to_china, china_zaf)]:
    x = exporter_series.reindex(span)
    y = china_series.reindex(span)
    lag_corr[label] = []
    for k in range(5):
        both = pd.concat([x, y.shift(-k)], axis=1)[fit_months].dropna()
        lag_corr[label].append(both.corr().iloc[0, 1])
    print(label, np.round(lag_corr[label], 2))

fig = plt.figure(figsize=(6.5, 2.2))
ax = fig.add_axes([0.07, 0.17, 0.52, 0.65])
ax.axhline(1, color="#B0B7BE", lw=0.6)
ax.plot(years, same_brazil, color=IMPORTER_INK, lw=1.6, marker="o", ms=3)
ax.plot(years, same_zaf, color=IMPORTER_INK, lw=1.2, ls="--", marker="o",
        ms=2.5, mfc="white")
ax.plot(years, aggregate, color=GREY, lw=1.2, marker="o", ms=2.5)
ax.text(2024.1, same_brazil.iloc[-1], "Brazil to China,\nsame flow",
        fontsize=6.5, color=IMPORTER_INK, va="center")
ax.text(2024.1, same_zaf.iloc[-1], "South Africa\nto China", fontsize=6.5,
        color=IMPORTER_INK, va="center")
ax.text(2024.1, aggregate.iloc[-1], "Brazil, all\nimporters / all\nexports",
        fontsize=6.5, color=GREY, va="center")
ax.set_ylim(0.5, 1.45)
ax.set_xlim(2015.6, 2026.2)
ax.set_xticks(list(range(2016, 2025, 2)))
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("importer record / exporter record")
fig.text(0.0, 0.93, "(a) Yearly ratio of the records, 2016 to 2024",
         fontsize=8, fontweight="bold", color=NAVY)

ax = fig.add_axes([0.71, 0.20, 0.27, 0.62])
ax.plot(range(5), lag_corr["Brazil to China"], color=IMPORTER_INK, lw=1.6,
        marker="o", ms=3)
ax.plot(range(5), lag_corr["South Africa to China"], color=IMPORTER_INK,
        lw=1.2, ls="--", marker="o", ms=2.5, mfc="white")
ax.text(2.1, 0.88, "Brazil", fontsize=6.5, color=IMPORTER_INK)
ax.text(2.1, 0.55, "South Africa", fontsize=6.5, color=IMPORTER_INK)
ax.set_xticks(range(5))
ax.set_xlabel("lag (months)", labelpad=1)
ax.set_ylim(0, 1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#EEF0F2", lw=0.5)
fig.text(0.65, 0.93, "(b) Correlation of monthly tonnes by lag", fontsize=8,
         fontweight="bold", color=NAVY)
fig.savefig(CHARTS / "report2_fig4_same_flow.png", dpi=300)
plt.close(fig)


# Part 4 - Figure 5, what each record sees and when, Brazil
brazil_cover = coverage[coverage["country"] == "BRA"]
ship_low = brazil_cover["ship_cover_pct"].min()
ship_high = brazil_cover["ship_cover_pct"].max()
importer_low = brazil_cover["importer_cover_aligned_pct"].min()
importer_high = brazil_cover["importer_cover_aligned_pct"].max()
print("Brazil ship share %.1f to %.1f%%, importer share after lag %.1f to "
      "%.1f%%, lags %d and %d" % (ship_low, ship_high, importer_low,
                                  importer_high, lag_china, lag_eu))

fig = plt.figure(figsize=(6.5, 2.55))
ax = fig.add_axes([0.0, 0.0, 0.56, 1.0])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.01, 0.955, "(a) What each record sees, and when", fontsize=8.5,
        fontweight="bold", color=NAVY)

# A timeline in weeks after the ship is loaded, 0 to 14 weeks
line_start, line_end, n_weeks = 0.37, 0.97, 14


def week_x(week):
    """Where a number of weeks after loading sits on the timeline."""
    return line_start + (line_end - line_start) * week / n_weeks


y_exporter, y_ship, y_importer = 0.70, 0.45, 0.20
labels = [("Exporter", "Brazil customs", "all destinations, all ports",
           EXPORTER, y_exporter),
          ("Ships", "IMF PortWatch, 6 ports", "dry bulk at named ports",
           SHIP, y_ship),
          ("Importers", "China + EU27 customs", "2 of about 60 destinations",
           IMPORTER, y_importer)]
for name, source, scope, colour, y in labels:
    box(ax, 0.01, y - 0.085, 0.33, 0.17, LIGHT)
    ax.add_patch(plt.Rectangle((0.01, y - 0.085), 0.012, 0.17, fc=colour))
    ax.text(0.035, y + 0.045, name, fontsize=8.5, fontweight="bold",
            color=NAVY, va="center")
    ax.text(0.035, y - 0.005, source, fontsize=6.8, color=GREY, va="center")
    ax.text(0.035, y - 0.050, scope, fontsize=6.8, color=GREY, va="center",
            style="italic")
    ax.plot([line_start, line_end], [y, y], color="#D5DADF", lw=0.8,
            zorder=1)

# Exporter and ships both record at departure
ax.scatter([week_x(0)], [y_exporter], s=38, color=EXPORTER, ec=NAVY, lw=0.5,
           zorder=3)
ax.text(week_x(0) + 0.015, y_exporter + 0.055, "declared at departure: 100%",
        fontsize=6.8, color=EXPORTER_INK, va="center")
ax.scatter([week_x(0)], [y_ship], s=38, color=SHIP, ec=NAVY, lw=0.5,
           zorder=3)
ax.text(week_x(0) + 0.015, y_ship + 0.055,
        "draught change at loading: %.0f-%.0f%%" % (ship_low, ship_high),
        fontsize=6.8, color=SHIP_INK, va="center")

# Importers: the sailing time as a bar, the fitted lag as a marker
ax.add_patch(plt.Rectangle((week_x(5), y_importer + 0.005),
                           week_x(7) - week_x(5), 0.035, fc=IMPORTER,
                           ec="none", zorder=2))
ax.add_patch(plt.Rectangle((week_x(2.5), y_importer - 0.040),
                           week_x(3.5) - week_x(2.5), 0.035, fc=IMPORTER,
                           ec="none", alpha=0.55, zorder=2))
ax.scatter([week_x(lag_china * 4.35)], [y_importer + 0.022], marker="v",
           s=30, color=NAVY, zorder=4)
ax.scatter([week_x(lag_eu * 4.35)], [y_importer - 0.022], marker="v", s=30,
           color=NAVY, zorder=4)
ax.text(week_x(6), y_importer + 0.075, "China: arrives 5-7 wk", fontsize=6.4,
        color=IMPORTER_INK, va="center", ha="center")
ax.text(week_x(3), y_importer - 0.085, "EU: 3 wk", fontsize=6.4,
        color=IMPORTER_INK, va="center", ha="center")
ax.text(week_x(lag_china * 4.35) + 0.015, y_importer + 0.022,
        "lag %d mo" % lag_china, fontsize=6.4, color=NAVY, va="center")
ax.text(week_x(lag_eu * 4.35) - 0.018, y_importer - 0.022,
        "lag %d mo" % lag_eu, fontsize=6.4, color=NAVY, va="center",
        ha="right")
ax.text(line_end, y_importer + 0.16, "after lag: %.0f-%.0f%% of exporter"
        % (importer_low, importer_high), fontsize=6.8, color=IMPORTER_INK,
        ha="right", va="center")
for week in [0, 4, 8, 12]:
    ax.plot([week_x(week), week_x(week)], [0.075, 0.085], color=GREY, lw=0.6)
    ax.text(week_x(week), 0.045, str(week), fontsize=6.4, color=GREY,
            ha="center")
ax.text((line_start + line_end) / 2, 0.0, "weeks after loading (▼ = fitted "
        "lag)", fontsize=6.4, color=GREY, ha="center")

# Right: the three monthly series, importers after the lag
ax = fig.add_axes([0.655, 0.12, 0.335, 0.70])
brazil_table = table[table["country"] == "BRA"]
ax.plot(brazil_table["month"], brazil_table["exporter_t"] / 1e6,
        color=EXPORTER, lw=1.4)
ax.plot(brazil_table["month"], brazil_table["ship_t"] / 1e6, color=SHIP,
        lw=1.4)
ax.plot(brazil_table["month"], brazil_table["importer_aligned_total_t"] / 1e6,
        color=IMPORTER, lw=1.4)
for x, word, ink in [(0.0, "exporter", EXPORTER_INK),
                     (0.24, "ships", SHIP_INK),
                     (0.40, "importers (after lag)", IMPORTER_INK)]:
    ax.text(x, 1.0, word, transform=ax.transAxes, color=ink, fontsize=6.8,
            va="bottom", fontweight="bold")
ax.set_ylim(0, 42)
ax.set_ylabel("million tonnes per month")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#E6E9EC", lw=0.5)
fig.text(0.60, 0.955, "(b) Monthly tonnes, Brazil, 2019 to 2024",
         fontsize=8.5, fontweight="bold", color=NAVY)
fig.savefig(CHARTS / "report2_fig5_three_records.png", dpi=300)
plt.close(fig)
