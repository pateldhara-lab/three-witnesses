"""
Build step: clean files -> Processed/side_by_side.csv and companion tables.

Reads only Processed/by_source and the lookup tables, never touches Data, and
rebuilds everything from scratch. Run from the repo folder:
    python Pipeline/build_side_by_side.py BRA ZAF PER LBR AUS
Countries default to those five; the table keeps one block of rows per
country, in the order given.

Writes to Processed/:
  side_by_side.csv      the 13-column table, one row per country and month
  lag_fit.csv           fit quality at lags 0 to 3, per country and importer
  lag_table.csv         start values, fitted lag, inside the range or not
  coverage_by_year.csv  per country and year: how much of the exporter's
                        tonnes the importer total and the ship total see
  steadiness_test.csv   how far the yearly cover moves across 2019-2024

Rules used:
  - Window 2019-01 to 2024-12 for fitting and the steadiness test. The
    table runs from 2019-01 to the newest export month.
  - Everything is dated to the export month. Importer months move back
    by the lag: aligned[m] = raw[m + lag].
  - Lag = whole months 0 to 3, fitted separately for China and the EU. Fit
    rule: the lag with the highest correlation of year-over-year changes
    between the exporter's monthly tonnes and the aligned importer tonnes, on
    export months 2019-01 to 2024-09 (the same months for every lag). Ties go
    to the smaller lag. The same fit on year-over-year PERCENT changes is run
    as a sensitivity check.
  - A blank is "not available", never zero. A true zero inside a
    customs source's coverage stays 0.
  - importer_aligned_total_t = China aligned + EU aligned, blank unless BOTH
    parts exist.
  - ship_t sums the PortWatch ports with in_country_ship_total = yes and
    feeds_table_column = ship_t for the country.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BY_SOURCE = ROOT / "Processed" / "by_source"
OUT_DIR = ROOT / "Processed"
PORTS = ROOT / "Lookups" / "ports.csv"
LAG_START = ROOT / "Lookups" / "lag_start.csv"

EXPORTER_SOURCE = {"BRA": "brazil_exporter", "ZAF": "south_africa_exporter",
                   "PER": "peru_exporter", "LBR": "liberia_exporter",
                   "AUS": "australia_exporter"}
IMPORTERS = {"CHN": "china_importer", "EU27": "eu_importer"}  # code -> source
WINDOW = (pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-01"))
FIT_MONTHS = (pd.Timestamp("2019-01-01"), pd.Timestamp("2024-09-01"))
LAGS = range(0, 4)
MIN_COVER, MAX_RANGE = 60.0, 7.0  # percent, percentage points

COLUMNS = [
    "country", "month", "exporter_t", "ship_t", "ship_extra_ports_t",
    "china_raw_t", "eu_raw_t", "lag_china", "lag_eu", "china_aligned_t",
    "eu_aligned_t", "importer_aligned_total_t", "flag",
]


def load(name):
    """One clean file as text, with month as a date and tonnes as a number."""
    df = pd.read_csv(BY_SOURCE / f"{name}.csv", keep_default_na=False,
                     dtype=str)
    df["month"] = pd.to_datetime(df["month"])
    df["tonnes"] = pd.to_numeric(df["tonnes"], errors="coerce")
    return df


def monthly(df):
    """Tonnes per month (blank stays blank when every row is blank)."""
    return df.groupby("month")["tonnes"].sum(min_count=1)


def paired(a, b, mask):
    """Two series side by side, only the masked months, no blanks."""
    return pd.concat([a, b], axis=1, keys=["x", "y"])[mask].dropna()


countries = sys.argv[1:] or ["BRA", "ZAF", "PER", "LBR", "AUS"]
n_countries = len(countries)
ship = load("portwatch_ship")
ports = pd.read_csv(PORTS, dtype=str, keep_default_na=False)
start = pd.read_csv(LAG_START, dtype=str, keep_default_na=False)

blocks, fits, coverage_rows = [], [], []
for country in countries:
    exporter_df = load(EXPORTER_SOURCE[country])
    exporter = monthly(exporter_df)
    raw = {}
    for code, source in IMPORTERS.items():
        df = load(source)
        raw[code] = monthly(df[df["exporter"] == country])

    # The lag fit. Lags are fitted seperately for China and the EU, on a
    # gap-free index long enough for every shift (EU runs to 2026-07, the
    # exporter to 2026-09)
    span = pd.date_range(exporter.index.min(),
                         max(exporter.index.max(), raw["EU27"].index.max()),
                         freq="MS")
    x = exporter.reindex(span)
    in_fit = (span >= FIT_MONTHS[0]) & (span <= FIT_MONTHS[1])
    x_change = x - x.shift(12)
    # a zero in the base month gives a blank, not infinity
    x_pct = (x / x.shift(12) - 1).replace([np.inf, -np.inf], np.nan)

    fit_rows = []
    for importer in IMPORTERS:
        importer_raw = raw[importer].reindex(span)
        for lag in LAGS:
            y = importer_raw.shift(-lag)
            y_change = y - y.shift(12)
            y_pct = (y / y.shift(12) - 1).replace([np.inf, -np.inf], np.nan)
            levels = paired(x, y, in_fit)
            changes = paired(x_change, y_change, in_fit)
            pct = paired(x_pct, y_pct, in_fit)
            fit_rows.append({"country": country, "importer": importer,
                             "lag": lag, "n_months_levels": len(levels),
                             "corr_levels": levels["x"].corr(levels["y"]),
                             "n_months_yoy": len(changes),
                             "corr_yoy": changes["x"].corr(changes["y"]),
                             "n_months_yoy_pct": len(pct),
                             "corr_yoy_pct": pct["x"].corr(pct["y"]),
                             "approx_se": 1 / np.sqrt(len(changes)-3)})
    fit = pd.DataFrame(fit_rows)

    # idxmax keeps the first (smaller) lag on a tie
    by_pair = fit.groupby(["country", "importer"])
    best = fit.loc[by_pair["corr_yoy"].idxmax()]
    fit["chosen"] = np.where(fit.index.isin(best.index), "yes", "")
    best_pct = fit.loc[by_pair["corr_yoy_pct"].idxmax()]
    fit["best_by_pct"] = np.where(fit.index.isin(best_pct.index), "yes", "")
    fits.append(fit)
    chosen = fit[fit["chosen"] == "yes"].set_index("importer")["lag"]
    lags = {code: int(chosen[code]) for code in IMPORTERS}

    # Table rows for this country (what country_block does)
    last = max(exporter.index.max(),
               ship[ship["exporter"] == country]["month"].max())
    index = pd.date_range(WINDOW[0], last, freq="MS")
    full = pd.date_range(min(exporter.index.min(), WINDOW[0]),
                         max(last, raw["EU27"].index.max()), freq="MS")
    china_raw = raw["CHN"].reindex(full)
    eu_raw = raw["EU27"].reindex(full)
    china_aligned = china_raw.shift(-lags["CHN"])
    eu_aligned = eu_raw.shift(-lags["EU27"])
    both_there = china_aligned.notna() & eu_aligned.notna()
    total = (china_aligned + eu_aligned).where(both_there)

    block = pd.DataFrame(index=index)
    block["country"] = country
    block["exporter_t"] = exporter.reindex(index)

    # ship_t and the extra ports column come from the PortWatch ports whose
    # ports.csv row says yes. No such ports means a blank column
    for column in ["ship_t", "ship_extra_ports_t"]:
        ok = ((ports["country_iso3"] == country)
              & (ports["in_country_ship_total"] == "yes")
              & (ports["feeds_table_column"] == column))
        ids = list(ports.loc[ok, "portid"])
        block[column] = np.nan
        if ids:
            port_rows = ship[(ship["exporter"] == country)
                             & ship["port"].isin(ids)]
            months_per_port = port_rows.groupby("port")["month"].nunique()
            assert len(months_per_port) == len(ids), (
                f"{country}: a port in ports.csv has no PortWatch rows")
            assert months_per_port.nunique() == 1, (
                f"{country}: ports do not all have the same months")
            block[column] = monthly(port_rows).reindex(index)

    block["china_raw_t"] = china_raw.reindex(index)
    block["eu_raw_t"] = eu_raw.reindex(index)
    block["lag_china"] = lags["CHN"]
    block["lag_eu"] = lags["EU27"]
    block["china_aligned_t"] = china_aligned.reindex(index)
    block["eu_aligned_t"] = eu_aligned.reindex(index)
    block["importer_aligned_total_t"] = total.reindex(index)

    # Flags of the exporter rows, joined with ';' when a month has several
    block["flag"] = ""
    flag_rows = exporter_df[exporter_df["flag"] != ""][["month", "flag"]]
    if len(flag_rows) > 0:
        flags = flag_rows.groupby("month")["flag"].agg(
            lambda s: ";".join(sorted(set(s))))
        block["flag"] = flags.reindex(index).fillna("")
    block.index.name = "month"
    block = block.reset_index()[COLUMNS]
    blocks.append(block)

    # Coverage per year: the share of exporter tonnes the ships and the
    # importers see, only on months where both sides exist
    b = block.copy()
    b["year"] = b["month"].dt.year
    importer_sum = (b["china_raw_t"] + b["eu_raw_t"]).where(
        b["china_raw_t"].notna() & b["eu_raw_t"].notna())
    pairs = {"ship_cover": b["ship_t"],
             "importer_cover_raw": importer_sum,
             "importer_cover_aligned": b["importer_aligned_total_t"]}
    for year, g in b.groupby("year"):
        row = {"country": g["country"].iloc[0], "year": year,
               "months_exporter": int(g["exporter_t"].notna().sum()),
               "exporter_t": g["exporter_t"].sum(min_count=1)}
        for name, series in pairs.items():
            both = g["exporter_t"].notna() & series.loc[g.index].notna()
            row[name + "_months"] = int(both.sum())
            if both.any():
                seen = series.loc[g.index][both].sum()
                exporter_sum = g["exporter_t"][both].sum()
                row[name + "_pct"] = 100 * seen / exporter_sum
            else:
                row[name + "_pct"] = np.nan
        coverage_rows.append(row)

table = pd.concat(blocks, ignore_index=True)
coverage = pd.DataFrame(coverage_rows)
fit = pd.concat(fits, ignore_index=True)

# Steadiness: a floor of 60% and a range of at most 7 points across
# 2019-2024, for every cover series
cover_series = ["importer_cover_raw", "importer_cover_aligned", "ship_cover"]
steady_rows = []
for country, g in coverage.groupby("country"):
    w = g[(g["year"] >= WINDOW[0].year) & (g["year"] <= WINDOW[1].year)]
    for series in cover_series:
        pct = w[series + "_pct"].dropna()
        full_years = (len(pct) == WINDOW[1].year - WINDOW[0].year + 1)
        lo, hi = (pct.min(), pct.max()) if len(pct) else (np.nan, np.nan)
        floor = "yes" if full_years and lo >= MIN_COVER else "no"
        narrow = "yes" if full_years and hi - lo <= MAX_RANGE else "no"
        steady_rows.append({"country": country, "series": series,
                            "years_used": len(pct), "min_pct": lo,
                            "max_pct": hi, "range_points": hi - lo,
                            "passes_60_floor": floor,
                            "passes_7_point_range": narrow})

# The lag table - fitted lag next to the lag from the percent version, and
# whether the fit sits inside the starting range
keep = ["country", "importer", "lag", "corr_yoy", "corr_levels", "approx_se"]
lag_table = fit[fit["chosen"] == "yes"][keep]
lag_table = lag_table.rename(columns={"lag": "fitted_lag"})
pct_lag = fit[fit["best_by_pct"] == "yes"][["country", "importer", "lag"]]
pct_lag = pct_lag.rename(columns={"lag": "lag_if_pct_changes"})
lag_table = lag_table.merge(pct_lag, on=["country", "importer"])
same = lag_table["fitted_lag"] == lag_table["lag_if_pct_changes"]
lag_table["same_lag_either_way"] = np.where(same, "yes", "no")
lag_table = lag_table.merge(start, on=["country", "importer"], how="left")
low = pd.to_numeric(lag_table["start_low_months"])
high = pd.to_numeric(lag_table["start_high_months"])
inside = (lag_table["fitted_lag"] >= low) & (lag_table["fitted_lag"] <= high)
lag_table["inside_start_range"] = np.where(inside, "yes", "no")

OUT_DIR.mkdir(parents=True, exist_ok=True)
table.to_csv(OUT_DIR / "side_by_side.csv", index=False,
             date_format="%Y-%m-%d")
fit.to_csv(OUT_DIR / "lag_fit.csv", index=False)
lag_table.to_csv(OUT_DIR / "lag_table.csv", index=False)
coverage.to_csv(OUT_DIR / "coverage_by_year.csv", index=False,
                float_format="%.4f")
pd.DataFrame(steady_rows).to_csv(OUT_DIR / "steadiness_test.csv",
                                 index=False, float_format="%.4f")
print(f"wrote {len(table)} rows to {OUT_DIR / 'side_by_side.csv'} "
      f"({', '.join(countries)})")
show = ["country", "importer", "fitted_lag", "corr_yoy", "start_low_months",
        "start_high_months", "inside_start_range", "lag_if_pct_changes"]
print(lag_table[show].to_string(index=False))