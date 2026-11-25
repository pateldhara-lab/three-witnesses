"""PortWatch ship: IMF PortWatch daily port rows -> portwatch_ship.csv

Ship-tracking record of dry-bulk tonnes leaving each exporter port, one row
per port and calendar month.

- Ports come from Lookups/ports.csv.
- Only complete calendar months count: every day of the month must be there.
- tonnes = the `export_dry_bulk` column. Dry bulk is a ship class, not a
  cargo (coal, grain and fertiliser count too). Other ship classes are left
  out on purpose.

"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
PORTWATCH = ROOT / "Data" / "Ship tracking" / "portwatch"
PORTS = ROOT / "Lookups" / "ports.csv"
COUNTRIES = ROOT / "Lookups" / "countries.csv"
OUT = ROOT / "Processed" / "by_source" / "portwatch_ship.csv"

SNAPSHOT = "2026-10-26"  # the dated folder to read, data run to 2026-10-16
FIRST_DAY = "2019-01-01"  # PortWatch history starts here
TONNES_COLUMN = "export_dry_bulk"

snapshot_dir = PORTWATCH / f"snapshot_{SNAPSHOT}"
assert snapshot_dir.is_dir(), f"snapshot folder not found: {snapshot_dir}"

# Exporter side ports only, joined by port id
ports = pd.read_csv(PORTS, dtype=str).fillna("")
ports = ports[ports["witness_side"] == "exporter port"]
known = set(pd.read_csv(COUNTRIES, dtype=str)["iso3"])
assert set(ports["country_iso3"]) <= known, "a port's country is unknown"
assert ports["portid"].is_unique, "a port id appears twice in ports.csv"
ports = ports[["portid", "country_iso3"]].reset_index(drop=True)

pieces = []
for i in range(len(ports)):
    portid, country = ports["portid"][i], ports["country_iso3"][i]

    # port id plus an underscore, so port91 never matches port910
    files = list((snapshot_dir / "daily").glob(f"{portid}_*.csv"))
    assert len(files) == 1, f"{portid}: found {len(files)} daily files"
    name = files[0].name
    daily = pd.read_csv(files[0], dtype={"portid": str, "ISO3": str})
    assert (daily["portid"] == portid).all(), f"{name}: another port's rows"
    assert (daily["ISO3"] == country).all(), f"{name}: country differs"
    daily["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d")
    assert daily["date"].is_unique, f"{name}: a date appears twice"
    first, last = daily["date"].min(), daily["date"].max()
    every_day = pd.date_range(first, last, freq="D")
    assert len(every_day) == len(daily), f"{name}: a day is missing"
    assert first == pd.Timestamp(FIRST_DAY), f"{name}: history start moved"
    values = daily[[TONNES_COLUMN, "portcalls_dry_bulk"]]
    assert (values >= 0).all().all(), f"{name}: negative values"

    # Add the days up by month and keep complete months only. Zero days are
    # real zeros, not gaps
    daily["month"] = daily["date"].dt.to_period("M").dt.to_timestamp()
    grouped = daily.groupby("month").agg(
        tonnes=(TONNES_COLUMN, "sum"),
        calls=("portcalls_dry_bulk", "sum"),
        days=("date", "size"),
    )
    grouped["days_in_month"] = grouped.index.days_in_month
    complete = grouped[grouped["days"] == grouped["days_in_month"]]
    complete = complete.reset_index()

    # A zero month with no dry bulk calls is a real quiet month, but a zero
    # with with calls would be odd, so we say so
    notes = []
    for j in range(len(complete)):
        if complete["tonnes"][j] == 0 and complete["calls"][j] == 0:
            notes.append("no dry bulk port calls this month")
        elif complete["tonnes"][j] == 0:
            notes.append("zero tonnes although dry bulk ships called")
        else:
            notes.append("")

    pieces.append(
        pd.DataFrame(
            {
                "source": "portwatch_ship",
                "exporter": country,
                "counterparty": "",
                "port": portid,
                "month": complete["month"],
                "tonnes": complete["tonnes"].astype(float),  # metric tonnes
                "flag": "",
                "retrieved": SNAPSHOT,
                "note": notes,
            }
        )
    )
out = pd.concat(pieces, ignore_index=True)
out = out.sort_values(["month", "exporter", "port"]).reset_index(drop=True)

# No month missing, and every port has every month
months = out["month"].drop_duplicates()
every_month = pd.date_range(months.min(), months.max(), freq="MS")
assert len(months) == len(every_month), "a month is missing"
assert (out.groupby("port").size() == len(months)).all(), "a port lacks months"

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
print(f"wrote {len(out)} rows to {OUT}")