"""
Known-number tests for Processed/by_source/portwatch_ship.csv
"""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "Processed" / "by_source" / "portwatch_ship.csv"
PORTS = ROOT / "Lookups" / "ports.csv"
SNAPSHOT = "2026-10-26"
MONTHLY_FILE = (ROOT / "Data" / "Ship tracking" / "portwatch" /
                f"snapshot_{SNAPSHOT}" / "portwatch_monthly_selected_ports.csv")


@pytest.fixture(scope="module")
def ships():
    return pd.read_csv(CSV, keep_default_na=False,
                       dtype={"port": str, "flag": str, "note": str,
                              "counterparty": str})


@pytest.fixture(scope="module")
def ports():
    return pd.read_csv(PORTS, dtype=str).fillna("")


def tonnes(ships, port, month):
    row = ships[(ships["port"] == port) & (ships["month"] == month)]
    assert len(row) == 1
    return row["tonnes"].iloc[0]


def note(ships, port, month):
    row = ships[(ships["port"] == port) & (ships["month"] == month)]
    return row["note"].iloc[0]


def test_size(ships):
    assert len(ships) == (13 * 93), len(ships)
    assert ships["port"].nunique() == 13
    assert (ships.groupby("port").size() == 93).all()
    assert ships["month"].min() == "2019-01-01"
    assert ships["month"].max() == "2026-09-01"


def test_newest_month_left_out(ships):
    # the snapshot's data stop on 2026-10-16, so October 2026 is not a
    # complete month
    assert "2026-10-01" not in set(ships["month"])


def test_exporter_ports(ships, ports):
    is_exporter = ports["witness_side"] == "exporter port"
    is_importer = ports["witness_side"] == "importer port"
    expected = set(ports.loc[is_exporter, "portid"])
    assert set(ships["port"]) == expected
    assert "port702" not in set(ships["port"])   # Maputo, a control
    assert not set(ships["port"]) & set(ports.loc[is_importer, "portid"])


def test_snapshot_stamp(ships):
    assert (ships["retrieved"] == SNAPSHOT).all()
    assert (ships["source"] == "portwatch_ship").all()
    assert (ships["counterparty"] == "").all()


def test_fetch_script_file(ships):
    # 2 separate sums of the same daily rows have to agree _> this
    # loader's and the fetch script's monthly file
    monthly = pd.read_csv(MONTHLY_FILE)
    monthly["month"] = monthly["month"] + "-01"
    columns = monthly[["portid", "month", "export_dry_bulk_t"]]
    joined = ships.merge(columns, left_on=["port", "month"],
                         right_on=["portid", "month"], how="left")
    assert joined["export_dry_bulk_t"].notna().all()
    gap = (joined["tonnes"] - joined["export_dry_bulk_t"]).abs().max()
    assert gap < 1, gap


def test_quiet_month(ships):
    # Ponta Ubu loaded nothing in January 2019
    assert tonnes(ships, "port2368", "2019-01-01") == 0
    assert "no dry bulk port calls" in note(ships, "port2368", "2019-01-01")


def test_zero_with_calls(ships):
    assert tonnes(ships, "port2368", "2020-12-01") == 0
    text = note(ships, "port2368", "2020-12-01")
    assert "although dry bulk ships called" in text

