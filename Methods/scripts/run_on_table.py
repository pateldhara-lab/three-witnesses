"""Run triple collocation on Ben's side-by-side table, one country at a time.

    python scripts/run_on_table.py path\\to\\side_by_side.csv
    python scripts/run_on_table.py --fake        (try it on made-up data in the same layout)

ILLUSTRATION ONLY.  Do not present these numbers as results until the simulation
study has passed the bar written in RECOVERY_BAR.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from table_io import load_table, run_by_country

if len(sys.argv) != 2:
    sys.exit(__doc__)
if sys.argv[1] == "--fake":
    from fake_data import make_fake_table
    df = load_table(make_fake_table(flags={("BRA", "2021-03-01"): "known_source_error"}))
else:
    df = load_table(sys.argv[1])

for w in df.attrs["warnings"]:
    print("WARNING:", w)
print("ILLUSTRATION ONLY - not a result until the simulation bar is met\n")
for c, (ci, res, note) in run_by_country(df).items():
    print(f"{c}: {ci.n_yoy_rows} usable year-over-year rows "
          f"({ci.n_excluded_by_flag} months excluded by flag)")
    print(res.summary() if res else f"  {note}")
    print()
