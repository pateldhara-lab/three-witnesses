"""Run the realistic-error scenario study (Stage 4A, part 2).

    python scripts/run_scenarios.py            # the real run (about 20-30 minutes)
    python scripts/run_scenarios.py --smoke    # 6 runs x 3 scenarios; prints NO numbers

BEFORE RUNNING: RECOVERY_BAR_scenarios.md must be filled in and committed to git.
The settings below must match that file. Do not change them after seeing results.

Writes results/scenario_report.txt and results/scenario_summary.csv.
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from scenarios import build_scenarios, run_scenario, summarise, TOL
from fake_data import NAMES

# ---- settings: keep identical to RECOVERY_BAR_scenarios.md ----------------
N_RUNS = 500
N_BOOT = 500
BLOCK_LEN = 6
LEVEL = 0.90
SEED = 20261109
# --------------------------------------------------------------------------

smoke = "--smoke" in sys.argv
root = os.path.join(os.path.dirname(__file__), "..")
try:
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_hash = None

scenarios = build_scenarios()
if smoke:
    scenarios = [s for s in scenarios if s.name in ("A4_ships_worst", "B1_short_window", "F1_combined")]
runs, boots = (6, 40) if smoke else (N_RUNS, N_BOOT)

summaries = []
for k, sc in enumerate(build_scenarios()):
    if sc not in scenarios:
        continue
    res = run_scenario(sc, runs, SEED + k, boots, BLOCK_LEN, LEVEL)
    summaries.append(summarise(res))
    print(f"  done {sc.name}", flush=True)

if smoke:
    print("SMOKE TEST OK: the script runs end to end. No results are shown on purpose.")
    sys.exit(0)

L = ["=" * 78, "STAGE 4A, PART 2: realistic-error scenarios", "=" * 78,
     f"runs = {N_RUNS} per scenario   bootstrap = {N_BOOT} x blocks of {BLOCK_LEN}   seed = {SEED} (+k)",
     f"code version (git): {git_hash or 'not a git repo'}",
     f"bar: coverage >= {TOL['coverage_min']:.0%} (importer, ships); |bias| <= {TOL['bias_max']:.0%}; "
     f"wrong call <= {TOL['wrong_call_max']:.0%}", ""]
rows = []
for s in summaries:
    p, m, c = s["planted"], s["median_est"], s["coverage"]
    L.append(f"{s['name']}   [{'HOLDS' if s['holds'] else 'BREAKS'}]   usable rows (median) = {s['median_rows']}"
             f"   too few rows: {s['failed_share']:.0%}")
    L.append(f"  {'record':<10}{'planted':>8}{'median':>9}{'bias':>8}{'undefined':>11}{'coverage':>10}{'median 90% range':>22}")
    for i, nm in enumerate(NAMES):
        lo, hi = s["median_range"][i]
        L.append(f"  {nm:<10}{p[i]:>8.2f}{m[i]:>9.3f}{s['bias'][i]:>8.0%}{s['undefined'][i]:>11.0%}"
                 f"{c[i]:>10.1%}{lo:>13.3f} -{hi:>6.3f}")
    L.append(f"  ships truly {s['ships_truth']}: right call {s['right_call']:.1%}, wrong call {s['wrong_call']:.1%}"
             f"   | ships ranked noisiest {s['ships_top_rank']:.1%}   | flagged {s['flag_share']:.1%}")
    failed = [k for k, v in s["checks"].items() if not v]
    L.append(f"  failed checks: {', '.join(failed) if failed else 'none'}")
    L.append("")
    rows.append({"scenario": s["name"], "holds": s["holds"], "median_rows": s["median_rows"],
                 "ships_truth": s["ships_truth"], "right_call": s["right_call"], "wrong_call": s["wrong_call"],
                 "flag_share": s["flag_share"], "failed_checks": ";".join(failed),
                 **{f"{nm}_planted": p[i] for i, nm in enumerate(NAMES)},
                 **{f"{nm}_median_est": m[i] for i, nm in enumerate(NAMES)},
                 **{f"{nm}_coverage": c[i] for i, nm in enumerate(NAMES)}})
report = "\n".join(L)
print(report)
out = os.path.join(root, "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "scenario_report.txt"), "w") as f:
    f.write(report + "\n")
pd.DataFrame(rows).to_csv(os.path.join(out, "scenario_summary.csv"), index=False)
print("\nSaved results/scenario_report.txt and results/scenario_summary.csv")
