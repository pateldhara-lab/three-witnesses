"""Run the Stage 4A simulation study.

    python scripts/run_study.py

BEFORE RUNNING: RECOVERY_BAR.md must be written and committed to git.
The numbers in the BAR block below must match what you wrote in that file.
Do not change them after seeing results.

Writes results/study_report.txt and results/study_runs.csv.
"""
import os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from fake_data import SimSpec, NAMES
from study import run_study, format_report

# ---- settings: keep identical to RECOVERY_BAR.md -------------------------
N_RUNS = 500
N_BOOT = 500
BLOCK_LEN = 6
LEVEL = 0.90
SEED = 20261023
SPEC = SimSpec(n_months=72, noise_levels=(0.10, 0.25, 0.40))
BAR = {
    "ships_noisiest_min": 0.80,       # item 1
    "full_order_min": 0.60,           # item 2
    "coverage_window": (0.85, 0.95),  # item 3, importer and ships
    "coverage_exporter_min": 0.80,    # item 3, exporter
    "any_flag_max": 0.20,             # item 5
}
# --------------------------------------------------------------------------

smoke = "--smoke" in sys.argv
runs = 20 if smoke else N_RUNS

root = os.path.join(os.path.dirname(__file__), "..")
try:
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_hash = None

if smoke:
    print("SMOKE TEST (20 runs): checks the script works. These are NOT the study results.\n")

res = run_study(runs, SEED, SPEC, N_BOOT, BLOCK_LEN, LEVEL)
report = format_report(res, BAR, git_hash)
print(report)

if not smoke:
    out = os.path.join(root, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "study_report.txt"), "w") as f:
        f.write(report + "\n")
    cols = {}
    for i, nm in enumerate(NAMES):
        cols[f"{nm}_noise_var"] = res["noise_var"][:, i]
        cols[f"{nm}_rel_noise"] = res["rel"][:, i]
        cols[f"{nm}_range_lo"] = res["lo"][:, i]
        cols[f"{nm}_range_hi"] = res["hi"][:, i]
    cols["flagged"] = res["flagged"]
    cols["reasons"] = ["; ".join(r) for r in res["reasons"]]
    pd.DataFrame(cols).to_csv(os.path.join(out, "study_runs.csv"), index_label="run")
    print(f"\nSaved results/study_report.txt and results/study_runs.csv")
