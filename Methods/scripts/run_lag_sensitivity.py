"""Lag sensitivity on the real table (exploratory illustration).

    python scripts/run_lag_sensitivity.py path/to/side_by_side.csv

For every country, rebuilds the importer record for lag_china in 0..5 and lag_eu in 0..5
and runs triple collocation on each.  Writes results/lag_sensitivity.csv and
results/lag_sensitivity_report.txt.  Brazil's "EU lag set two ways" (the open question
in the Week 5 plan) is printed separately: the fitted 3 months vs about 1 month (a
voyage of about 3 weeks), with China held at the table's lag.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from table_io import load_table
from lag_sensitivity import grid, stable_verdict

if len(sys.argv) != 2:
    sys.exit(__doc__)
df = load_table(sys.argv[1])
root = os.path.join(os.path.dirname(__file__), "..")
out = os.path.join(root, "results"); os.makedirs(out, exist_ok=True)

allrows, L = [], ["LAG SENSITIVITY on the real table (exploratory; not a study result)", "=" * 78,
    "importer = china_raw shifted by lag_china + eu_raw shifted by lag_eu (table rule, rebuilt).",
    "STABLE = importer noise/signal moves <= 25% when both lags move 1 month either way.", ""]
for c in sorted(df.country.unique()):
    d = df[df.country == c]
    bc, be = int(d.lag_china.iloc[0]), int(d.lag_eu.iloc[0])
    g = grid(df, c, range(0, 6), range(0, 6))
    allrows.append(g)
    verdict, worst = stable_verdict(g, bc, be)
    L.append(f"{c}: table lags china={bc}, eu={be}   -> importer estimate {verdict}"
             + (f" (largest move {worst:.0%})" if np.isfinite(worst) else ""))
    L.append("  joint shift of both lags (clipped at 0):")
    L.append(f"  {'china':>5}{'eu':>4}{'rows':>6}{'status':>20}{'exporter':>10}{'importer':>10}{'ships':>8}"
             f"{'imp range':>16}{'corr imp-exp':>14}")
    for d_ in (-2, -1, 0, 1, 2):
        lc, le = max(bc + d_, 0), max(be + d_, 0)
        r = g[(g.lag_china == lc) & (g.lag_eu == le)].iloc[0]
        tag = " <- table" if d_ == 0 else ""
        f = lambda k: f"{r[k]:.2f}" if k in r and np.isfinite(r[k]) else "  nan"
        rng_ = f"{r.importer_lo:.2f}-{r.importer_hi:.2f}" if "importer_lo" in r and np.isfinite(r.importer_hi) else ""
        L.append(f"  {lc:>5}{le:>4}{int(r.n_rows):>6}{str(r.status)[:19]:>20}{f('exporter_rel'):>10}"
                 f"{f('importer_rel'):>10}{f('ships_rel'):>8}{rng_:>16}{f('corr_imp_exp'):>14}{tag}")
    L.append("")
pd.concat(allrows).to_csv(os.path.join(out, "lag_sensitivity.csv"), index=False)

# Brazil: EU lag two ways (China held at the table's lag)
g = pd.concat(allrows); b = g[g.country == "BRA"]; bc = int(df[df.country == "BRA"].lag_china.iloc[0])
L += ["BRAZIL, EU lag set two ways (China lag held at %d):" % bc,
      f"  {'eu lag':>6}{'rows':>6}{'status':>20}{'exporter':>10}{'importer':>10}{'ships':>8}{'imp range':>16}{'corr imp-exp':>14}"]
for le in (0, 1, 2, 3, 4):
    r = b[(b.lag_china == bc) & (b.lag_eu == le)].iloc[0]
    f = lambda k: f"{r[k]:.2f}" if np.isfinite(r[k]) else "  nan"
    tag = "  <- table (fitted)" if le == 3 else ("  <- about a 3-week voyage" if le == 1 else "")
    L.append(f"  {le:>6}{int(r.n_rows):>6}{str(r.status)[:19]:>20}{f('exporter_rel'):>10}{f('importer_rel'):>10}"
             f"{f('ships_rel'):>8}{r.importer_lo:>9.2f}-{r.importer_hi:<6.2f}{f('corr_imp_exp'):>14}{tag}")
rep = "\n".join(L); print(rep)
open(os.path.join(out, "lag_sensitivity_report.txt"), "w").write(rep + "\n")
print("\nSaved results/lag_sensitivity.csv and results/lag_sensitivity_report.txt")
