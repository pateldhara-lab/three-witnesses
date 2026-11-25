"""Sanity demo (NOT the evaluation).  Run:  python scripts/demo.py

Shows, in three small steps, that the pieces work together.  The real
evaluation (hundreds of runs, ranges, the bar for 'recovers') comes later,
after the bar is written down in RECOVERY_BAR.md.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fake_data import SimSpec, simulate, realized_noise_levels, NAMES
from triple_collocation import triple_collocation

print("=" * 70)
print("1) ONE fake dataset at the real size: 72 months -> 60 year-over-year changes")
print("   Planted noise/signal: exporter 10%, importer 25%, ships 40%")
sim = simulate(SimSpec(), seed=42)
print("   planted  :", [f"{v:.2f}" for v in sim.planted])
print("   realised :", [f"{v:.2f}" for v in realized_noise_levels(sim)], "(random draw differs a bit)")
print(triple_collocation(sim.records_yoy, NAMES).summary())
print("   -> a single run is only ONE draw; how far such estimates scatter is what the")
print("      simulation study measures.")

print("=" * 70)
print("2) Same planted levels, but with a huge sample (convergence check)")
big = simulate(SimSpec(n_months=1_000_000, noise_levels=(0.3, 0.5, 0.7)), seed=1)
print("   planted:", [0.3, 0.5, 0.7])
print(triple_collocation(big.records_yoy, NAMES).summary())

print("=" * 70)
print("3) Exporter and importer share half their mistakes (error_corr = 0.5)")
print("   All three truly have noise/signal = 0.40")
cor = simulate(SimSpec(n_months=1_000_000, noise_levels=(0.4, 0.4, 0.4), error_corr=0.5), seed=2)
print(triple_collocation(cor.records_yoy, NAMES).summary())
print("   -> exporter and importer look better than they are; ships look worse.")
print("      This is the method's biggest threat (stress test, task 6).")
