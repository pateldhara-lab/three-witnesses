# Realistic-error scenarios: pre-specified bar (Stage 4A, part 2)

Author: Dhara Patel (methods lane, Stage 4A)
Date written: November 9,2026
Committed in git (hash): 711eb91 (fill in after committing; add the hash in a second commit, as for the first bar)

This file is written and committed BEFORE `scripts/run_scenarios.py` is run for real, and is not
changed after seeing results. Idoes doest follows the same discipline as `RECOVERY_BAR.md`.

## Why this exists
The first study (`RECOVERY_BAR.md`) was a baseline: independent mistakes, a steady share, a full
72 months, noise 10/25/40%. This does not say what the method does on the real errors. 
The EDA of the real table (`results/eda/`) measured them. This study plants those errors
and asks whether the method survives them. Each scenario below is tied to a row of the error catalog.

## Disclosure: what I had seen before writing this bar
- The full EDA of the real table (coverage, shares, drift, wild months, yoy correlations).
- One illustration run of triple collocation on the real table (not a result). It gave noise/signal
  of about 0.2 to 3 and "assumptions_broken" for AUS and PER, so the real records are much noisier than
  the 0.10 / 0.25 / 0.40 of the baseline. That is why the noise levels below are larger.
- After first drafting 13 scenarios I compared them with the error catalog and added G1, G2, H1, I1, J1
  (timing, Jan/Feb merge, non-ore cargo, steady shares), which the first draft missed. I had still seen no scenario results.
- I had NOT seen the results of any scenario. The scenario code was checked by unit tests and a
  smoke run that prints no numbers.

## Setup (identical for every scenario unless stated)
- 500 fake datasets per scenario, 500 bootstrap resamples, moving blocks of 6 months, 90% percentile
  ranges, as in the first study. Fixed seed: 20261109; scenario k uses seed + k so that adding a
  scenario later never changes the others.
- Planted noise/signal = the INDEPENDENT noise on the year-over-year scale (same definition as before).
  Drift, wild months and gaps are extra damage; the study reports how far the estimates move.
- Base noise for the stress scenarios: exporter 0.50, importer 0.80, ships 1.60 ("ships clearly worst,
  at a realistic scale"). The real yoy correlations (0.36-0.80) correspond to noise/signal of roughly 0.5 to 1.3.

## The 18 scenarios and the catalog row each one stands for
| ID | What is planted | Catalog row (EDA) |
|---|---|---|
| A1 | noise 0.75 / 1.55 / 1.60 (ZAF-like) | ZAF yoy correlations 0.43 / 0.43 / 0.28 |
| A2 | noise 1.25 / 1.25 / 0.20 (BRA-like), ships cleanest | BRA ships track exporter at 0.61 |
| A3 | noise 1.0 / 1.0 / 1.0 | PER/ZAF: equal-noise equivalent 1.3 |
| A4 | noise 0.5 / 0.8 / 1.6, ships clearly worst | base for B to F |
| B1 | A4 with 26 months (14 yoy rows) | Liberia: 26 months, 14 usable rows |
| C1 | ships' scale drifts -9% over the window | ZAF ships +7 pts, BRA ships -2 pts, share range 5-12 pts |
| C2 | ships' scale drifts -24% | LBR -23 pts, PER -19 pts |
| D1 | exporter and importer share 25% of mistakes | same customs papers (not yet measured) |
| D2 | exporter and importer share 50% of mistakes | same |
| E1 | one importer month falsely 0, not flagged | Peru importer June 2020 |
| E2 | one exporter month x10, not flagged | unit slip (Stage 2 will plant these too) |
| E3 | the E1 zero, caught and dropped in all records | the pipeline's flag rule |
| F1 | C1 + D1 + one dropped month + two importer gaps | the real table has all of these |
| G1 | importer recorded 1 month late | timing: lag fitted on yoy correlation is weakly identified (EDA table 4) |
| G2 | importer recorded 3 months late | EU fitted at 3 months against a voyage of about 3 weeks |
| H1 | importer: January blank, February = Jan + Feb, every year | China's merged January and February |
| I1 | ships carry a persistent non-ore component, half the ore's own variability | mixed ports (coal, grain) in ship tonnes |
| J1 | steady shares 100% / 146% / 77% | PER importer above exporter; ZAF ships at 77% |

## What is reported for every scenario
Not planted, and why: partial national coverage (South Africa's rail and Richards Bay, Australia's two ports) is
a steady or drifting share, so it is covered by J1, C1 and C2. Its size is not measured in the EDA yet; (we need to figure this out together).

Median estimate per record against planted, undefined share (negative noise variance), coverage of the
90% ranges, share of runs flagged "assumptions_broken" with the causes, median range, usable rows, and:
- **right call**: the share of runs where ships' 90% range lies entirely above both customs ranges
  (when ships truly are worst) or entirely below both (when ships truly are cleanest). This is the
  failure-sentence test, so it is the method's power.
- **wrong call**: the share of runs where the method confidently says the opposite.

## The bar: the method "holds" in a scenario if ALL of these are true
1. Coverage of the 90% range is at least 80% for the importer and for ships (the exporter is excluded
   because its range piles up against zero when it is clean, as in the first study).
2. The median estimate for the importer and for ships is within 25% of the planted value.
3. Wrong calls are at most 5% of runs.
Flags are reported by cause, not scored here (at realistic noise a flag rate above the first study's
20% limit is expected; it is a finding, not a failure).

## What each outcome means (decided in advance)
- **Holds in A and B1, breaks nowhere else:** the baseline conclusion carries over to realistic noise;
  Stage 4 results can be reported with ranges.
- **Breaks in B1:** a country with about 14 year-over-year rows (Liberia) is reported as "not enough
  data to measure", not as a ranking.
- **Breaks in C1 or C2:** a country whose measured share drifts that much gets no ranking involving
  ships unless the drift is removed first. Report which countries (ZAF, BRA, LBR, PER) are in range of that drift.
- **Breaks in D1 or D2:** the "ok" status cannot be trusted for exporter and importer; the fourth
  witness (Stage 4 extension) becomes a requirement, and any customs-versus-ships claim is stated as
  an upper/lower bound only.
- **Breaks in E1 or E2 but holds in E3:** wild months must be caught by the flag rule (Stage 2)
  before Stage 4 runs. This is the link to Stage 2: the planted errors there are these kinds.
- **A3 shows a right-call rate below 20%:** when all three records are equally noisy, no ranking is
  claimed; only "cannot be told apart".
- **Breaks in G1 or G2:** the importer lag must be fixed before Stage 4, and the lag uncertainty
  (EDA table 4) goes into the reported ranges.
- **Breaks in H1:** merged months must be repaired or blanked in the importer record before Stage 4.
- **Breaks in I1:** ships' noise is overstated wherever ports carry other cargo, so a "ships noisier"
  finding is reported as an upper bound for those countries (the PortWatch extra-ports column helps here).
- **Breaks in J1:** the method's claim of scale invariance fails in practice at these shares; report that.
- **Failure sentence:** a scenario where the method holds and the right-call rate is at least 60% is
  one where "ships not worse than customs" could be established at that data size. In any scenario
  where it is not, a real country resembling it gets "inconclusive", never "failure".
- If the method breaks in most stress scenarios, Stage 4 on real data is exploratory only: show
  ranges, and claim only comparisons whose ranges do not overlap.
