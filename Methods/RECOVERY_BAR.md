# What counts as "recovers"? Pre-specified bar for the simulation study

Author: Dhara Patel (methods lane, Stage 4A)
Date written: 10/21/2026
Committed in git (hash): ed2a2b7(added after committing)

This bar is written and committed before the many-runs study is run, and is not
changed after seeing its results.

## Disclosure: what I had seen before writing this bar
I had run the demo and one five-country fake-table run. In that run the exporter record
(planted noise 10%) returned "assumptions_broken" in 3 of 5 datasets even though the
errors were independent. The bar therefore treats a flag on the cleanest record as
expected and reports it separately. I had not seen any many-runs results.

## Setup being tested
- Sample: 72 months, which gives 60 year-over-year changes (the real three-way window).
- Planted noise/signal: exporter 10%, importer 25%, ships 40%; independent errors; scales 1.
  Defined on the year-over-year scale (see src/fake_data.py).
- Number of fake datasets: 500. At 500 runs a true 90% coverage rate is measured to
  about +/-1.3 points, enough to judge an 85-95% window.
- Bootstrap: moving blocks of 6 months, 500 resamples, 90% percentile ranges. Six months
  keeps half a year of time order, and 60 rows give 10 blocks.
- The random seed is fixed in the script so the study can be repeated exactly.

## The bar (all must hold unless stated)
1. Ships identified as noisiest: in at least 80% of runs, ships has the largest estimated
   noise. Our research question is whether ships are noisier than customs, so this is the
   ranking that matters most.
2. Full order: in at least 60% of runs the order is exporter < importer < ships. The
   exporter (10%) versus importer (25%) pair is hard at 60 months; 60% is far above the
   17% expected from guessing among the 6 possible orders. Ranking uses the raw estimated
   noise variance; a negative value counts as the lowest (cleanest) and is not discarded.
3. Coverage: the 90% ranges contain the planted value in 85-95% of runs for importer and
   ships, and in at least 80% for the exporter. Percentile ranges pile up against zero for
   a very clean record, so a slightly lower floor is set for it in advance.
4. Range width: the median 90% range for the exporter lies entirely below the median 90%
   range for ships. This is the concrete meaning of "narrow enough to tell 10% from 40%".
5. Impossible answers: every run flagged "assumptions_broken" is counted and reported by
   cause (negative noise per record, weak link, non-positive signal), never dropped. When
   errors are independent, at most 20% of runs may carry any flag. A flag on the exporter
   alone is reported as "too clean to measure at 60 months", not as broken assumptions.

## If the bar is not met (decided in advance)
Report which noise levels can be told apart at 60 months and which cannot. Treat Stage 4
as exploratory: present ranges, not point estimates, and claim only comparisons whose
ranges do not overlap. If only item 3 or 5 fails because of the exporter's boundary
effect, report that and keep the importer and ships conclusions.

## Failure sentence (workplan item 1A.5)
The idea does not work, for ship tracking as a check, if in at least three of the four
strong-statistics countries the estimated noise of ship tracking is larger than that of
both customs records, with 90% ranges that do not overlap, AND the simulation shows the
method can tell those noise levels apart at our sample size. If the simulation cannot,
the result is reported as inconclusive, not as a failure.
This extends the debrief's wording ("ship tracking's noise share is larger than the
customs records' in at least three of the four strong-statistics countries") with the
second condition, so that a "failure" cannot simply be the method being too blunt.