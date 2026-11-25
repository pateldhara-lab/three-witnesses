# Three Witnesses

**When exporters, importers and bulk carriers disagree about cargo records, who is right?**

Ben Rutabana and Dhara Patel · DS 5500 capstone · Northeastern University · Sponsor: Prof. Bemis

## Abstract

Maritime trade datasets rest on two customs records of the same exchange, which in theory match and in practice do not reconcile. A third primary source (a third witness) exists: ship-based AIS signals processed by IMF PortWatch. We test whether this record shows how far each record is from the true volume, which we never observe, using triple collocation (TC) on iron ore for five exporters (Australia, Brazil, Liberia, Peru, South Africa) from January 2019 to December 2024. We standardize eight sources into one monthly table holding three records per exporter, validate TC on simulated data and apply it to year-over-year changes. With 57 to 59 rows per country, TC does not identify per-record error: the constant-scale condition fails in at least three of five countries and the exporter estimate is undefined in three. The data instead support pairwise agreement and same-flow comparisons: China's customs record is 0.95 to 1.06 of Brazil's exports to China in each year from 2017 to 2024.

## What is in this repository

```
Data/        Source files as published (not edited): exporter, importer and ship records
Lookups/     Country, port and lag-start tables that every loader uses
Pipeline/    Eight loader scripts (one per source) and the build script
Tests/       One test script per loader, plus one for the build script
EDA/         Scripts behind every chart and number in the presentations and reports
Methods/     Simulation study and triple collocation code (src, scripts, tests, results)
Processed/   Output of Pipeline/ (clean file per source, the side-by-side table)
Charts/      Output of EDA/ (PNG and CSV)
```

The data flows one way: `Data/` → `Pipeline/` loaders → `Processed/by_source/` → `Pipeline/build_side_by_side.py` → `Processed/side_by_side.csv` (and the coverage, lag and steadiness tables) → `EDA/` and `Methods/`. `Processed/` and `Charts/` can be deleted and rebuilt from the steps below.

## Reproduce the results

Python 3.11 is the tested version.

```
pip install -r requirements.txt
```

Run every command from the repository root. Each script finds its own folders from its location, so the working directory only has to be the root.

1. **Clean the eight sources.** Each loader reads one source from `Data/` and writes one clean file to `Processed/by_source/`. The order does not matter.
   ```
   python Pipeline/loader_australia_exporter.py
   python Pipeline/loader_brazil_exporter.py
   python Pipeline/loader_china_importer.py
   python Pipeline/loader_eu_importer.py
   python Pipeline/loader_liberia_exporter.py
   python Pipeline/loader_peru_exporter.py
   python Pipeline/loader_portwatch_ship.py
   python Pipeline/loader_south_africa_exporter.py
   ```
2. **Link, check coverage and fit the lags.** This writes `side_by_side.csv`, `lag_fit.csv`, `lag_table.csv`, `coverage_by_year.csv` and `steadiness_test.csv` to `Processed/`.
   ```
   python Pipeline/build_side_by_side.py BRA ZAF PER LBR AUS
   ```
3. **Run the tests.** They pin known numbers in each source (for example Brazil's 28.93 Mt in January 2021) and check the build step.
   ```
   python -m pytest Tests
   ```
4. **Make the charts and numbers.** Each script writes to `Charts/`.
   ```
   python EDA/EDA_1.py
   python EDA/EDA_2.py
   python EDA/EDA_3.py
   python EDA/EDA_4.py
   python EDA/EDA_5.py
   python EDA/EDA_6.py
   python EDA/EDA_7.py
   ```
   `EDA_7.py` imports the methods code, so it needs `Methods/src`.

From an empty `Processed/` and `Charts/`, steps 1 to 4 take under a minute and the tests give 68 passes.

## Where each figure and table comes from

| Result | Script | Output |
|---|---|---|
| Presentation 1, slide 9 · Presentation 2, slide 4: coverage by exporter | `EDA/EDA_1.py` | `Charts/coverage_summary.csv` |
| Presentation 1, slide 10: China's record by lag; ship share by year | `EDA/EDA_5.py` | `Charts/first_look_lag_levels.*`, `first_look_ship_share.*` |
| Presentation 2, slide 5 · Report 1, Figure 1 and Report 2, Figure 6 (numbers): Brazil, January 2021 | `EDA/EDA_1.py` | `Charts/brazil_jan_2021.csv` |
| Presentation 2, slide 7: usual ratio and drift | `EDA/EDA_1.py` | `Charts/usual_ratio_level_drift.png`, `usual_ratio_table.csv` |
| Presentation 2, slides 8 to 10: worked example of signal and noise | `EDA/EDA_6.py` | `Charts/worked_example.csv` |
| Presentation 2, slides 12 and 13: noise on the real records | `EDA/EDA_2.py` | `Charts/noise_real_estimates.png`, `noise_table.csv` |
| Presentation 2, slides 14 and 15: estimating the official figure early | `EDA/EDA_3.py` | `Charts/guesses_*.png`, `guesses_table.csv` |
| Report 1, Table 3: Stage 1 tests by exporter | `Pipeline/build_side_by_side.py` | `Processed/lag_table.csv`, `coverage_by_year.csv`, `steadiness_test.csv` |
| Report 2, Figure 1: the four pipeline steps | `EDA/EDA_4.py` | `Charts/report2_fig1_pipeline_steps.png` |
| Report 2, Figure 2: how the importer lag is fitted | `EDA/EDA_4.py` | `Charts/report2_fig2_lag_fit.png` |
| Report 2, Figure 3 and Table 3: scale by year and noise on the real records | `EDA/EDA_7.py` | `Charts/report2_fig3_scale_and_noise.png`, `report2_table3_noise.csv` |
| Report 2, Figure 4: same-flow comparison | `EDA/EDA_4.py` | `Charts/report2_fig4_same_flow.png` |
| Report 2, Figure 5: the three records, Brazil | `EDA/EDA_4.py` | `Charts/report2_fig5_three_records.png` |
| Presentation 1, slide 12 · Presentation 2, slide 11 · Report 1, Figure 2 and Table 4: simulation study | `Methods/scripts/run_study.py` | `Methods/results/` |

## Data

`Data/` holds the unprocessed source files, about 13 MB in total, and nothing that fetches or edits them. `Data/README.md` lists the publisher, link, retrieval date and units of every source. The table of PortWatch ship records is a snapshot taken on 26 October 2026; the other sources were retrieved between 9 October and 9 November 2026. Later downloads will differ because publishers revise their figures.

## Methods

`Methods/` holds the triple collocation estimator, the fake-data generator, the block bootstrap, the simulation study and its tests (`cd Methods && python -m pytest`). It reads `Processed/side_by_side.csv` and nothing else from this repository.

## Notes on reproducibility

- **Floating point.** Numbers match to about 1e-15 across machines and pandas versions (for example 33135773.307 against 33135773.306999996). Rounded values in the tables and charts are identical.
- **Fonts.** Where a script sets a font it uses Aptos, then Inter, then Liberation Sans, then DejaVu Sans, whichever is installed first. Layouts shift slightly between fonts; the numbers do not.
- **Line endings.** `.gitattributes` keeps text files on LF so that Windows and macOS checkouts give the same bytes.
- **Namibia.** Its ISO code `NA` is read as missing by pandas unless `keep_default_na=False` is set; the South Africa loader and the build script set it.
- **Not included.** The scripts that downloaded the sources, and the source PDFs of the Central Bank of Liberia: `Data/Export/Liberia/` holds the monthly table that was read from them.

## Methods lane (triple collocation): `Methods/`

Estimates the error in each of the three records (exporter, importers' customs, ship tracking) with triple collocation, on year-over-year changes of `data/side_by_side.csv`.

Run from `Methods/` after `pip install -r requirements.txt`:

| What | Command | Report item |
|---|---|---|
| Tests (67) | `python -m pytest` | |
| Real-data TC with 90% block-bootstrap ranges | `python scripts/run_real_data.py` | Table 3, Figure 3 |
| Lag, Jan+Feb merge, matched-scope and seasonal checks | `python scripts/run_sensitivity.py` | 2.2 (d), 3.2 |
| Lag sensitivity | `python scripts/run_lag_sensitivity.py` | Table 3 lag column |
| Base simulation (500 datasets, pre-set bar in `RECOVERY_BAR.md`) | `python scripts/run_study.py` | Table G1, Figure 2a |
| 18 stress scenarios (bar in `RECOVERY_BAR_scenarios.md`) | `python scripts/run_scenarios.py` | Figure 2b |

Status: the Stage 2 planted-error test on real data (`src/stage2_plant.py`) is built and tested but has not been run.
The raw source files are in Data/ (see Data/README.md for sources and retrieval dates).
