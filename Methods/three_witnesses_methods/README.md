# Three Witnesses: methods lane (Stage 4A)

Code for proving the measuring tool, **triple collocation**, works on made-up data
where we know the right answer, before it is pointed at real trade data.

## In plain words
Imagine weighing yourself on three bathroom scales. Your real weight changes show
up on all three. Each scale's own wobble shows up on only one. Triple collocation
uses that: whatever two records have in common is the "truth"; whatever is left in
a record is its own error. With only two scales you see a disagreement but cannot
tell which scale is off. With three you can.

In our project the three scales are the exporter's own record, the importers'
customs records, and ship tracking (IMF PortWatch).

**Why fake data first?** If the method cannot find noise we planted ourselves, we
cannot trust it on real data. And because we only have about 60 usable months, we
need to know honestly how fuzzy the answers will be.

## What is in this folder
| File | What it does |
|---|---|
| `src/triple_collocation.py` | The method. Takes 3 columns, returns each record's noise and a status ("ok" or "assumptions_broken" with reasons) |
| `src/fake_data.py` | Builds a hidden true series (seasons + drift) and three records = truth + planted noise. Can make two records share mistakes |
| `tests/` | 52 automatic checks (exact formula, recovery with lots of data, scale invariance, flags, NaN handling, correlated mistakes, and the table reader) |
| `src/table_io.py` | Reads Ben's 13-column side-by-side table, checks it, builds year-over-year changes per country and runs the method |
| `src/study.py` | The many-runs study: block bootstrap, ranges, scoring against the bar |
| `scripts/run_study.py` | Runs the study and writes `results/`. Settings at the top must match `RECOVERY_BAR.md` |
| `src/eda.py` | Exploratory analysis of Ben's table: coverage, odd months, shares, lags, seasons, error behaviour, odd one out, and an error catalog |
| `scripts/run_eda.py` | `python scripts/run_eda.py table.csv` (or `--fake`): writes tables, a report and charts to `results/eda/` |
| `scripts/demo.py` | Small sanity demo, not the evaluation |
| `scripts/run_on_table.py` | `python scripts/run_on_table.py table.csv` (or `--fake`): per-country results, labelled illustration only |
| `RECOVERY_BAR.md` | Template: what counts as "recovers". **Fill in before running the study** |
| `src/scenarios.py` | Realistic-error scenarios built from the EDA catalog: realistic noise, short window, drifting share, shared mistakes, false zeros, unit slips, gaps |
| `scripts/run_scenarios.py` | Runs the 18 scenarios (`--smoke` prints no numbers). Writes `results/scenario_report.txt` |
| `RECOVERY_BAR_scenarios.md` | Pre-specified bar for the scenario study. **Fill in and commit before running** |

## Run it in VS Code (Windows)
1. Install Python 3.10+ and open this folder in VS Code (File > Open Folder).
2. Open a terminal (Ctrl+`) and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pytest
   python scripts/demo.py
   ```
3. Choose the `.venv` interpreter when VS Code asks. The Testing panel (flask icon)
   will find the tests automatically (settings are in `.vscode/`).

## How Ben's table feeds the method
Columns (one row per country and month): `country, month, exporter_t, ship_t,
ship_extra_ports_t, china_raw_t, eu_raw_t, lag_china, lag_eu, china_aligned_t,
eu_aligned_t, importer_aligned_total_t, flag`.

The method uses three of them as the three witnesses:
**exporter = `exporter_t`, importer = `importer_aligned_total_t`, ships = `ship_t`.**
Everything else is kept for checks. The reader:
- stops with a plain message if a column is missing, a month is not the 1st, a
  country-month repeats, or a flag has an unknown value (allowed: `missing`,
  `known_source_error`, `single_source`);
- warns if column 12 is not column 10 + column 11;
- uses Jan 2019 to Dec 2024 (72 months) and a full monthly calendar, so a missing
  month can never make "a year earlier" point 13 months back;
- blanks months flagged `missing` or `known_source_error` in all three witnesses
  (`single_source` months are unconfirmed but present, so they are kept; pass
  `exclude_flags` to drop them in a sensitivity run). One bad month costs two
  year-over-year rows (that month and the same month a year later);
- never treats a blank as zero;
- refuses to run a country with fewer than 24 usable rows, and says so.
Not used by the method: `ship_extra_ports_t` (Cape Lambert check), the raw China/EU
columns and the lags.

## Key definitions (put these in the report)
- **Year-over-year change** = a month minus the same month a year earlier. It removes
  the shared seasonal pattern, which would make records look more alike than they are.
  72 months become 60 rows.
- **Noise / signal** (`rel_noise`) = size of a record's own error divided by the size
  of the real movement it sees. 0.25 means "error is a quarter the size of the real
  movement". It ignores steady scale and offset, so an importer that always sees 70%
  of the ore is not penalised.
- **Planted level** is defined on the year-over-year scale (see `fake_data.py`).
- **"assumptions_broken"** is returned for negative noise variance, near-zero links
  between records, or non-positive shared signal. It is shown, never hidden. It can
  also fire by bad luck at 60 months; the simulation measures how often.

## Things worth knowing before you build on this
- **Status "ok" does not prove the assumptions hold.** Demo step 3 shows shared
  mistakes between exporter and importer produce plausible-looking, wrong answers
  with status "ok". Only the stress test (task 6) and the fourth witness address it.
- **10% noise is hard to measure.** Its noise variance is only 1% of the signal's,
  a tiny remainder after subtracting two large numbers. Expect wide ranges for the
  best record. That is an honest finding, not a bug.
- The scenario study (`src/scenarios.py`) covers shared mistakes, drift, short windows and wild months. Not built yet: the Brazil independent check.

## Next steps
1. Commit `RECOVERY_BAR.md`, then run `python scripts/run_study.py` once (about a minute). Do not edit the bar afterwards.
2. Stress test with shared mistakes; Brazil check once Ben's table exists.
3. Agree the table columns with Ben so real data drops into `triple_collocation`
   exactly like `records_yoy` does (3 columns: exporter, importer, ships).
