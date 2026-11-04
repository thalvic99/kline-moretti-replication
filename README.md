# Kline & Moretti (2014) — replication in Python

Replicates Tables I–IV and Figure III of Kline & Moretti (2014, QJE), "Local Economic
Development, Agglomeration Economies, and the Big Push: 100 Years of Evidence from the
Tennessee Valley Authority". Python port of `evaluation.do`, `evaluation_aut.do` and
`x_ols_JP_v11.ado` from the authors' archive
(https://eml.berkeley.edu/~pkline/papers/tva_replication.zip).

## How to run

```bash
pip install -r requirements.txt
# put build.dta from the authors' zip in data/
python replicate.py --data data/build.dta
```

Takes ~20 seconds. County boundaries for Figure III are downloaded automatically
(plotly GeoJSON, by FIPS) the first time.

## Files

| File | What it does |
|---|---|
| `src/build.py` | Variable construction (lines 1–235 of `evaluation.do`) |
| `src/estimators.py` | OLS with clustered S.E., Conley spatial HAC (200 miles), Oaxaca-Blinder (Kline 2011) |
| `src/paper_values.py` | Published numbers, used only to check the replication |
| `replicate.py` | Samples, Tables I–IV, Figure III, LaTeX output and the check |

## Output (`output/`)

- `table1.csv` … `table4.csv`: our estimates next to the published ones
- `table1.tex` … `table4.tex`: LaTeX tables (booktabs) for Overleaf
- `figure3_weights.png/.pdf`: Oaxaca weights on control counties
- `check_summary.txt`: cells that match the paper

## Results of the check

| | Matching cells |
|---|---|
| Table I | 180 / 180 |
| Table II | 108 / 110 |
| Table III | 127 / 128 |
| Table IV | 44 / 45 |

Differences (marked with a dagger in the .tex tables):
- Employment and housing units, Table II, Panel C (unadjusted): same size as the paper
  to three decimals, opposite sign. The raw difference in means is also positive (as is
  population in the same panel), so a sign error in the published table is likely.
- Farm production, Table III, Panel C (unadjusted): essentially zero (−0.0018 here,
  +0.0004 without winsorizing, +0.002 in the paper); the sign depends on winsorization.
- Table IV, farm production 1960–2000 vs. authorities: −0.0335 here vs. −0.033 in the
  paper (a small numerical difference).
- Figure III: same spatial pattern; only some legend cutoffs match, and neither the
  trimmed nor the untrimmed sample reproduces the published minimum. Cause undetermined.

Comparison is with the tables of the 2013 working-paper version. The paper's text says
584 controls are trimmed; its Table I implies 582, which is what the code produces.

## Extension: event study by decade (`event_study.py`)

```bash
python event_study.py --data data/build.dta
```

Oaxaca-Blinder effect of TVA for each census year, on the change relative to 1940
(cumulative log points), for manufacturing and agricultural employment, population
and manufacturing wages, against both comparison groups (rest of US with clustered S.E.;
proposed authorities with spatial HAC). Main version: balanced panel by outcome (same
counties in every year), because in 1900 about a quarter of counties report zero
manufacturing employment and drop out of the log. The version where the sample changes
across years is saved as `event_study_unbalanced.csv`; its 2000 coefficient divided by 6
equals Table III, Panel A, column 3. Outputs `output/event_study.png/.pdf/.csv`.

## Robustness: trimming rule (`robustness_trimming.py`)

Re-estimates Table III, Panel A (manufacturing and agricultural employment) dropping
0%, 10%, 25% (paper) and 50% of the controls with the lowest propensity score.
Output: `output/robustness_trimming.csv`.

## Stata details reproduced (needed to match exactly)

1. Perfect prediction in the logit: counties without 1920 wage data are never TVA;
   Stata drops these 83 counties and the variable.
2. Stata abbreviations in `$X`: `PRADIO` = `PRADIO30`, `nowage20` = `nowage20dum`, etc.
3. Trimming at p25 of the propensity score also drops the control exactly at the cutoff.
4. Winsorizing (1st/99th percentiles) uses the whole sample in memory, also when the
   regression is restricted to the South.
5. The counterfactual mean in the Oaxaca estimator averages X over all treated
   counties, including those with a missing outcome.
6. Conley S.E.: degrees-of-freedom correction N/(N−K) counting omitted columns; earth
   radius based on the first observation's latitude, as in the Mata code.
7. Original quirks kept: 1930 urban share divided by 1920 population; 1990 family
   income uses the 1980 level.

Not replicated: Tables V–VIII and Figure V (structural part, `structural.do`), Figure II
(appropriations data are not in the archive), Figures I and IV (map image and theory
diagrams).
