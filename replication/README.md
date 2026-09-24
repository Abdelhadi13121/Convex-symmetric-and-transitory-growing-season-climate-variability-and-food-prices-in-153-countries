# Replication package

**Convex, Symmetric and Transitory: Growing-Season Climate Variability and Food Prices in 153 Countries**

This package reproduces every table in the paper from the two raw CSV files,
end to end, with no manual steps.

---

## 1. Quick start

```bash
cd replication
pip install -r requirements.txt
python run_all.py
```

A quick check that everything works (fewer randomisation draws — about 4
minutes instead of 6; see the timing table in section 4):

```bash
python run_all.py --fast
```

A single stage:

```bash
python run_all.py --stage 2
```

Everything is written to `results/` as both plain-text tables and CSVs.

## 2. Where the data must live

By default the scripts look for the two raw files **one directory above**
`replication/`:

```
claude code research paper/
├── climate_food_monthly_estimation.csv
├── climate_macro_annual_with_exposure.csv
└── replication/
    ├── run_all.py
    └── ...
```

If you keep the CSVs elsewhere, point the package at them:

```bash
set CLIMATE_DATA_DIR=D:\path\to\data        # Windows cmd
$env:CLIMATE_DATA_DIR="D:\path\to\data"     # PowerShell
export CLIMATE_DATA_DIR=/path/to/data       # bash
```

`run_all.py` checks for the files first and stops with a clear message if they
are missing, before doing any work.

## 3. Files

| File | Role |
|---|---|
| `config.py` | Paths, sample filters, lag depth, run-control switches |
| `rep_toolkit.py` | Fixed-effect absorption; Driscoll–Kraay, one-way and two-way cluster variance estimators |
| `rep_lp.py` | The local-projection estimator used by every table |
| `rep_data.py` | Stage 1 — builds the analysis panel from the raw CSVs |
| `rep_tables.py` | Stage 2 — Tables 1–6 |
| `rep_dml.py` | Stage 3 — Table 7 (double machine learning) |
| `rep_inference.py` | Stage 4 — Tables 8–13 (balance, diagnostics, stability, jackknife, randomisation inference, volatility trend) |
| `run_all.py` | Driver; checks the environment, then runs stages 1–4 |

## 4. What each stage produces

| Stage | `--fast` | full | Output |
|---|---|---|---|
| 1 Data | 5s | 5s | `results/analysis_panel.pkl` |
| 2 Tables | 59s | 59s | `results/tables.txt`, `table1..table6*.csv` |
| 3 DML | 25s | 25s | `results/dml.txt`, `table7_dml.csv` |
| 4 Inference | 143s | 287s | `results/diagnostics.txt`, `table8..table13*.csv` |
| **Total** | **3.9 min** | **6.3 min** | |

Measured on the author's machine (Windows 10, Python 3.13, laptop-class CPU).
Only stage 4 differs between modes: the randomisation test runs 200 permutations
under `--fast` and up to 1,000 in full mode, subject to the wall-clock budget in
`config.py`. The jackknife (one re-estimation per country, 153 of them) runs in
full in both modes.

## 5. Design decisions worth knowing

**Maize is excluded by construction.** The monthly file is read with a column
filter (`"maize" not in c.lower()`), so no maize variable is ever loaded into
memory. Stage 1 reports how many columns were dropped (37).

**Leads and lags run on a complete calendar grid.** Stage 1 reindexes the panel
to a full country × month grid before differencing, so a horizon-*h* lead is
always exactly *h* calendar months ahead even where the sample filter removes
interior observations. Without this, `shift(-h)` silently jumps across gaps.

**The shock is built on the full panel; the filter only selects regression rows.**
Weather has nothing to do with price-data quality, so the climate surprise is
constructed before any sample restriction. The `insample` flag decides which
rows may serve as a time-*t* observation.

**The shock is a surprise, not a level.** For each crop the growing-area- and
crop-calendar-weighted temperature *z*-score is residualised on country ×
calendar-month fixed effects **and** a country-specific linear trend. This
removes harvest seasonality and each country's own warming path. Wheat and rice
surprises are then combined with growing-area weights. `|S|` is the absolute
value — the climate-variability shock.

**Pre-trend horizons shift the control window.** At *h* < 0 the dependent
variable is an exact linear combination of the outcome lags, so with the usual
control set the regression is degenerate (R² = 1, coefficient identically zero).
`rep_lp.lp` therefore shifts the whole control window back by |*h*| at negative
horizons. This is why the panel stores lags to depth 24 (`config.P_LAG_MAX`).

**Estimation sample.** Chronic-hyperinflation economies, FAO-imputed price
observations, and country-months with |headline year-on-year inflation| ≥ 40%
are excluded. Monthly log changes are winsorised at 0.5/99.5. These choices are
in `config.py` and are easy to vary.

## 6. Honest caveats, reproduced by the package itself

The package prints these rather than hiding them:

* **Table 6, rows 8–10.** The baseline coefficient weakens when the 2007–09 food
  crisis is excluded and is insignificant when both crisis windows are dropped.
  Identification leans on episodes with large cross-country dispersion in
  growing-season anomalies. This is stated in the paper.
* **Table 5 / Table 6, row 9.** The financial-depth interaction is *not* robust
  to excluding 2020m1–2022m12. It is reported as a secondary result; nothing in
  Tables 2–4 depends on it.
* **Table 13.** Growing-season volatility is rising significantly, but the
  implied contribution to food inflation over the sample is small. The paper
  says so explicitly and does not claim to measure the cost of warming.

## 7. Platform notes

* Written for Windows, macOS and Linux. All paths are resolved relative to the
  package directory, so a project path containing non-ASCII characters (for
  example Arabic) is fine.
* All printed output is **ASCII-only**, so a Windows console running the cp1252
  code page will not raise `UnicodeEncodeError`. Files are written as UTF-8.
* No module shadows a standard-library name (every module is prefixed `rep_`),
  so running from inside the package directory is safe.
* Dependencies are numpy, pandas, scipy, scikit-learn and statsmodels only.
  No compilation, no R, no proprietary packages.
* Seeds are fixed in `config.py` (`SEED = 7`), so the DML and randomisation
  results are bit-reproducible on a given scikit-learn version. Cross-fitting
  folds are clustered by country via `GroupKFold`, which is deterministic.
