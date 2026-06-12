# Analysis experiments

Scripts in this folder reproduce the empirical results for the causaltensor paper: real- and semi-synthetic estimator comparisons, power analysis, DGP ablations, and synthetic load tests. All artifacts are written under `results/` (gitignored in most setups—create it by running a script).

**Estimators** (unless `--methods` restricts the set): `DCPR`, `MC_NNM_CV`, `CovPCA`, `OLS_DID`, `SDID`, `SC`, `RSC`.

## Install causaltensor

**Python 3.10+** is required.

### From PyPI

For the released package only (no local analysis scripts):

```bash
pip install causaltensor
```

Add Kaleido for PNG export when using Plotly elsewhere:

```bash
pip install "causaltensor[static-plots]"
```

### From this repo (paper reproduction)

Clone the repository, then install in editable mode with analysis dependencies. **Poetry** (recommended):

```bash
git clone https://github.com/TianyiPeng/causaltensor.git
cd causaltensor
poetry install -E static-plots
poetry shell
```

**pip** alternative from the repo root:

```bash
pip install -e ".[static-plots]"
```

The `static-plots` extra pulls in Kaleido. Real-data and semi-synthetic `--plots` need it.

### Data

Raw panels live under `datasets/raw/` at the repo root (or pass `--raw-path` on dataset scripts). Built-in names include `smoking`, `basque`, `pwt`, `dunnhumby`, `movielens`, and others—see `causaltensor.datasets.available_datasets()`.

## Run the scripts

Commands below assume you `cd` into this directory:

```bash
cd src/causaltensor/analysis
```

Equivalent module form from the repo root:

```bash
python -m causaltensor.analysis.<script_name> ...
```

---

## Paper runs

### 1. Real data

Point estimates and (optionally) counterfactual plots on datasets with an observed treatment matrix `Z`.

```bash
python real_dataset_report.py smoking --plots
python real_dataset_report.py dunnhumby --plots
```

**Outputs** (`results/real_data/<dataset>/`):

| File | Contents |
|------|----------|
| `real_data_report_<dataset>.csv` | `tau_hat` (and related fields) per method |
| `counterfactual_<method>.png` | Per-method counterfactual (with `--plots`) |
| `counterfactual_all_methods.png` | Overlay of all methods |

`movielens` and `retailrocket` are registered loaders but omitted from the default real-data workflow when no `Z` is shipped.

---

### 2. Semi-synthetic

Inject synthetic treatment into a real panel, sweep treatment levels and assignment patterns, compare relative error `|τ̂ − τ*| / |τ*|`.

```bash
python semi_synthetic.py smoking --plots
python semi_synthetic.py pwt --plots
python semi_synthetic.py movielens --plots --treatment-patterns "Adaptive,IID"
python semi_synthetic.py dunnhumby --plots --treatment-patterns "Adaptive,IID"
```

| Setting | Value |
|---------|-------|
| Baseline | `control` (CLI default) |
| Treatment levels | `0.2`, `0.1`, `0.05` |
| Trials per `(method, pattern, level)` | `100` |
| Default patterns | `Block`, `Staggered` (overridden above for movielens / dunnhumby) |

**Outputs** (`results/semi_synthetic_data/<dataset>/`):

| File | Contents |
|------|----------|
| `semi_synthetic_control_results_detailed.csv` | All Monte Carlo trials |
| `semi_synthetic_control_results_aggregated.csv` | Mean ± std per `(method, pattern, level)` |
| `semi_synthetic_control_error_boxplot.png` | Box plots by treatment level (with `--plots`) |

Rebuild a box plot from an existing detailed CSV:

```bash
python semi_synthetic.py --plot-from-csv results/semi_synthetic_data/smoking/semi_synthetic_control_results_detailed.csv
```

---

### 3. Power analysis

A/A null simulations, empirical `|τ|` thresholds, and Monte Carlo power over a grid of relative effects δ.

**PWT** — control baseline, fine δ grid, two assignment patterns:

```bash
python -m causaltensor.analysis.power_analysis pwt --baseline control --pattern Block --rel-effects 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
python -m causaltensor.analysis.power_analysis pwt --baseline control --pattern Staggered --rel-effects 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
```

**MovieLens** — Adaptive and IID patterns (default δ grid: nine points from 0 to 0.08):

```bash
python power_analysis.py movielens --pattern Adaptive --baseline control
python power_analysis.py movielens --pattern IID --baseline control
```

**Outputs** (`results/power_analysis/<dataset>/`), per `(baseline, pattern)` run:

| File | Contents |
|------|----------|
| `<pattern>_null_trials.csv` | A/A null draws |
| `<pattern>_empirical_thresholds.csv` | Critical `|τ|` at α = 0.05 |
| `<pattern>_empirical_power.csv` | Power vs δ |
| `<pattern>_null_tau_distribution.png` | Null τ distribution |
| `<pattern>_empirical_power.png` | Power curves |

With `--baseline control`, filenames use the `control_` prefix when multiple baselines share a folder (e.g. `control_Block_null_trials.csv`).

---

### 4. Synthetic DGP ablation

Sweep rank, unit heterogeneity δ, time heterogeneity η, and noise σ on a fully synthetic panel (`N=200`, `T=50`, `30` MC trials per grid point, `Block` assignment).

```bash
python synthetic_ablation.py
```

Equivalent:

```bash
python -m causaltensor.analysis.synthetic_ablation
```

**Outputs** (`results/synthetic_ablation/`):

| File | Contents |
|------|----------|
| `ablation_N200_T50_Block_trials30_trials.csv` | Per-trial relative errors |
| `ablation_N200_T50_Block_trials30_summary.csv` | Mean ± std per `(axis, value, method)` |
| `ablation_N200_T50_Block_trials30.png` | 1×4 line plot (one subplot per ablated axis) |

Useful overrides: `--pattern Staggered`, `--trials 50`, `--methods OLS_DID SDID`, `--out-dir <path>`.

---

### 5. Synthetic — load tests

Wall time, peak RSS during fitting (`rss_fit_peak_mb`), and ATT relative error on an `N × T` grid. Tight caps below are useful for a quick smoke run; drop them for the full grid.

```bash
python load_tests.py --timeout 20 --memory-mb 100 --n-reps 3
```

| Flag | Paper value | Default |
|------|-------------|---------|
| `--timeout` | `20` | none (no limit) |
| `--memory-mb` | `100` | none |
| `--n-reps` | `3` | `3` |

**Outputs** (`results/load_tests/`):

| File | Contents |
|------|----------|
| `load_test_trials.csv` | One row per `(N, T, method, rep)` |
| `load_test_cells.csv` | Aggregated per cell |
| `load_test_summary.csv` | Per-estimator rollup |
| `load_test_heatmaps.png` | Time / memory / error heatmaps |

---

## Results layout

```
results/
├── real_data/<dataset>/
├── semi_synthetic_data/<dataset>/
├── power_analysis/<dataset>/
├── synthetic_ablation/
└── load_tests/
```

## Script reference

| Script | Role |
|--------|------|
| `real_dataset_report.py` | Real `Z`: tabular estimates + counterfactuals |
| `semi_synthetic.py` | Real panel + injected τ: error distributions |
| `power_analysis.py` | Null calibration + empirical power |
| `synthetic_ablation.py` | Synthetic DGP sensitivity (rank, heterogeneity, noise) |
| `load_tests.py` | Synthetic scalability: time, memory, ATT error |

Each script supports `--help` for the full CLI.
