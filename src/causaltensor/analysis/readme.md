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

The `static-plots` extra pulls in Kaleido, for saving Plotly figures from the notebooks. `plot.py` writes the paper PNGs with Matplotlib and does not need it.

### Data

Raw panels live under `datasets/raw/` at the repo root (or pass `--raw-path` on dataset scripts). Built-in names include `smoking`, `basque`, `pwt`, `jsa_dc`, `wreb`, `dunnhumby`, `movielens`, and others—see `causaltensor.datasets.available_datasets()`.

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

Point estimates on datasets with an observed treatment matrix `Z`. Counterfactual figures are drawn by `plot.py` from the series CSV.

```bash
python real_dataset_report.py smoking
python real_dataset_report.py dunnhumby
```

**Outputs** (`results/real_data/<dataset>/`):

| File | Contents |
|------|----------|
| `real_data_report_<dataset>.csv` | `tau_hat` (and related fields) per method |
| `counterfactual_series_<dataset>.csv` | Actual path and each method's counterfactual for the plotted unit |
| `counterfactual_all_methods.png` | Overlay of all methods (`python -m causaltensor.analysis.plot`) |

`movielens` and `retailrocket` are registered loaders but omitted from the default real-data workflow when no `Z` is shipped.

**D.C. Job Search Assistance RCT** (`jsa_dc`). 1996Q1 `CONTROL` versus `SJSA` (486 each). Outcomes are quarterly earnings from 1994Q4 through 1995Q4 and 1996Q2 through 1998Q3; the entry quarter is dropped. Treatment is a Block onset in 1996Q2. `RCT` is the difference in mean post-period earnings, with a 95% interval.

**Washington reemployment bonus** (`wreb`). Final analytic sample, 1988Q2 cohort, control versus treatment 3 (high bonus, short qualification period). Units with any quarter above $100,000 are dropped (9 controls, no treated people), leaving 1,022 and 494. Outcomes are quarterly earnings from 1985Q1 through 1988Q1 and 1988Q3 through 1989Q4; the enrollment quarter is dropped. Quarters with no wage record are 0. Treatment is a Block onset in 1988Q3.

```bash
python -m causaltensor.analysis.rct
python -m causaltensor.analysis.rct --dataset wreb
```

**Outputs** (`results/rct/`):

| File | Contents |
|------|----------|
| `jsa_dc.csv` | `RCT` with its standard error and 95% interval, then one estimate per method |
| `wreb.csv` | Same columns for the Washington reemployment bonus cohort |

---

### 2. Semi-synthetic

Inject synthetic treatment into a real panel, sweep treatment levels and assignment patterns, compare relative error `|τ̂ − τ*| / |τ*|`.

```bash
python semi_synthetic.py smoking
python semi_synthetic.py pwt
python semi_synthetic.py movielens --treatment-patterns "Adaptive,IID"
python semi_synthetic.py dunnhumby --treatment-patterns "Adaptive,IID"
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
| `semi_synthetic_control_results_aggregated.csv` | Per `(method, pattern, level)`: mean relative error, signed bias, RMSE, error quantiles, successful runs |

### 3. Heterogeneous effects (CATE)

PWT only, Block assignment, one treatment level. The baseline is the same control-panel `M` as the ATT benchmark. Unit effects are a fixed linear function of standardized `hc`, `csh_i`, and `openness`, centered on the full sample so their population average is `tau*`. A trial's ATT is the average of those fixed effects over the units treated in that trial. Group cuts are the full-sample medians. Each estimator is scored from the unit effects implied by its fitted untreated surface.

```bash
python -m causaltensor.analysis.cate
```

| Setting | Value |
|---------|-------|
| Dataset | `pwt` |
| Pattern | `Block` |
| Treatment level | `0.1` |
| Trials | `100` |
| Covariates | `hc`, `csh_i`, `openness` |
| Coefficients | `1, 0.5, -0.5` on `hc`, `csh_i`, `openness`; heterogeneity scale `gamma = 0.5` |

**Outputs** (`results/cate/`):

| File | Contents |
|------|----------|
| `cate_pwt_block_delta0.1_trials100_metrics.csv` | Per trial and method: root PEHE, correlation, and the trial ATT |
| `cate_pwt_block_delta0.1_trials100_gates.csv` | Per trial, method, and covariate group: true and estimated GATE |
| `cate_pwt_block_delta0.1_trials100.png` | PEHE–correlation scatter and group-effect panel (`python -m causaltensor.analysis.plot`) |

---

### 4. Power analysis

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

`python -m causaltensor.analysis.plot` writes `results/power_analysis/composite_power_1x4.png` and `composite_null_tau_1x4.png` when the four paper CSVs are present (PWT Block, PWT Staggered, MovieLens IID, MovieLens Adaptive).

With `--baseline control`, filenames use the `control_` prefix when multiple baselines share a folder (e.g. `control_Block_null_trials.csv`).

---

### 5. Synthetic DGP ablation

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
| `ablation_N200_T50_Block_trials30.png` | 1×4 line plot (`python -m causaltensor.analysis.plot`) |

Useful overrides: `--pattern Staggered`, `--trials 50`, `--methods OLS_DID SDID`, `--out-dir <path>`.

---

### 6. Synthetic DGP robustness

One treatment level (`Δ = 0.1`) on the same `N=200`, `T=50` panel. The reference DGP is the usual Gaussian low-rank baseline, iid Gaussian noise, and additive heterogeneous effect. The other settings are AR(1) noise (`ρ = 0.5`), Student-t noise (3 degrees of freedom), a nonnegative low-rank baseline with Gamma-distributed factors and Poisson noise (`mean(M) = 1`), and a cell effect proportional to `|M|`. AR(1) and Student-t noise are scaled to the same variance as the reference, and the baseline-dependent effect is scaled so the treated-cell average still equals the reference ATT. Estimators run on IID, Block, Staggered, and Adaptive only where they are valid.

```bash
python -m causaltensor.analysis.synthetic_robustness
```

**Outputs** (`results/synthetic_robustness/`):

| File | Contents |
|------|----------|
| `robustness_N200_T50_delta0.1_trials30_trials.csv` | Per-trial relative errors |
| `robustness_N200_T50_delta0.1_trials30_summary.csv` | Mean ± std per `(dgp, pattern, method)` |
| `robustness_N200_T50_delta0.1_trials30.png` | Mean relative error ± 1 s.d. by DGP (`python -m causaltensor.analysis.plot`); panels are Block, Staggered, IID, Adaptive |

---

### 7. Synthetic — load tests

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
| `load_test_heatmaps.png` | Wide heatmaps (`python -m causaltensor.analysis.plot`): 3 metric rows × one column per estimator |

---

## Results layout

```
results/
├── real_data/<dataset>/
├── rct/
├── semi_synthetic_data/<dataset>/
├── power_analysis/<dataset>/
├── synthetic_ablation/
├── synthetic_robustness/
├── cate/
└── load_tests/
```

## Script reference

| Script | Role |
|--------|------|
| `real_dataset_report.py` | Real `Z`: tabular estimates and counterfactual series |
| `rct.py` | JSA and WREB RCTs: randomized estimates next to the seven fits |
| `semi_synthetic.py` | Real panel + injected τ: error distributions |
| `power_analysis.py` | Null calibration + empirical power |
| `synthetic_ablation.py` | Synthetic DGP sensitivity (rank, heterogeneity, noise) |
| `synthetic_robustness.py` | One-level check: AR(1), heavy tails, baseline-dependent effect |
| `cate.py` | PWT Block CATE runner; the experiment is `semi_synthetic.cate.run_cate` |
| `load_tests.py` | Synthetic scalability: time, memory, ATT error |
| `plot.py` | Figures from the CSVs in `results/` |

Analysis scripts write CSVs. `python -m causaltensor.analysis.plot` reads `results/` and writes the PNGs. Each script supports `--help` for the full CLI.
