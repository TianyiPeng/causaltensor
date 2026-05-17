# CausalTensor

CausalTensor is a Python package for causal inference and policy evaluation on panel data. Given an outcome matrix **O** (N units x T time periods) and a treatment mask **Z**, it estimates the average treatment effect on the treated (ATT) using seven modern estimators.

[PyPI Version](https://pypi.org/project/causaltensor/)
[Documentation Status](https://causaltensor.readthedocs.io/en/latest/?badge=latest)
[Downloads](https://pepy.tech/project/causaltensor)

---

## Installation

```bash
pip install causaltensor
```

Optional dependencies: `cvxpy` (required for SDID), `pyreadr` (required for some built-in datasets)

---

## Three ways to use CausalTensor


| Track                           | When to use                                      | Key API                                      |
| ------------------------------- | ------------------------------------------------ | -------------------------------------------- |
| **Real panels**                 | You have observed O and Z; want ATT estimates    | `PanelDataset`, `*PanelSolver.fit()`         |
| **Synthetic DGP**               | You want full ground-truth control over the data | `causaltensor.synthetic.generate`            |
| **Semi-synthetic benchmarking** | You want to evaluate estimators on your own data | `causaltensor.semi_synthetic.run_experiment` |


---

## Quick start

```python
from causaltensor.datasets import load_dataset
from causaltensor.cauest import DIDPanelSolver

ds = load_dataset("smoking")          # California Prop 99 panel
O, Z = ds.O, ds.Z

result = DIDPanelSolver(O, Z).fit()
print(f"ATT estimate: {result.tau:.3f}")
print(f"Counterfactual shape: {result.baseline.shape}")
```

---

## Seven estimators

All seven estimators share the same two-step API: construct with `(O, Z)`, call `.fit()`.


| Estimator                 | Class                      | `estimate()` key         | Treatment patterns | Reference                                                                |
| ------------------------- | -------------------------- | ------------------------ | ------------------ | ------------------------------------------------------------------------ |
| Difference-in-Differences | `DIDPanelSolver`           | `OLS_DID`                | Block, Staggered   | [Chamberlain 1982](http://web.mit.edu/insong/www/pdf/FEmatch-twoway.pdf) |
| Synthetic Diff-in-Diffs   | `SDIDPanelSolver`          | `SDID`                   | Block, Staggered   | [Arkhangelsky et al. 2021](https://arxiv.org/pdf/1812.09970.pdf)         |
| De-biased Convex PR       | `DCPanelSolver`            | `DCPR`                   | All                | [Farias, Li & Peng 2021](https://arxiv.org/abs/2106.02780)               |
| Matrix Completion NNM     | `MCNNMPanelSolver`         | `MC_NNM_CV`              | All                | [Athey et al. 2021](https://arxiv.org/abs/1710.10251)                    |
| Covariance PCA            | `CovariancePCAPanelSolver` | `CovPCA`                 | All                | [Xiong & Pelger 2019](https://arxiv.org/abs/1901.09056)                  |
| OLS Synthetic Control     | `OLSSCPanelSolver`         | `SC`                     | Block, Staggered   | [Abadie & Gardeazabal 2003](http://www.jstor.org/stable/3132164)         |
| Robust Synthetic Control  | `RSCPanelSolver`           | `RSC`                    | Block, Staggered   | [Amjad et al. 2018](https://arxiv.org/abs/1811.07426)                    |


---

## Tutorials

Three self-contained notebooks cover each workflow end-to-end.

### Track 1 -- Real observed panels

Apply all seven estimators to built-in datasets (California Prop 99, Basque terrorism, German reunification). Covers data loading, fitting, and interactive counterfactual plots.

[Open in GitHub](https://github.com/TianyiPeng/causaltensor/blob/main/tutorials/guides/01_real_observed_panels.ipynb)
[Open In Colab](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/guides/01_real_observed_panels.ipynb)

### Track 2 -- Synthetic DGP

Explore estimation accuracy under full experimental control: convergence as N/T grow, sensitivity to rank misspecification, and noise sensitivity.

[Open in GitHub](https://github.com/TianyiPeng/causaltensor/blob/main/tutorials/guides/02_synthetic_dgp.ipynb)
[Open In Colab](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/guides/02_synthetic_dgp.ipynb)

### Track 3 -- Semi-synthetic benchmarks

Inject synthetic treatment effects into the Basque dataset and benchmark all seven methods across four treatment patterns (IID, Block, Staggered, Adaptive).

[Open in GitHub](https://github.com/TianyiPeng/causaltensor/blob/main/tutorials/guides/03_semi_synthetic_benchmarks.ipynb)
[Open In Colab](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/guides/03_semi_synthetic_benchmarks.ipynb)

### Interpreting results

Understand what `solver.fit()` returns: how to read the ATT estimate, fitted counterfactual (`baseline`), fit diagnostics (`untreated_r2`, `control_rmse`, `pre_exposure_rmse`, `rmspe_ratio`), the `summary()` table, and `plot_actual_vs_counterfactual()`. Also covers estimator-specific internals (SDID donor/time weights, MC-NNM low-rank component, SC donor weights, DC-PR sandwich SE, CovPCA factor matrix).

[Open in GitHub](https://github.com/TianyiPeng/causaltensor/blob/main/tutorials/guides/04_inspecting_results.ipynb)
[Open In Colab](https://colab.research.google.com/github/TianyiPeng/causaltensor/blob/main/tutorials/guides/04_inspecting_results.ipynb)

---

## Built-in datasets

```python
from causaltensor.datasets import load_dataset, available_datasets

print(available_datasets())
ds = load_dataset("basque")   # returns a PanelDataset with .O, .Z, .unit_names, .time_names
```

Included panels: `smoking`, `basque`, `germany`, and others. See `PanelDataset` in the [API docs](https://causaltensor.readthedocs.io/).

---

## Documentation

Full API reference, tutorial overviews, and installation notes at **[causaltensor.readthedocs.io](https://causaltensor.readthedocs.io/)**.