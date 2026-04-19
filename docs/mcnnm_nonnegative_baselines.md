# MCNNM Non-Negative Baseline Projection

This document describes the local branch changes for optional non-negative
post-processing in `MCNNMPanelSolver`. These notes are branch-local and are not
package release notes.

## Local Branch Changelog

- Added `MCNNMPanelSolver.fit(...)` with dispatch to suggested-rank,
  regularizer, or cross-validation solving paths.
- Added `baseline_projection="clip_nonnegative"` to `fit`,
  `solve_with_regularizer`, `solve_with_suggested_rank`, and
  `solve_with_cross_validation`.
- Preserved raw outputs as canonical: `res.baseline`, `res.tau`,
  `res.baseline_raw`, and `res.tau_raw`.
- Added projected companion outputs: `res.baseline_projected`,
  `res.tau_projected`, and `res.projection_diagnostics`.
- Added tests for regularizer, suggested-rank, cross-validation, invalid
  projection names, no-op projection, and a fixed-effects/covariate example.

## Rationale

For MC-NNM with fixed effects and covariates, the support restriction belongs to
the full baseline:

```python
baseline = fitted_value + M
```

The low-rank component `M` is a residual correction around fixed effects and
covariates, so forcing `M >= 0` is not generally the right target. The supported
projection is therefore a post-estimation correction on the final baseline:

```python
baseline_projected = np.maximum(baseline, 0)
tau_projected = np.sum((O - baseline_projected) * Z) / np.sum(Z)
```

This is not a constrained MC-NNM estimator. It is a transparent companion output
that lets users inspect how a non-negative baseline support correction changes
the implied treatment effect.

## Example

```python
from causaltensor.cauest.MCNNM import MCNNMPanelSolver

solver = MCNNMPanelSolver(Z=Z, X=X)
res = solver.fit(
    O=O,
    l=1.0,
    baseline_projection="clip_nonnegative",
)

print(res.baseline)                # raw fitted baseline
print(res.tau)                     # raw tau
print(res.baseline_projected)      # np.maximum(res.baseline, 0)
print(res.tau_projected)           # tau from projected baseline
print(res.projection_diagnostics)  # clipping and tau-shift diagnostics
```
