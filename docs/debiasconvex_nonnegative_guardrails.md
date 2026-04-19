# DebiasConvex Non-Negative Guardrails

This document describes local branch changes for experimental non-negative
support in `DebiasConvex`. These notes are branch-local and are not package
release notes.

## Local Branch Changelog

- Marked `method_non_neg` in `DebiasConvex` as an experimental
  point-estimation heuristic.
- Limited the supported upstream-facing non-negative method to
  `method_non_neg="svd"`.
- Rejected `method_non_neg="nnmf"` with a clear top-level `ValueError` because
  it changes the model family and is not currently compatible with the
  debiasing and standard-error formulas.
- Emitted a `RuntimeWarning` whenever `method_non_neg` is used.
- Returned `std=None`, `inference_valid=False`, and `non_negative_method="svd"`
  for non-negative fits.
- Added diagnostics for baseline negativity, rank, stable rank, projected
  treatment-design conditioning, and residual norm.
- Kept raw DebiasConvex mode backward compatible: `std` is still returned and
  `inference_valid=True`.
- Added tests for invalid methods, `nnmf` policy, raw-mode inference outputs,
  non-negative baseline invariants, wrapper behavior, and diagnostics.

## Rationale

The non-negative option should not be presented as inference-valid
DebiasConvex estimation. The standard-error calculation depends on the
low-rank geometry used by the debiasing step, and non-negative projections
modify that geometry. The implementation therefore treats the option as a
transparent point-estimation heuristic:

```python
from causaltensor.cauest.DebiasConvex import DCPanelSolver

solver = DCPanelSolver(Z=Z, O=O)
res = solver.fit(suggest_r=2, method="non-convex", method_non_neg="svd")

print(res.tau)
print(res.std)                 # None
print(res.inference_valid)     # False
print(res.non_negative_method) # "svd"
print(res.diagnostics)
```

The `nnmf` path remains research-only for now. It is intentionally rejected in
this branch because it is a different non-negative factorization family and was
not consistently compatible with the convex debiasing continuation path.

## Issue Context

This follows the non-negativity discussion from
https://github.com/TianyiPeng/causaltensor/issues/12. In particular, hard
clipping inside the iterative DebiasConvex update should not be treated as a
safe default: preliminary experiments ran but produced extremely large
standard errors, which is why this implementation disables standard-error
outputs when the non-negative heuristic is requested.
