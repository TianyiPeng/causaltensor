# Legacy / unused modules

These modules are not part of the public API and are kept for reference only.

- `matcomple/` -- early matrix completion helpers (ALS solver, hard impute);
  superseded by the MC-NNM implementation in `cauest/MCNNM.py`.
- `sample_data/` -- original data-fetching utilities; superseded by
  `causaltensor.datasets` (`load_dataset`, `PanelDataset`).

NOTE: `matlib/` was intentionally NOT archived -- it is still imported by
`cauest/DebiasConvex.py`, `cauest/MCNNM.py`, and `cauest/panel_solver.py`.
