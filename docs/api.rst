API Reference
=============

All estimators follow a common two-step API::

    solver = <SolverClass>(O, Z, ...)   # attach data
    result = solver.fit(...)            # run the algorithm

.. contents::
   :local:
   :depth: 1


Result Objects
--------------

Every ``fit()`` call returns a subclass of :class:`~causaltensor.cauest.result.Result`.
The table below lists **all** attributes you can read off each result object.

Common attributes (available on every result)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Attribute
     - Description
   * - ``result.tau``
     - ATT estimate — scalar, or array for multi-treatment solvers.
   * - ``result.baseline``
     - Fitted counterfactual panel (N × T ndarray).
   * - ``result.O``
     - Original observed outcome panel (N × T).
   * - ``result.Z``
     - Original treatment mask (N × T).
   * - ``result.std_tau``
     - Standard deviation of ``tau`` (if inference was run).
   * - ``result.covariance_tau``
     - Covariance matrix of ``tau`` (if available).
   * - ``result.inference_method``
     - String describing the inference approach (or ``None``).
   * - ``result.residuals``
     - ``O - baseline`` for all cells (computed property).
   * - ``result.effect_matrix``
     - Residuals restricted to treated cells ``(O - baseline) * (Z > 0)``.
   * - ``result.z_pattern``
     - Detected treatment pattern: ``'block'``, ``'staggered'``, or ``'non-monotone'``.
   * - ``result.untreated_rmse``
     - RMSE on all Z=0 cells (computed property).
   * - ``result.control_rmse``
     - RMSE on pure control units across all time periods (computed property).
   * - ``result.pre_exposure_rmse``
     - RMSE on all units before first treatment (computed property).
   * - ``result.untreated_r2``
     - R² on Z=0 cells — scale-free model-fit diagnostic.
   * - ``result.summary()``
     - Print a formatted diagnostics table.
   * - ``result.plot_actual_vs_counterfactual(unit)``
     - Interactive Plotly figure for one unit.

Estimator-specific attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------+-----------------------------+-------------------------------------------+
| Estimator         | Attribute                   | Description                               |
+===================+=============================+===========================================+
| **DID**           | ``row_fixed_effects``       | Unit FE vector (N,)                       |
|                   +-----------------------------+-------------------------------------------+
|                   | ``column_fixed_effects``    | Time FE vector (T,)                       |
|                   +-----------------------------+-------------------------------------------+
|                   | ``beta``                    | Covariate coefficients (K,) — ``None``    |
|                   |                             | when no covariates.                       |
+-------------------+-----------------------------+-------------------------------------------+
| **SDID**          | ``unit_weights``            | Donor unit weights ``w_sdid`` (N,).       |
|                   |                             | Nonzero entries = synthetic-control pool. |
|                   +-----------------------------+-------------------------------------------+
|                   | ``time_weights``            | Time weights ``l_sdid`` (T,).             |
|                   |                             | Concentrated near treatment onset.        |
+-------------------+-----------------------------+-------------------------------------------+
| **MC-NNM**        | ``M``                       | Low-rank component of counterfactual (N×T)|
|                   +-----------------------------+-------------------------------------------+
|                   | ``row_fixed_effects``       | Unit FE vector (N,)                       |
|                   +-----------------------------+-------------------------------------------+
|                   | ``column_fixed_effects``    | Time FE vector (T,)                       |
|                   +-----------------------------+-------------------------------------------+
|                   | ``beta``                    | Covariate coefficients (K,) — ``None``    |
|                   |                             | when no covariates.                       |
+-------------------+-----------------------------+-------------------------------------------+
| **DC-PR**         | ``M`` / ``baseline``        | De-biased counterfactual panel (N×T)      |
|                   +-----------------------------+-------------------------------------------+
|                   | ``std`` / ``std_tau``       | Sandwich SE for ``tau``.                  |
+-------------------+-----------------------------+-------------------------------------------+
| **OLS SC**        | ``beta``                    | List of simplex weight vectors, one per   |
|                   |                             | treated unit. ``result.beta[i]`` is a     |
|                   |                             | (n_control,) array.                       |
|                   +-----------------------------+-------------------------------------------+
|                   | ``individual_te``           | Per-unit ATT list                         |
|                   |                             | ``[unit_idx, tau_hat]`` or                |
|                   |                             | ``[unit_idx, tau_hat, p_val]``.           |
|                   +-----------------------------+-------------------------------------------+
|                   | ``control_units``           | Row indices of donor units (list[int])    |
|                   +-----------------------------+-------------------------------------------+
|                   | ``treatment_units``         | Row indices of treated units (list[int])  |
|                   +-----------------------------+-------------------------------------------+
|                   | ``V``                       | Predictor importance weights (list)       |
|                   |                             | when covariates are used.                 |
+-------------------+-----------------------------+-------------------------------------------+
| **Covariance PCA**| ``U``                       | Left factor matrix (N × r ndarray)        |
+-------------------+-----------------------------+-------------------------------------------+
| **RSC**           | ``M`` / ``baseline``        | Low-rank counterfactual panel (N×T)       |
+-------------------+-----------------------------+-------------------------------------------+

Quick access examples::

    # DID
    result = DIDPanelSolver(O, Z).fit()
    result.row_fixed_effects    # unit FE, shape (N,)
    result.column_fixed_effects # time FE, shape (T,)

    # SDID
    result = SDIDPanelSolver(O, Z).fit()
    result.unit_weights         # donor weights, shape (N,)
    result.time_weights         # time weights, shape (T,)

    # MC-NNM
    result = MCNNMPanelSolver(O, Z).fit()
    result.M                    # low-rank component only
    result.row_fixed_effects    # unit FE, shape (N,)
    result.column_fixed_effects # time FE, shape (T,)
    result.beta                 # covariate coefs (or None)

    # DC-PR
    result = DCPanelSolver(O, Z).fit()
    result.std                  # sandwich SE

    # OLS Synthetic Control
    result = OLSSCPanelSolver(O, Z).fit()
    result.beta[0]              # donor weights for first treated unit
    result.individual_te        # [[unit_idx, tau_hat], ...]
    result.control_units        # donor row indices
    result.treatment_units      # treated row indices

    # Covariance PCA
    result = CovariancePCAPanelSolver(O, Z).fit()
    result.U                    # left factor matrix (N × r)

    # RSC
    result = RSCPanelSolver(O, Z).fit()
    result.M                    # low-rank counterfactual


Difference-in-Differences (DID)
--------------------------------

.. autoclass:: causaltensor.cauest.DID.DIDPanelSolver
   :members: fit
   :show-inheritance:


Synthetic Difference-in-Differences (SDID)
-------------------------------------------

.. autoclass:: causaltensor.cauest.SDID.SDIDPanelSolver
   :members: fit
   :show-inheritance:


De-biased Convex Panel Regression (DC-PR)
------------------------------------------

.. autoclass:: causaltensor.cauest.DebiasConvex.DCPanelSolver
   :members: fit
   :show-inheritance:


Matrix Completion with Nuclear Norm Minimisation (MC-NNM)
----------------------------------------------------------

.. autoclass:: causaltensor.cauest.MCNNM.MCNNMPanelSolver
   :members: fit
   :show-inheritance:


Covariance PCA
--------------

.. autoclass:: causaltensor.cauest.CovariancePCA.CovariancePCAPanelSolver
   :members: fit
   :show-inheritance:


OLS Synthetic Control (SC)
---------------------------

.. autoclass:: causaltensor.cauest.OLSSyntheticControl.OLSSCPanelSolver
   :members: fit
   :show-inheritance:


Robust Synthetic Control (RSC)
-------------------------------

.. autoclass:: causaltensor.cauest.RobustSyntheticControl.RSCPanelSolver
   :members: fit
   :show-inheritance:


Data Utilities
--------------

.. autoclass:: causaltensor.datasets.panel_dataset.PanelDataset
   :members:
   :show-inheritance:


Synthetic Data Generation
--------------------------

Use :func:`~causaltensor.synthetic.dgp.generate` to create a fully controlled
panel from a low-rank factor model.  The function returns ``(O, Z, tau_true)``
so ground-truth evaluation is always possible.

.. autofunction:: causaltensor.synthetic.dgp.generate


Semi-Synthetic Experiments
---------------------------

Use real panel data as a baseline, inject a known synthetic treatment effect,
and benchmark estimators under controlled conditions.

.. autofunction:: causaltensor.semi_synthetic.experiment.run_experiment

.. autofunction:: causaltensor.semi_synthetic.aa_test.run_aa_test
