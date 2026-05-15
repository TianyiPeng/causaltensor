.. :changelog:

Changelog
=========

0.1.14 (2026-05-15)
------------------

New features
~~~~~~~~~~~~

- **Rich result objects.** Every ``solver.fit()`` call now returns a fully self-contained result object.

  - ``result.O`` and ``result.Z`` are attached automatically, enabling all diagnostics without re-running the model.
  - New computed properties: ``residuals``, ``effect_matrix``, ``z_pattern``, ``untreated_r2``, ``control_rmse``, ``pre_exposure_rmse``, ``rmspe_ratio``.
  - **``result.summary()``** prints a formatted table covering panel info, ATT estimate (with SE when available), fit diagnostics, and estimator-specific model internals (weights, rank, fixed effects, factor matrix shape). Returns ``self`` for method chaining.
  - **``result.plot_actual_vs_counterfactual(unit)``** renders an interactive Plotly chart — actual vs. counterfactual with green treatment-period shading, T0 marker, and a unit-level annotation box.

- **Estimator-specific result attributes** (all new, accessed directly on the returned object):

  - DID / SDID / MC-NNM: ``row_fixed_effects``, ``column_fixed_effects``
  - SDID: ``unit_weights`` (donor simplex weights), ``time_weights``
  - MC-NNM: ``M`` (low-rank component separate from fixed effects), ``beta``
  - DC-PR: ``std`` / ``std_tau`` (sandwich SE), ``inference_method``
  - OLS SC: ``beta`` (per-unit donor weight vectors), ``individual_te``, ``control_units``, ``treatment_units``
  - CovPCA: ``U`` (left factor matrix, N × r)

- **Large recommendation / retail panels enabled.** ``load_dataset`` now accepts two keyword arguments forwarded to the four large-panel loaders (``retailrocket``, ``dunnhumby``, ``truus``, ``movielens``):

  - ``n_units`` (default 2500) — retain only the top-N items by event count.
  - ``time_freq`` (``'W'`` / ``'M'`` / ``'D'``, default weekly) — aggregate raw daily data before pivoting, controlling panel width.

- **Tutorial Guide 04** (``tutorials/guides/04_inspecting_results.ipynb``) — end-to-end walkthrough of the result object API using synthetic data with a known ground-truth ATT.

Performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~

- **SDID**: Replaced ``np.eye(N) @ w >= 0`` / ``np.ones(N).T @ w == 1`` constraint matrices with native CVXPY ``w >= 0`` / ``cp.sum(w) == 1`` forms; solver pinned to ``CLARABEL`` for faster, more reliable convergence.
- **OLS SC**: Inner donor-weight optimisation replaced with a ``CLARABEL``-based QP (``cvxpy``), dropping the ``sklearn`` dependency. The outer predictor-importance loop retains ``fmin_slsqp`` as it operates in a low-dimensional covariate space.
- **RSC**: Per-treated-unit projection vectorised — pseudoinverse of the pre-period donor matrix is computed once and applied to all treated rows in a single matrix multiply instead of a Python loop.

Documentation
~~~~~~~~~~~~~

- ``docs/api.rst`` extended with a "Result Objects" section: common-attribute table and per-estimator attribute reference with quick-access code examples.
- ``fit()`` docstrings for SDID, MC-NNM, and OLS SC updated with complete ``Returns`` sections.
- Cross-reference from Guide 01 to Guide 04 added.

0.1.13 (2026-05-12)
------------------

Breaking changes
~~~~~~~~~~~~~~~~

- **Solver constructor API changed.** All seven ``PanelSolver`` classes now require ``(O, Z)`` at construction time and expose a single ``fit()`` method. The old pattern of constructing a solver with no arguments and calling ``solve_with_cross_validation`` / ``solve_with_suggested_rank`` is removed. Migration::

    # Before
    solver = DCPanelSolver()
    res = solver.solve_with_suggested_rank(O, Z, suggest_r=3)

    # After
    res = DCPanelSolver(O, Z).fit(suggest_r=3)

- **``MC_NNM`` cross-validation is now an argument to ``fit()``.** Pass ``cross_validation=True`` (default) instead of calling the old ``solve_with_cross_validation`` entry-point.

- **``causaltensor.matlib.util`` removed.** All linear-algebra helpers (``SVD``, ``SVD_soft``, ``transform_to_3D``, etc.) have moved to ``causaltensor.utils.linalg``. Update any direct imports::

    # Before
    from causaltensor.matlib.util import SVD_soft

    # After
    from causaltensor.utils.linalg import SVD_soft

- **``causaltensor.matlib.generation`` and ``causaltensor.matlib.generation_treatment_pattern`` removed.** Use the replacement modules instead:

  - ``generate_low_rank_M``, ``add_noise`` → ``causaltensor.synthetic.utils``
  - ``Z_iid``, ``Z_block``, ``Z_stagger``, ``Z_adaptive`` → ``causaltensor.utils.treatment_patterns``

New features
~~~~~~~~~~~~

- Three tutorial notebooks: real observed panels, synthetic DGP study, semi-synthetic benchmarks.
- Sphinx API documentation with full NumPy-style docstrings for all solvers, ``PanelDataset``, ``generate``, ``run_experiment``, and ``run_aa_test``.
- ``causaltensor.utils.linalg`` — consolidated linear-algebra utilities with seeded ``rng=`` API.

0.1.12 (2025-03-12)
------------------
- Added CVXPY package for SDID method

0.1.11 (2025-03-12)
------------------
- Fix a bug in the DC method: suggest_r was ignored due to the priority of auto_rank and now it will be prioritized over auto_rank

0.1.10 (2025-02-08)
------------------
- Added Covariate support for SDID method

0.1.9 (2025-02-07)
------------------
- Added Panel Solver Interface
- Added more test cases
- Added covariate support for synthetic control 

0.1.8 (2023-11-05)
------------------
- Enhanced MC-NNM functionality with covariate integration and improved handling of missing data.

0.1.7 (2023-08-24)
------------------
- Introduced support for synthetic control methodology.

0.1.5 (2023-05-16)
------------------
- Expanded capabilities to address multiple-treatment problems using panel regression methods with debiasing features.