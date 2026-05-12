.. :changelog:

Changelog
=========

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