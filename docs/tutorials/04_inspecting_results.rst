Inspecting Result Objects
=========================

**Notebook:** ``tutorials/guides/04_inspecting_results.ipynb``

Every ``solver.fit()`` call returns a result object that contains the ATT
estimate, the fitted counterfactual panel, fit diagnostics, and
estimator-specific internals (weights, fixed effects, factor matrices, etc.).
This tutorial gives a complete tour of the result API using a synthetic panel
with a known ground-truth ATT so every diagnostic can be interpreted directly.

Topics covered
--------------

1. **``result.summary()``** — formatted diagnostics table covering panel info,
   ATT, Untreated R², Control RMSE, Pre-exposure RMSE, RMSPE ratio, and
   estimator-specific model internals.
2. **``result.plot_actual_vs_counterfactual(unit)``** — interactive Plotly chart
   showing actual vs. counterfactual outcome for a single unit, with green
   treatment-period shading and a unit-level annotation box.
3. **Common attributes** — ``tau``, ``baseline``, ``O``, ``Z``, ``residuals``,
   ``effect_matrix``, ``z_pattern``, and all diagnostic properties.
4. **Estimator-specific attributes** accessed directly on the result object:

   * **DID / SDID / MC-NNM** — ``row_fixed_effects``, ``column_fixed_effects``
   * **SDID** — ``unit_weights``, ``time_weights``
   * **MC-NNM** — ``M`` (low-rank component), ``beta``
   * **DC-PR** — ``std`` / ``std_tau`` (sandwich SE); 95 % CI construction
   * **OLS SC** — ``beta`` (per-unit donor weight vectors), ``individual_te``,
     ``control_units``, ``treatment_units``
   * **CovPCA** — ``U`` (left factor matrix, N × r)
   * **RSC** — ``baseline`` / ``M``

All seven estimators are demonstrated on the same synthetic Block-treatment
panel so summaries and plots can be compared side by side.

See also
--------

* :doc:`api` for the complete attribute reference table.
* :doc:`01_real_observed_panels` for applying estimators to real data.
* :doc:`02_synthetic_dgp` for accuracy benchmarks across DGP configurations.
