Semi-Synthetic Benchmarks
==========================

**Notebook:** ``tutorials/guides/03_semi_synthetic_benchmarks.ipynb``

This tutorial uses the real Basque dataset as a baseline and injects
synthetic treatment effects to benchmark all seven estimators across
four treatment patterns under controlled conditions.

Topics covered
--------------

1. **Dataset overview** -- the Basque economic panel (18 units, 43 time
   periods).
2. **Semi-synthetic experiment** -- ``run_experiment`` injects a known ATT
   and evaluates each method's relative error
   ``|tau* - tau_hat| / |tau*|``.
3. **Treatment patterns** -- IID, Block, Staggered, Adaptive.
4. **Summary visualisations** -- mean error heatmaps, box plots, and
   effect-size vs. accuracy scatter plots.
5. **A/A test** -- verifies that estimators report near-zero effects when
   no treatment is applied.

Key functions
-------------

.. code-block:: python

   from causaltensor.semi_synthetic import run_experiment, run_aa_test

   results = run_experiment(
       dataset="basque",
       method="DC-PR",
       treatment_pattern="Block",
       treatment_level=0.2,
       n_trials=10,
       seed=0,
   )
