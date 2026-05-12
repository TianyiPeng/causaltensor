Usage
=====

.. _installation:

Installation
------------

CausalTensor is compatible with Python 3.8 or later and depends on NumPy,
SciPy, and CVXPY. Install from PyPI with:

.. code-block:: console

   $ pip install causaltensor

To upgrade to the latest release:

.. code-block:: console

   $ pip install -U causaltensor

Quick-Start
-----------

Every estimator accepts an outcome matrix ``O`` (N x T) and a binary treatment
mask ``Z`` (N x T), constructs a solver, and calls ``fit()``:

.. code-block:: python

   import numpy as np
   from causaltensor.cauest.DID import DIDPanelSolver

   O = np.random.randn(20, 30)          # 20 units, 30 time periods
   Z = np.zeros((20, 30))
   Z[0, 15:] = 1                        # unit 0 treated from period 15 onward

   result = DIDPanelSolver(O, Z).fit()
   print(result.tau)                    # ATT estimate
   print(result.baseline.shape)         # (20, 30) counterfactual panel

See :doc:`api` for the full list of estimators and their parameters.

Tutorial Notebooks
------------------

Three Jupyter notebooks walk through the main use cases in depth.  They are
available in the ``tutorials/guides/`` folder of the repository.

Track 1 -- Real Observed Panels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook:** ``tutorials/guides/01_real_observed_panels.ipynb``

Applies all seven estimators to built-in real-world datasets (California
tobacco, Basque terrorism, German reunification).  Covers:

* Loading data with ``PanelDataset``.
* Instantiating and fitting each ``*PanelSolver`` class.
* Plotting actual vs. counterfactual outcome trajectories with Plotly.

Track 2 -- Synthetic DGP
~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook:** ``tutorials/guides/02_synthetic_dgp.ipynb``

Explores estimation accuracy under full experimental control using
``causaltensor.synthetic.generate``.  Covers:

* Convergence as N and T grow.
* Sensitivity to rank misspecification.
* Sensitivity to noise (signal-to-noise ratio).

Track 3 -- Semi-Synthetic Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Notebook:** ``tutorials/guides/03_semi_synthetic_benchmarks.ipynb``

Uses the real Basque dataset as a baseline and injects synthetic treatment
effects to benchmark all seven methods across four treatment patterns (IID,
Block, Staggered, Adaptive).  Covers:

* Running ``run_experiment`` for each method x pattern combination.
* Summarising relative error with heatmaps and box plots.
* A/A tests to check false-positive rates.

Seven Estimators at a Glance
-----------------------------

+----------------+----------------------------------+--------------+
| Short name     | Class                            | Pattern      |
+================+==================================+==============+
| DID            | ``DIDPanelSolver``               | All          |
+----------------+----------------------------------+--------------+
| SDID           | ``SDIDPanelSolver``              | Block        |
+----------------+----------------------------------+--------------+
| DC-PR          | ``DCPanelSolver``                | All          |
+----------------+----------------------------------+--------------+
| MC-NNM         | ``MCNNMPanelSolver``             | Block        |
+----------------+----------------------------------+--------------+
| CovPCA         | ``CovariancePCAPanelSolver``     | IID/Adaptive |
+----------------+----------------------------------+--------------+
| SC (OLS)       | ``OLSSCPanelSolver``             | Block        |
+----------------+----------------------------------+--------------+
| RSC            | ``RSCPanelSolver``               | Block        |
+----------------+----------------------------------+--------------+
