API Reference
=============

All estimators follow a common two-step API::

    solver = <SolverClass>(O, Z, ...)   # attach data
    result = solver.fit(...)            # run the algorithm

Every ``fit()`` call returns a result object with at least two attributes:

* ``result.tau``      -- ATT estimate (scalar or array).
* ``result.baseline`` -- fitted counterfactual panel (N x T ndarray).

.. contents:: Estimators
   :local:
   :depth: 1


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
