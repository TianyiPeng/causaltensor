Real Observed Panels
====================

**Notebook:** ``tutorials/guides/01_real_observed_panels.ipynb``

This tutorial applies all seven ``causaltensor`` estimators to three
classic panel datasets included in the package:

* **Smoking** -- California Proposition 99 (Abadie & Gardeazabal, 2003)
* **Basque** -- Basque terrorism economic impact (Abadie & Gardeazabal, 2003)
* **Germany** -- German reunification (Abadie, Diamond & Hainmueller, 2015)

Topics covered
--------------

1. **Loading data** with :class:`~causaltensor.matlib.data.PanelDataset`.
2. **Fitting estimators** -- instantiate each solver with ``(O, Z)`` and call
   ``fit()``.
3. **Counterfactual plots** -- interactive Plotly charts comparing actual vs.
   estimated counterfactual outcome trajectories for the treated unit.

Estimators demonstrated
-----------------------

* :class:`~causaltensor.cauest.DID.DIDPanelSolver` (DID)
* :class:`~causaltensor.cauest.SDID.SDIDPanelSolver` (SDID)
* :class:`~causaltensor.cauest.DebiasConvex.DCPanelSolver` (DC-PR)
* :class:`~causaltensor.cauest.MCNNM.MCNNMPanelSolver` (MC-NNM)
* :class:`~causaltensor.cauest.CovariancePCA.CovariancePCAPanelSolver` (CovPCA)
* :class:`~causaltensor.cauest.OLSSyntheticControl.OLSSCPanelSolver` (SC)
* :class:`~causaltensor.cauest.RobustSyntheticControl.RSCPanelSolver` (RSC)
