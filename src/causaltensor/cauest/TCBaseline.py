"""Contemporaneous treated-minus-control baseline estimator."""

from __future__ import annotations

import numpy as np

from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


class TCBaselineResult(Result):
    """Result returned by :class:`TCBaselinePanelSolver`."""

    def __init__(self, baseline: np.ndarray, tau: float):
        super().__init__(baseline=baseline, tau=tau, return_tau_scalar=True)
        self.M = baseline


class TCBaselinePanelSolver(PanelSolver):
    """Estimate ATT from contemporaneous treated-versus-control differences.

    At each period, the counterfactual for every unit is the mean outcome among
    units that are currently untreated. The reported ATT is the mean treated
    residual, equivalently the treated-count-weighted average of period-level
    treated-minus-control mean differences.
    """

    def __init__(self, O: np.ndarray, Z: np.ndarray):
        self.O = np.asarray(O, dtype=float)
        z = np.asarray(Z, dtype=float)
        super().__init__(z)
        self.Z = z

    def fit(self) -> TCBaselineResult:
        if self.O.ndim != 2 or self.Z.ndim != 2 or self.O.shape != self.Z.shape:
            raise ValueError(
                f"O and Z must be two-dimensional with equal shapes; got "
                f"O={self.O.shape}, Z={self.Z.shape}."
            )
        if not np.all(np.isfinite(self.O)) or not np.all(np.isfinite(self.Z)):
            raise ValueError("O and Z must contain only finite values.")
        if not np.all((self.Z == 0) | (self.Z == 1)):
            raise ValueError("Z must be binary.")

        treated = self.Z == 1
        if not np.any(treated):
            raise ValueError("TC-baseline requires at least one treated cell.")

        controls = ~treated
        treated_periods = np.any(treated, axis=0)
        missing_controls = treated_periods & ~np.any(controls, axis=0)
        if np.any(missing_controls):
            periods = np.flatnonzero(missing_controls).tolist()
            raise ValueError(
                "TC-baseline requires at least one currently untreated unit in "
                f"every treated period; missing controls at periods {periods}."
            )

        control_counts = controls.sum(axis=0)
        control_means = (self.O * controls).sum(axis=0) / control_counts
        baseline = np.broadcast_to(control_means, self.O.shape).copy()
        tau = float(np.mean((self.O - baseline)[treated]))

        result = TCBaselineResult(baseline=baseline, tau=tau)
        result.O = self.O
        result.Z = self.Z
        return result


def tc_baseline(O: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the contemporaneous-control baseline and ATT."""
    result = TCBaselinePanelSolver(O, Z).fit()
    return result.baseline, result.tau
