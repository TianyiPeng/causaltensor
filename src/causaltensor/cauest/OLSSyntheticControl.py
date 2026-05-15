"""
Synthetic control via constrained least squares (Abadie-Diamond-Hainmueller style).

Weights on donor units are chosen to minimize pre-period outcome error subject to
simplex constraints; optional covariates enter a nested predictor-reweighting step.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import fmin_slsqp

from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


def _solve_simplex_qp(y_c: np.ndarray, y_t: np.ndarray) -> np.ndarray:
    """Solve ``min ||y_t - y_c @ W||²  s.t. W >= 0, sum(W) = 1``."""
    W_var = cp.Variable(y_c.shape[1])
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(y_t - y_c @ W_var)),
        [W_var >= 0, cp.sum(W_var) == 1],
    )
    prob.solve(solver=cp.CLARABEL)
    return np.clip(np.asarray(W_var.value, dtype=float), 0.0, None)


def _solve_simplex_qp_weighted(X0: np.ndarray, X1: np.ndarray,
                                V: np.ndarray) -> np.ndarray:
    """Solve ``min sum_k V_k*(X1_k - X0_k @ W)²  s.t. W >= 0, sum(W) = 1``.

    ``X0`` is (K, n_control), ``X1`` is (K,), ``V`` is (K,) predictor weights.
    """
    sqrtV = np.sqrt(np.maximum(V, 0.0))
    W_var = cp.Variable(X0.shape[1])
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(cp.multiply(sqrtV, X1 - X0 @ W_var))),
        [W_var >= 0, cp.sum(W_var) == 1],
    )
    prob.solve(solver=cp.CLARABEL)
    return np.clip(np.asarray(W_var.value, dtype=float), 0.0, None)


def _slsqp_minimizer(out):
    """Normalize ``fmin_slsqp`` return value across SciPy versions (ndarray vs tuple)."""
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=float)
    return np.asarray(out, dtype=float)


class OLSSCResult(Result):
    """Result container for :class:`OLSSCPanelSolver`."""

    def __init__(
        self,
        baseline=None,
        tau=None,
        beta=None,
        return_tau_scalar=False,
        individual_te=None,
        V=None,
        control_units=None,
        treatment_units=None,
    ):
        super().__init__(baseline=baseline, tau=tau, return_tau_scalar=return_tau_scalar)
        self.beta = beta          # list of weight arrays, one per treated unit (len = n_control each)
        self.M = baseline         # counterfactual panel
        self.individual_te = individual_te
        self.V = V                # predictor importance weights when covariates are used
        self.control_units = control_units    # original row indices of control units
        self.treatment_units = treatment_units  # original row indices of treated units

    def _summary_internals(self):
        lines = []
        if not self.beta:
            return lines
        has_pval = self.individual_te and len(self.individual_te[0]) >= 3
        tu_list = self.treatment_units if self.treatment_units is not None else list(range(len(self.beta)))
        lines.append(f"{'n_treated_units':<24s}: {len(self.beta)}")
        if self.control_units is not None:
            lines.append(f"{'n_donor_units':<24s}: {len(self.control_units)}")
        lines.append(f"{'  (weights per unit)':<24s}: result.beta  (list of arrays)")
        lines.append(f"{'  (unit indices)':<24s}: result.treatment_units, result.control_units")
        if self.individual_te:
            lines.append("")
            lines.append("per-unit ATT:")
            for k, (tu, W) in enumerate(zip(tu_list, self.beta)):
                te_row = next((r for r in self.individual_te if r[0] == tu), None)
                if te_row is None:
                    continue
                pv_str = f"  p={te_row[2]:.4g}" if has_pval else ""
                W = np.asarray(W)
                nz_support = int(np.sum(W > 1e-6))
                lines.append(f"  unit {tu:<6d}: tau={te_row[1]:.4g}{pv_str}  ({nz_support} donors)")
        return lines


class OLSSCPanelSolver(PanelSolver):
    """
    OLS / synthetic-control weights with simplex constraints.

    Parameters
    ----------
    Y : ndarray, shape (N, T)
        Panel outcomes (units x time).
    Z : ndarray, shape (N, T)
        Treatment indicators; first period with any ``Z==1`` defines the end of
        the pre-period for all units (common ``T0`` index).
    X : ndarray, shape (N, K), optional
        Time-invariant covariates per unit (same row order as ``Y``).
    pval : bool, optional
        If True, run a placebo-style comparison and attach approximate p-values
        to unit-level effects. Only supported when ``X is None`` (no covariates).
    """

    def __init__(self, Y, Z, X=None, pval=False):
        if pval and X is not None:
            raise ValueError(
                "pval=True is only supported when X is None (no covariates)."
            )
        self._O_raw = np.asarray(Y, dtype=float)   # (N, T) original outcome panel
        self._Z_raw = np.asarray(Z, dtype=float)   # (N, T) original treatment mask
        # All internal logic uses (T, N) convention; transpose (N, T) inputs once here.
        self.Y = np.asarray(Y, dtype=float).T
        self.X = np.asarray(X, dtype=float).T if X is not None else None
        (
            self.T0,
            self.Y0,
            self.Y1,
            self.X0,
            self.X1,
            self.control_units,
            self.treatment_units,
        ) = self.preprocess(np.asarray(Z, dtype=float).T)
        self.individual_te = np.zeros(len(self.treatment_units))
        self.pval = pval

    def preprocess(self, Z):
        """
        Split outcomes into treated vs control columns and locate pre-period length.

        Parameters
        ----------
        Z : ndarray, shape (T, N)
            Treatment matrix (already transposed to internal convention).

        Returns
        -------
        T0 : int
            Index of the first period with any treatment (pre-period is ``0:T0``).
        Y0 : ndarray
            Control columns of ``Y`` (time x n_control).
        Y1 : ndarray
            Treated columns of ``Y``.
        X0, X1 : ndarray or None
            Covariates restricted to control / treated columns if ``X`` was given.
        control_units, treatment_units : ndarray of int
            Column indices of control and treated units.
        """
        control_units = np.where(np.all(Z == 0, axis=0))[0]
        treatment_units = np.where(np.any(Z == 1, axis=0))[0]
        Y0 = self.Y[:, control_units]
        Y1 = self.Y[:, treatment_units]
        if self.X is not None:
            X0 = self.X[:, control_units]
            X1 = self.X[:, treatment_units]
        else:
            X0 = None
            X1 = None
        T0 = np.where(Z.any(axis=1))[0][0]
        return T0, Y0, Y1, X0, X1, control_units, treatment_units

    def ols_inference(self, Y1, Y0, X1=None, X0=None):
        """
        Fit synthetic-control weights on the pre-period and extrapolate.

        Returns counterfactual path ``M = Y0 @ W``, ATT on post periods, weights
        ``W``, and optional predictor weights ``V`` when covariates are used.

        Parameters
        ----------
        Y1 : ndarray, shape (T,) or (T, 1)
            Treated unit outcome series.
        Y0 : ndarray, shape (T, n_control)
            Control unit outcomes.
        X1 : ndarray, shape (K,), optional
            Covariates for the treated unit.
        X0 : ndarray, shape (K, n_control), optional
            Covariates for control units.

        Returns
        -------
        M : ndarray, shape (T,)
            Synthetic control prediction for the treated unit's column.
        tau : float
            Mean post-treatment gap ``mean(Y1 - M)`` after ``T0``.
        W : ndarray
            Simplex weights on control units.
        V : ndarray or None
            Predictor weights if covariates are used; otherwise None.
        """
        y_c = Y0[: self.T0]
        y_t = Y1[: self.T0]

        if X1 is not None:
            # Covariate path: alternate between optimising W (donor weights) and
            # V (predictor importance weights).  V lives in a low-dimensional
            # space (K covariates), so the outer loop over V stays with SLSQP;
            # only the inner high-dimensional W optimisation is replaced.
            v_start = np.full(X0.shape[0], 1.0 / X0.shape[0])

            def optimize_W(V):
                return _solve_simplex_qp_weighted(X0, X1, V)

            def loss_v_at_V(V_raw):
                W_at_v = optimize_W(np.clip(V_raw, 0.0, None))
                r = y_t - y_c.dot(W_at_v)
                return float(r.dot(r)) / len(y_t)

            def v_eq_simplex(V):
                return np.sum(V) - 1.0

            V_out = fmin_slsqp(
                loss_v_at_V,
                v_start,
                bounds=[(0.0, 1.0)] * len(v_start),
                f_eqcons=v_eq_simplex,
                acc=1e-6,
                disp=False,
            )
            V = _slsqp_minimizer(V_out)
            W = optimize_W(V)
        else:
            V = None
            W = _solve_simplex_qp(y_c, y_t)

        M = Y0 @ W
        tau = np.mean((Y1 - M)[self.T0 :])
        return M, tau, W, V

    def fit(self):
        """
        Estimate counterfactuals and (optional) unit-level placebo p-values.

        Returns
        -------
        OLSSCResult
            Result object.  Key attributes: ``tau`` (average ATT float),
            ``baseline`` (counterfactual panel N x T), ``individual_te``
            (per-unit ``[unit_idx, tau_hat]`` list, extended to
            ``[unit_idx, tau_hat, p_value]`` when ``pval=True``),
            ``beta`` (list of simplex weight vectors).
        """
        T = len(self.Y1)
        V = []
        weights = []
        tau = 0.0
        M = np.copy(self.Y)
        self.individual_te = []
        for i, s in enumerate(self.treatment_units):
            Y1_s = self.Y1[:, i].reshape((T,))
            if self.X is not None:
                K = len(self.X1)
                X1_s = self.X1[:, i].reshape((K,))
                counterfactual_s, tau_s, W_s, V_s = self.ols_inference(
                    Y1_s, self.Y0, X1_s, self.X0
                )
                V.append(V_s)
            else:
                counterfactual_s, tau_s, W_s, V_s = self.ols_inference(Y1_s, self.Y0)
            tau += tau_s
            M[:, s] = counterfactual_s
            weights.append(W_s)
            self.individual_te.append([s, tau_s])

        tau /= len(self.treatment_units)

        if self.pval:
            self.individual_te = self.permutation_test()

        res = OLSSCResult(
            baseline=M.T,  # return (N, T) to match all other solvers
            tau=tau,
            individual_te=self.individual_te,
            beta=weights,
            V=V,
            control_units=list(self.control_units),
            treatment_units=list(self.treatment_units),
        )
        res.O = self._O_raw
        res.Z = self._Z_raw
        return res

    def permutation_test(self):
        """
        Placebo effects for each control unit vs other controls; rank-based p-values.

        For every control column, treat it as a pseudo-treated unit, rebuild
        synthetic-control weights using the remaining donors, and record the
        implied post-period gap. Ranks treated vs control gaps to assign
        ``p ≈ (rank)/(n)`` for each treated unit.

        Returns
        -------
        list of list
            Rows ``[unit_index, tau_hat, approximate_p]`` for treated units only,
            sorted by unit index.
        """
        T = len(self.Y1)
        individual_te_control = []
        for i, cu in enumerate(self.control_units):
            Y1_s = self.Y0[:, i].reshape((T,))
            Y0_loo = np.hstack((self.Y0[:, :i], self.Y0[:, i + 1 :]))
            # ols_inference returns (M, tau, W, V); unpack all four for SciPy compatibility.
            _, tau_s, _, _ = self.ols_inference(Y1_s, Y0_loo)
            individual_te_control.append([cu, tau_s])

        sorted_te = sorted(
            self.individual_te + individual_te_control,
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        n = len(sorted_te)
        p_values = []
        treat_set = set(self.treatment_units)
        for i, unit_te in enumerate(sorted_te):
            if unit_te[0] in treat_set:
                p_values.append(unit_te + [round((i + 1) / n, 4)])
        p_values = sorted(p_values, key=lambda x: x[0])
        return p_values


def ols_synthetic_control(O, Z, X=None):
    """
    Convenience wrapper: fit :class:`OLSSCPanelSolver` and return counterfactual and ATT.

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Outcomes (units x time).
    Z : ndarray, shape (N, T)
        Treatment indicators.
    X : ndarray, shape (N, K), optional
        Time-invariant covariates per unit.

    Returns
    -------
    M : ndarray, shape (N, T)
        Counterfactual panel (same shape as ``O``).
    tau : float
        Average treated-unit effect (mean of unit-level ATTs).
    """
    solver = OLSSCPanelSolver(O, Z, X)
    res = solver.fit()
    return res.M, res.tau
