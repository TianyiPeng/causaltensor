"""
Synthetic control via constrained least squares (Abadie-Diamond-Hainmueller style).

Weights on donor units are chosen to minimize pre-period outcome error subject to
simplex constraints; optional covariates enter a nested predictor-reweighting step.
"""

import numpy as np
from scipy.optimize import fmin_slsqp
from sklearn.metrics import mean_squared_error

from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


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
    ):
        super().__init__(baseline=baseline, tau=tau, return_tau_scalar=return_tau_scalar)
        self.beta = beta  # control unit weights (per treated unit) or list thereof
        self.M = baseline  # counterfactual panel
        self.individual_te = individual_te
        self.V = V  # predictor importance when covariates are used


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
        def loss_v(W, y_c, y_t):
            return np.mean((y_t - y_c.dot(W)) ** 2)

        def w_eq_simplex(W, y_c, y_t):
            del y_c, y_t
            return np.sum(W) - 1.0

        y_c = Y0[: self.T0]
        y_t = Y1[: self.T0]
        w_start = np.array([1.0 / y_c.shape[1]] * y_c.shape[1])

        if X1 is not None:
            v_start = np.array([1.0 / X0.shape[0]] * X0.shape[0])

            def v_eq_simplex(V, W, X0, X1, y_c, y_t):
                del W, X0, X1, y_c, y_t
                return np.sum(V) - 1.0

            def w_eq_with_V(W, V, X0, X1):
                del V, X0, X1
                return np.sum(W) - 1.0

            def loss_w(W, V, X0, X1):
                return mean_squared_error(X1, X0.dot(W), sample_weight=V)

            def optimize_W(W, V, X0, X1):
                out = fmin_slsqp(
                    loss_w,
                    W,
                    bounds=[(0.0, 1.0)] * len(W),
                    f_eqcons=w_eq_with_V,
                    args=(V, X0, X1),
                    disp=False,
                    full_output=True,
                )
                return _slsqp_minimizer(out)

            def optimize_V(V, W, X0, X1, y_c, y_t):
                w_at_v = optimize_W(W, V, X0, X1)
                return loss_v(w_at_v, y_c, y_t)

            V_out = fmin_slsqp(
                optimize_V,
                v_start,
                args=(w_start, X0, X1, y_c, y_t),
                bounds=[(0.0, 1.0)] * len(v_start),
                disp=False,
                f_eqcons=v_eq_simplex,
                acc=1e-6,
            )
            V = _slsqp_minimizer(V_out)
            W = optimize_W(w_start, V, X0, X1)
        else:
            V = None
            W_out = fmin_slsqp(
                loss_v,
                w_start,
                args=(y_c, y_t),
                f_eqcons=w_eq_simplex,
                bounds=[(0.0, 1.0)] * len(w_start),
                disp=False,
            )
            W = _slsqp_minimizer(W_out)

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

        return OLSSCResult(
            baseline=M.T,  # return (N, T) to match all other solvers
            tau=tau,
            individual_te=self.individual_te,
            beta=weights,
            V=V,
        )

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
