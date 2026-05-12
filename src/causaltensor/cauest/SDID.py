import numpy as np
import cvxpy as cp
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result

'''
    An implementation of "Synthetic Difference-in-Differences" from [1]
    
    Created by Tianyi Peng, 2021/03/01
    Credit to Andy Zheng for the revised version, 2022/01/15
    
    [1] Arkhangelsky, Dmitry, Susan Athey, David A. Hirshberg, Guido W. Imbens, and Stefan Wager. 2021. 
        "Synthetic Difference-in-Differences." American Economic Review, 111 (12): 4088-4118
'''
class SDIDResult(Result):
    def __init__(self, baseline = None, tau=None, beta=None, row_fixed_effects=None, column_fixed_effects=None, return_tau_scalar=False):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.M = baseline 



class SDIDPanelSolver(PanelSolver):
    """
    Synthetic Difference-in-Differences (SDID).

    Computes unit weights ``w`` (matching pre-treatment trends) and time
    weights ``l`` (down-weighting periods far from the treatment onset), then
    applies a weighted two-way fixed-effects regression to estimate the ATT
    (Arkhangelsky et al., 2021).

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Observed outcome panel (units x time).
    Z : ndarray, shape (N, T)
        Binary treatment mask (1 = treated, block / staggered).
    X_cov : ndarray, shape (N, T, P), optional
        Exogenous time-varying covariates.  When provided, the algorithm
        first residualises outcomes against ``X_cov`` at each time period
        (see footnote 4 in Arkhangelsky et al.).
    treat_units : list of int, optional
        Indices of treated rows.  Inferred automatically from ``Z`` when set
        to ``[-1]`` (default).
    starting_time : int, optional
        Column index of the first treated period.  Inferred automatically
        from ``Z`` when set to ``-1`` (default).

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G., & Wager, S.
    (2021). Synthetic difference-in-differences. *American Economic Review*,
    111(12), 4088-4118.

    Examples
    --------
    >>> solver = SDIDPanelSolver(O, Z)
    >>> result = solver.fit()
    >>> result.tau       # ATT scalar
    >>> result.baseline  # weighted counterfactual surface (N x T)
    """

    def __init__(self, O=None, Z=None, X_cov=None, treat_units = [-1], starting_time = -1):
        super().__init__(Z)
        if self.Z.shape[2] == 1:
            self.Z = self.Z.reshape(self.Z.shape[0], self.Z.shape[1])
        self.X = O 
        self.treat_units = treat_units
        self.starting_time = starting_time
        self.X_cov = X_cov
        if (starting_time == -1):
            self.SDID_preprocess()

    def SDID_preprocess(self):
        n1, n2 = self.X.shape
        self.treat_units = []
        for i in range(n1):
            if self.Z[i, -1] != 0:
                self.treat_units.append(i)
        if len(self.treat_units) == 0:
            print('no treated unit, or the treatment is not a block!!')
            return
        i = self.treat_units[0]
        for j in range(n2-1, -1, -1):
            if self.Z[i, j] == 0:
                break
        self.starting_time = j + 1

    def adjust_for_covariates(self):
        """
        For each time period t, regress the outcome Y[:, t] on the covariates X_cov[:, t]
        (with intercept) and replace Y[:, t] by the residuals.
        
        Input:
        - self.X is an (n x T) outcome matrix.
        - self.X_cov is an (n x T x p) array of covariates.
        
        Output:
            self.X will contain the residuals computed as: 
            Y_res = Y - X_cov * beta_t, for each time period t.
        """
        n, T = self.X.shape
        X_resid = np.zeros_like(self.X)
        
        for t in range(T):
            X_t = self.X_cov[:, t, :]
            X_t_aug = np.concatenate([np.ones((n, 1)), X_t], axis=1)
            y_t = self.X[:, t]
            beta_t, _, _, _ = np.linalg.lstsq(X_t_aug, y_t, rcond=None)
            y_pred = X_t_aug @ beta_t
            X_resid[:, t] = y_t - y_pred
            
        self.X = X_resid
        
    
    def _solve_all_steps(self, X):
        """Run the four SDID steps on the given panel matrix X.

        Returns (tau, M, feasible) where feasible=False if any CVXPY
        sub-problem was infeasible (solution value is None).
        """
        Nco = len(self.donor_units)
        Ntr = len(self.treat_units)
        Tpre = self.starting_time
        Tpost = X.shape[1] - self.starting_time

        ##Step 1, Compute regularization parameter
        D = X[self.donor_units, 1:Tpre] - X[self.donor_units, :Tpre-1]
        D_bar = np.mean(D)
        z_square = np.mean((D - D_bar)**2) * (np.sqrt(Ntr * Tpost))

        ##Step 2, Compute w^{sdid}
        w = cp.Variable(Nco)
        w0 = cp.Variable(1)
        mean_treat = np.mean(X[self.treat_units, :Tpre], axis=0)
        prob = cp.Problem(
            cp.Minimize(
                cp.sum_squares(w0 + X[self.donor_units, :Tpre].T @ w - mean_treat)
                + z_square * Tpre * cp.sum_squares(w)),
            [np.eye(Nco) @ w >= 0, np.ones(Nco).T @ w == 1])
        prob.solve()

        if w.value is None:
            return None, None, False

        w_sdid = np.zeros(X.shape[0])
        w_sdid[self.donor_units] = w.value
        w_sdid[self.treat_units] = 1.0 / Ntr

        ##Step 3, Compute l^{sdid}
        l = cp.Variable(Tpre)
        l0 = cp.Variable(1)
        mean_treat = np.mean(X[self.donor_units, Tpre:], axis=1)
        prob = cp.Problem(
            cp.Minimize(
                cp.sum_squares(l0 + X[self.donor_units, :Tpre] @ l - mean_treat)),
            [np.eye(Tpre) @ l >= 0, np.ones(Tpre).T @ l == 1])
        prob.solve()

        if l.value is None:
            return None, None, False

        l_sdid = np.zeros(X.shape[1])
        l_sdid[:Tpre] = l.value
        l_sdid[Tpre:] = 1.0 / Tpost

        ##Step 4, Compute SDID estimator
        n1 = X.shape[0]
        n2 = X.shape[1]
        weights = w_sdid.reshape((n1, 1)) @ l_sdid.reshape((1, n2))

        a = np.zeros((n1, 1))
        b = np.zeros((n2, 1))
        tau = 0.0
        one_row = np.ones((1, n2))
        one_col = np.ones((n1, 1))
        M = np.zeros((n1, n2))
        for _ in range(1000):
            a_new = np.sum((X - tau*self.Z - one_col.dot(b.T))*weights, axis=1).reshape((n1, 1)) / np.sum(weights, axis=1).reshape((n1, 1))
            b_new = np.sum((X - tau*self.Z - a.dot(one_row))*weights, axis=0).reshape((n2, 1)) / np.sum(weights, axis=0).reshape((n2, 1))
            if (np.sum((b_new - b)**2) < 1e-7 * np.sum(b**2) and
                    np.sum((a_new - a)**2) < 1e-7 * np.sum(a**2)):
                a, b = a_new, b_new
                M = a.dot(one_row) + one_col.dot(b.T)
                tau = np.sum(self.Z * (X - M) * weights) / np.sum(self.Z * weights)
                break
            a = a_new
            b = b_new
            M = a.dot(one_row) + one_col.dot(b.T)
            tau = np.sum(self.Z * (X - M) * weights) / np.sum(self.Z * weights)

        return tau, M, True

    def fit(self):
        """
        Estimate ATT via SDID.

        Returns
        -------
        SDIDResult
            ``.tau``      -- ATT scalar.
            ``.baseline`` / ``.M`` -- weighted fixed-effects surface (N x T).

        Raises
        ------
        RuntimeError
            If CVXPY fails to find a feasible solution for both the original
            and variance-normalised panels.
        """
        if self.X_cov is not None:
            self.adjust_for_covariates()

        self.donor_units = []
        for i in range(self.X.shape[0]):
            if (i not in self.treat_units):
                self.donor_units.append(i)

        tau, M, feasible = self._solve_all_steps(self.X)

        if not feasible:
            # Rescale to unit std to improve CVXPY numerical conditioning,
            # then unscale tau and M back to the original units.
            scale = float(np.nanstd(self.X))
            if not (np.isfinite(scale) and scale > 0):
                scale = 1.0
            tau_scaled, M_scaled, feasible_retry = self._solve_all_steps(self.X / scale)
            if feasible_retry:
                tau = tau_scaled * scale
                M = M_scaled * scale
            else:
                raise RuntimeError(
                    "SDID failed: CVXPY returned no feasible solution for both the "
                    "original outcomes and a variance-normalized rescaling. "
                    "Check treatment timing, panel scales, and pre/post support."
                )

        res = SDIDResult(baseline=M, tau=tau)
        return res
    
# backward compatibility
def SDID(O, Z, X_cov=None, treat_units = [-1], starting_time = -1):
    solver = SDIDPanelSolver(O, Z, X_cov, treat_units, starting_time)
    res = solver.fit()
    return res.tau
