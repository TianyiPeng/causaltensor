import numpy as np
import cvxpy as cp

from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


class SDIDResult(Result):
    def __init__(self, baseline=None, tau=None, beta=None, row_fixed_effects=None, column_fixed_effects=None,
                 return_tau_scalar=False, unit_weights=None, time_weights=None,
                 cohort_effects=None, cohort_weights=None, cohort_units=None,
                 cohort_unit_weights=None, cohort_time_weights=None):
        super().__init__(baseline=baseline, tau=tau, return_tau_scalar=return_tau_scalar)
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.M = baseline
        self.unit_weights = unit_weights
        self.time_weights = time_weights
        self.cohort_effects = cohort_effects or {}
        self.cohort_weights = cohort_weights or {}
        self.cohort_units = cohort_units or {}
        self.cohort_unit_weights = cohort_unit_weights or {}
        self.cohort_time_weights = cohort_time_weights or {}

    def _summary_internals(self):
        lines = []
        if len(self.cohort_effects) > 1:
            lines.append(f"{'adoption_cohorts':<24s}: {len(self.cohort_effects)}")
            lines.append(f"{'cohort_effects':<24s}: result.cohort_effects")
            lines.append(f"{'cohort_weights':<24s}: result.cohort_weights")
        elif self.unit_weights is not None:
            w = np.asarray(self.unit_weights)
            nz = int(np.sum(w > 1e-6))
            if nz > 0:
                top_i = int(np.argmax(w))
                lines.append(f"{'unit_weights':<24s}: {nz} nonzero  (top: unit {top_i}, w={w[top_i]:.4g})")
            lines.append(f"{'  (full array)':<24s}: result.unit_weights")
        if self.time_weights is not None:
            l = np.asarray(self.time_weights)
            lines.append(f"{'time_weights':<24s}: {int(np.sum(l > 1e-6))} nonzero")
            lines.append(f"{'  (full array)':<24s}: result.time_weights")
        return lines


class SDIDPanelSolver(PanelSolver):
    """
    Synthetic Difference-in-Differences (SDID).

    Supports monotone block and staggered adoption. Treatment timing is inferred
    directly from Z. For staggered adoption, treated units are grouped by first
    treatment time; common-onset SDID is fit separately for each cohort using
    never-treated units as donors, and cohort effects are aggregated by their
    number of treated unit-time cells.

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Observed outcome panel.
    Z : ndarray, shape (N, T)
        Binary treatment mask. Once treatment starts for a unit, it must remain on.
    X_cov : ndarray, shape (N, T, P), optional
        Exogenous time-varying covariates.
    """

    def __init__(self, O=None, Z=None, X_cov=None):
        if O is None or Z is None:
            raise ValueError("O and Z must be provided.")

        O = np.asarray(O, dtype=float)
        Z_raw = np.asarray(Z, dtype=float)

        if O.ndim != 2 or Z_raw.ndim != 2:
            raise ValueError("O and Z must both have shape (N, T).")
        if O.shape != Z_raw.shape:
            raise ValueError(f"O and Z must have the same shape; got O={O.shape}, Z={Z_raw.shape}.")
        if not np.all((Z_raw == 0) | (Z_raw == 1)):
            raise ValueError("Z must contain only 0/1 values.")

        super().__init__(Z_raw)

        if self.Z.shape[2] != 1:
            raise ValueError("SDID currently supports one treatment only.")

        self.Z = self.Z[:, :, 0]
        self._Z_raw = Z_raw
        self.X = O.copy()
        self.X_cov = None if X_cov is None else np.asarray(X_cov, dtype=float)

        if self.X_cov is not None and (self.X_cov.ndim != 3 or self.X_cov.shape[:2] != O.shape):
            raise ValueError("X_cov must have shape (N, T, P) with first two dimensions matching O.")

        self._preprocess_treatment()

    def _preprocess_treatment(self):
        self.treat_units = np.where(np.any(self.Z == 1, axis=1))[0]
        self.donor_units = np.where(np.all(self.Z == 0, axis=1))[0]

        if len(self.treat_units) == 0:
            raise ValueError("SDID requires at least one treated unit.")
        if len(self.donor_units) == 0:
            raise ValueError("SDID requires at least one never-treated donor unit.")

        self.starting_times = {}
        self.cohorts = {}

        for i in self.treat_units:
            start = int(np.where(self.Z[i] == 1)[0][0])

            if start < 2:
                raise ValueError(f"Unit {i} is first treated at t={start}; SDID requires at least two pre-treatment periods.")
            if not np.all(self.Z[i, start:] == 1):
                raise ValueError(f"SDID supports monotone adoption only; treatment switches off for unit {i}.")

            self.starting_times[int(i)] = start
            self.cohorts.setdefault(start, []).append(int(i))

        self.starting_time = next(iter(self.cohorts)) if len(self.cohorts) == 1 else None

    def adjust_for_covariates(self):
        """Residualise outcomes against X_cov separately at each time period."""
        n, T = self.X.shape
        X_resid = np.zeros_like(self.X)

        for t in range(T):
            X_t = self.X_cov[:, t, :]
            X_t_aug = np.concatenate([np.ones((n, 1)), X_t], axis=1)
            beta_t, _, _, _ = np.linalg.lstsq(X_t_aug, self.X[:, t], rcond=None)
            X_resid[:, t] = self.X[:, t] - X_t_aug @ beta_t

        self.X = X_resid

    @staticmethod
    def _solve_all_steps(X, Z, treat_units, donor_units, starting_time):
        """Run ordinary common-onset SDID on one cohort-specific panel."""
        treat_units = np.asarray(treat_units, dtype=int)
        donor_units = np.asarray(donor_units, dtype=int)

        Nco = len(donor_units)
        Ntr = len(treat_units)
        Tpre = int(starting_time)
        Tpost = X.shape[1] - Tpre

        if Nco == 0 or Ntr == 0 or Tpre < 2 or Tpost <= 0:
            raise ValueError("Invalid SDID cohort: check donors, treated units, and pre/post periods.")

        # Step 1: regularization parameter
        D = X[donor_units, 1:Tpre] - X[donor_units, :Tpre - 1]
        D_bar = np.mean(D)
        z_square = np.mean((D - D_bar) ** 2) * np.sqrt(Ntr * Tpost)

        # Step 2: unit weights
        w = cp.Variable(Nco)
        w0 = cp.Variable(1)
        mean_treat = np.mean(X[treat_units, :Tpre], axis=0)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(w0 + X[donor_units, :Tpre].T @ w - mean_treat)
                        + z_square * Tpre * cp.sum_squares(w)),
            [w >= 0, cp.sum(w) == 1]
        )

        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            return None, None, False, None, None

        if w.value is None:
            return None, None, False, None, None

        w_sdid = np.zeros(X.shape[0])
        w_sdid[donor_units] = np.asarray(w.value).reshape(-1)
        w_sdid[treat_units] = 1.0 / Ntr

        # Step 3: time weights
        l = cp.Variable(Tpre)
        l0 = cp.Variable(1)
        mean_post = np.mean(X[donor_units, Tpre:], axis=1)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(l0 + X[donor_units, :Tpre] @ l - mean_post)),
            [l >= 0, cp.sum(l) == 1]
        )

        try:
            prob.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            return None, None, False, None, None

        if l.value is None:
            return None, None, False, None, None

        l_sdid = np.zeros(X.shape[1])
        l_sdid[:Tpre] = np.asarray(l.value).reshape(-1)
        l_sdid[Tpre:] = 1.0 / Tpost

        # Step 4: weighted TWFE
        n1, n2 = X.shape
        weights = w_sdid.reshape(n1, 1) @ l_sdid.reshape(1, n2)

        a = np.zeros((n1, 1))
        b = np.zeros((n2, 1))
        tau = 0.0
        one_row = np.ones((1, n2))
        one_col = np.ones((n1, 1))
        M = np.zeros((n1, n2))

        for _ in range(1000):
            row_den = np.sum(weights, axis=1).reshape(n1, 1)
            col_den = np.sum(weights, axis=0).reshape(n2, 1)

            a_new = np.sum((X - tau * Z - one_col @ b.T) * weights, axis=1).reshape(n1, 1) / row_den
            b_new = np.sum((X - tau * Z - a @ one_row) * weights, axis=0).reshape(n2, 1) / col_den

            M_new = a_new @ one_row + one_col @ b_new.T
            tau_new = np.sum(Z * (X - M_new) * weights) / np.sum(Z * weights)

            a_diff = np.sum((a_new - a) ** 2)
            b_diff = np.sum((b_new - b) ** 2)
            a_scale = max(np.sum(a ** 2), 1e-12)
            b_scale = max(np.sum(b ** 2), 1e-12)

            a, b, M, tau = a_new, b_new, M_new, tau_new

            if a_diff < 1e-7 * a_scale and b_diff < 1e-7 * b_scale:
                break

        return float(tau), M, True, w_sdid, l_sdid

    @classmethod
    def _solve_with_rescaling(cls, X, Z, treat_units, donor_units, starting_time):
        tau, M, feasible, w, l = cls._solve_all_steps(X, Z, treat_units, donor_units, starting_time)

        if feasible:
            return tau, M, w, l

        scale = float(np.nanstd(X))
        if not (np.isfinite(scale) and scale > 0):
            scale = 1.0

        tau, M, feasible, w, l = cls._solve_all_steps(X / scale, Z, treat_units, donor_units, starting_time)

        if not feasible:
            raise RuntimeError(
                "SDID failed on both the original outcomes and a variance-normalized rescaling."
            )

        return tau * scale, M * scale, w, l

    def fit(self):
        """
        Estimate ATT under block or staggered adoption.

        Block treatment is the one-cohort special case.
        Staggered cohort effects are weighted by the number of treated unit-time cells.
        """
        if self.X_cov is not None:
            self.adjust_for_covariates()

        N, T = self.X.shape
        M_global = np.full_like(self.X, np.nan, dtype=float)

        cohort_effects = {}
        cohort_weights = {}
        cohort_units = {}
        cohort_unit_weights = {}
        cohort_time_weights = {}

        weighted_tau_sum = 0.0
        total_treated_cells = 0

        for start in sorted(self.cohorts):
            cohort_global = np.asarray(self.cohorts[start], dtype=int)
            units_global = np.concatenate([self.donor_units, cohort_global])

            X_sub = self.X[units_global]
            Z_sub = self.Z[units_global]

            n_donors = len(self.donor_units)
            donor_local = np.arange(n_donors)
            treat_local = np.arange(n_donors, len(units_global))

            tau_g, M_sub, w_sub, l_sub = self._solve_with_rescaling(
                X_sub, Z_sub, treat_local, donor_local, start
            )

            n_treated_cells = int(np.sum(Z_sub[treat_local]))
            weighted_tau_sum += tau_g * n_treated_cells
            total_treated_cells += n_treated_cells

            cohort_effects[start] = tau_g
            cohort_weights[start] = n_treated_cells
            cohort_units[start] = cohort_global.tolist()
            cohort_time_weights[start] = l_sub

            w_global = np.zeros(N)
            w_global[units_global] = w_sub
            cohort_unit_weights[start] = w_global

            # Counterfactual rows for treated units belong uniquely to this cohort.
            for local_idx, global_idx in zip(treat_local, cohort_global):
                M_global[global_idx] = M_sub[local_idx]

            # Block case: only one cohort, so the whole baseline is well-defined.
            if len(self.cohorts) == 1:
                M_global[units_global] = M_sub

        tau = weighted_tau_sum / total_treated_cells

        # Preserve old unit_weights/time_weights behavior for ordinary block SDID.
        if len(self.cohorts) == 1:
            start = next(iter(self.cohorts))
            unit_weights = cohort_unit_weights[start]
            time_weights = cohort_time_weights[start]
        else:
            unit_weights = None
            time_weights = None

        res = SDIDResult(
            baseline=M_global,
            tau=tau,
            unit_weights=unit_weights,
            time_weights=time_weights,
            cohort_effects=cohort_effects,
            cohort_weights=cohort_weights,
            cohort_units=cohort_units,
            cohort_unit_weights=cohort_unit_weights,
            cohort_time_weights=cohort_time_weights,
        )

        res.O = self.X
        res.Z = self.Z
        return res


# backward compatibility
def SDID(O, Z, X_cov=None):
    return SDIDPanelSolver(O, Z, X_cov=X_cov).fit().tau