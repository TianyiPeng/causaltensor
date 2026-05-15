import numpy as np
from causaltensor.cauest.result import Result, _fmt_coefs, _fmt_fe
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.panel_solver import FixedEffectPanelSolver

class DIDResult(Result):
    def __init__(self, baseline=None, tau=None, beta=None, row_fixed_effects=None,
                 column_fixed_effects=None, return_tau_scalar=False):
        super().__init__(baseline=baseline, tau=tau, return_tau_scalar=return_tau_scalar)
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.M = baseline  # for backward compatibility

    def _summary_internals(self):
        lines = []
        if self.beta is not None:
            lines.append(_fmt_coefs(self.beta, "covariate_coefs"))
        if self.row_fixed_effects is not None:
            lines.append(_fmt_fe(self.row_fixed_effects, "row_FE (unit)"))
        if self.column_fixed_effects is not None:
            lines.append(_fmt_fe(self.column_fixed_effects, "col_FE (time)"))
        return lines

        
class DIDPanelSolver(PanelSolver):
    """
    Difference-in-Differences via two-way fixed effects (TWFE).

    Estimates the ATT by regressing outcomes on unit fixed effects, time fixed
    effects, and the treatment indicator ``Z`` (and optionally covariates ``X``):

    .. math::

        \\min_{a,b,\\tau} \\sum_{ij} (O_{ij} - a_i - b_j - \\tau Z_{ij})^2

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Observed outcome panel (units x time).
    Z : ndarray, shape (N, T)
        Binary treatment mask (1 = treated).
    X : ndarray, shape (N, T, K), optional
        Additional time-varying covariates (K features).  When provided their
        coefficients are estimated jointly with ``tau``.
    Omega : ndarray, shape (N, T), optional
        Observation mask (1 = observed).  Defaults to all ones.
    fixed_effects : str, optional
        Only ``'two-way'`` is currently implemented.

    Examples
    --------
    >>> solver = DIDPanelSolver(O, Z)
    >>> result = solver.fit()
    >>> result.tau       # ATT scalar
    >>> result.baseline  # fitted a_i + b_j surface (N x T)
    """

    def __init__(self, O=None, Z=None, X=None, Omega=None, fixed_effects='two-way', **kwargs):
        self._Z_raw = np.asarray(Z, dtype=float) if Z is not None else None
        super().__init__(Z)
        self.O = O
        self.X = X
        self.Omega = Omega
        self.fixed_effects = fixed_effects
        if fixed_effects != 'two-way':
            raise NotImplementedError('Only two-way fixed effects are implemented.')
        if X is None:
            new_X = self.Z
        else:
            new_X = np.concatenate([self.Z, X], axis=2)
        self.fixed_effects_solver = FixedEffectPanelSolver(X = new_X, Omega=Omega, fixed_effects=fixed_effects, **kwargs)

    def fit(self, O=None):
        """
        Estimate ATT via TWFE regression.

        Parameters
        ----------
        O : ndarray, shape (N, T), optional
            Outcome panel.  If omitted, uses the ``O`` passed at construction.

        Returns
        -------
        DIDResult
            ``.tau``                 -- ATT scalar (or array if ``X`` covariates).
            ``.baseline``  / ``.M`` -- fitted ``a_i + b_j`` surface (N x T).
            ``.row_fixed_effects``   -- unit FE vector (N,).
            ``.column_fixed_effects`` -- time FE vector (T,).
            ``.beta``               -- covariate coefficients (None if no ``X``).
        """
        if O is None:
            O = self.O
        if O is None:
            raise ValueError("O must be provided either at construction time or to fit()")
        res_fe = self.fixed_effects_solver.fit(O)
        k = self.Z.shape[0]
        tau = res_fe.beta[:k]
        beta = res_fe.beta[k:] if self.X is not None else None
        res = DIDResult(baseline=res_fe.fitted_value, tau=tau, beta=beta,
                        row_fixed_effects=res_fe.row_fixed_effects,
                        column_fixed_effects=res_fe.column_fixed_effects,
                        return_tau_scalar=self.return_tau_scalar)
        res.O = O
        res.Z = self._Z_raw
        return res
#deprecated
#for backward compatability  
def DID(O, Z):
    solver = DIDPanelSolver(O, Z)
    res = solver.fit()
    return res.M, res.tau
