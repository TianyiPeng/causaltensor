import numpy as np
from causaltensor.cauest.result import Result
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.panel_solver import FixedEffectPanelSolver

class DIDResult(Result):
    def __init__(self, baseline = None, tau=None, beta=None, row_fixed_effects=None, column_fixed_effects=None, return_tau_scalar=False):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.M = baseline # for backward compatability
class DIDPanelSolver(PanelSolver):
    def __init__(self, Z=None, X=None, Omega=None, fixed_effects='two-way', **kwargs):
        super().__init__(Z)
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

    def fit(self, O):
        res_fe = self.fixed_effects_solver.fit(O)
        k = self.Z.shape[0]
        tau = res_fe.beta[:k]
        beta = res_fe.beta[k:] if self.X is not None else None
        res = DIDResult(baseline = res_fe.fitted_value, tau = tau, beta = beta, 
                        row_fixed_effects = res_fe.row_fixed_effects, 
                        column_fixed_effects = res_fe.column_fixed_effects,
                        return_tau_scalar = self.return_tau_scalar)
        return res

#deprecated
#for backward compatability  
def DID(O, Z):
    solver = DIDPanelSolver(Z)
    res = solver.fit(O)
    return res.M, res.tau