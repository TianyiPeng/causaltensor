import numpy as np
from causaltensor.matlib.util import transform_to_3D
import causaltensor.matlib.util as util
from causaltensor.cauest.panel_solver import FixedEffectPanelSolver
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


def soft_impute(O, Omega, l, eps=1e-7, M_init=None, max_iter=2000):
    """Impute the missing entries of O under Ω with nuclear norm regularizer l.

    Parameters
    ----------    
    O: 2D numpy array
        Observed data.
    Ω: 2D numpy array
        Indicator matrix (1: observed, 0: missing).
    l: float
        Nuclear norm regularizer.
    eps: float
        Convergence threshold.
    M_init: 2D numpy array or None
        Initial guess of the underlying low-rank matrix.
    max_iter: int
        Maximum number of iterations.

    Returns
    -------
    M_new: 2D numpy array
        Imputed matrix.
    """
    if (M_init is None):
        M = np.zeros_like(O)
    else:
        M = M_init
    for T in range(max_iter):
        M_new = util.SVD_soft(O * Omega + M * (1-Omega), l)
        if (np.linalg.norm(M-M_new) < np.linalg.norm(M)*eps):
            break
        M = M_new
    return M_new

class MCNNMResult(Result):
    def __init__(self, baseline = None, M = None, tau=None, beta=None, row_fixed_effects=None, column_fixed_effects=None, return_tau_scalar=False):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.M = M # the low-rank component of the baseline model

class MCNNMPanelSolver(PanelSolver):
    """
    Matrix Completion with Nuclear Norm Minimisation (MC-NNM).

    Recovers a low-rank outcome matrix under a nuclear-norm penalty, jointly
    estimating row / column fixed effects and optional covariate coefficients
    on the control cells ``(1 - Z) ⊙ Omega``.  Regularisation is chosen by
    K-fold cross-validation (Athey et al., 2021).

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Observed outcome panel (units x time).
    Z : ndarray, shape (N, T), dtype bool or {0, 1}
        Binary treatment mask.
    X : ndarray, shape (N, T, K) or (N, T) or list of ndarray, optional
        Time-varying covariates.
    Omega : ndarray, shape (N, T), dtype bool, optional
        Observation mask (1 = present).  Defaults to all ones.
    fixed_effects : str, optional
        ``'two-way'`` (default).

    References
    ----------
    Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
    Matrix completion methods for causal panel data models. *JASA*, 116(536),
    1716-1730.

    Examples
    --------
    >>> solver = MCNNMPanelSolver(O, Z)
    >>> result = solver.fit()          # runs cross-validation by default
    >>> result.tau                     # ATT estimate
    >>> result.baseline                # low-rank + FE counterfactual (N x T)
    """

    def __init__(self, O=None, Z=None, X=None, Omega=None, fixed_effects = 'two-way'):
        """
        Parameters
        ----------
        O : ndarray, shape (N, T)
            Observed outcome panel (units x time).
        Z : ndarray, shape (N, T), dtype bool or {0, 1}
            Binary treatment mask.  Only block / single-treated-row patterns
            are supported (multiple-treatment extension is a TODO).
        X : ndarray, shape (N, T, K) or (N, T) or list of ndarray, optional
            Time-varying covariates; last dimension is the covariate index.
        Omega : ndarray, shape (N, T), dtype bool, optional
            Observation mask (1 = present, 0 = missing).  Defaults to all ones.
            Treated entries are internally re-classified as *missing* for the
            MC objective.
        fixed_effects : str, optional
            ``'two-way'`` (default).  One-way support is not yet implemented.
        """
        self.O = O
        if (Omega is None):
            Omega = np.ones_like(Z[:, :], dtype=bool)
        Omega = Omega.astype(bool)
        if np.sum(Z==1) + np.sum(Z == 0) != Z.shape[0]*Z.shape[1]:
            raise ValueError('Z should only consist of 0/1 in matrix completion solver') 
        Z = Z.astype(bool)
        self.raw_Omega = Omega
        self.Z = (Z & Omega) # we only care the treatment matrix for observed entries 
        self.Omega = ((1 - Z) & Omega) # we treat the treatment matrix as missing entries
        if (np.sum(np.sum(self.Omega, axis=1)==0)>0 or np.sum(np.sum(self.Omega, axis=0)==0) > 0):
            raise ValueError("Since a whole row or a whole column is treated, the matrix completion algorithm won't work!")
        
        self.X = X 
        if self.X is not None:
            self.X = transform_to_3D(X)
        self.fixed_effects = fixed_effects
        self.FE_beta_solver = FixedEffectPanelSolver(fixed_effects=self.fixed_effects, X=self.X, Omega=self.Omega)
        self.return_tau_scalar = False

    def fit(self, cross_validation=True, K=2, list_l=None):
        """Fit the MC-NNM model.

        Parameters
        ----------
        cross_validation : bool, optional
            If True (default), select the regularization parameter via K-fold
            cross-validation as in Athey et al. (2021). If False, fall back to
            the fixed regularizer stored in the solver (not recommended without
            specifying ``list_l``).
        K : int, optional
            Number of cross-validation folds (default 2).
        list_l : list of float or None, optional
            Candidate regularization values; auto-selected when None.

        Returns
        -------
        MCNNMResult
        """
        if self.O is None:
            raise ValueError("O must be provided at construction time: MCNNMPanelSolver(O, Z)")
        if cross_validation:
            return self.solve_with_cross_validation(self.O, K=K, list_l=list_l)
        else:
            if list_l is not None and len(list_l) > 0:
                return self.solve_with_regularizer(self.O, l=list_l[0])
            raise ValueError("Provide list_l when cross_validation=False")

    def solve_with_regularizer(self, O=None, l=None, M_init=None, eps=1e-7, max_iter=2000):
        """Solve the matrix completion problem with a fixed nuclear-norm regulariser.

        Parameters
        ----------
        O : ndarray, shape (N, T)
            Observed outcome panel.
        l : float
            Nuclear-norm regularisation strength.
        M_init : ndarray or None, optional
            Warm-start for the low-rank matrix.
        eps : float, optional
            Convergence threshold (default 1e-7).
        max_iter : int, optional
            Maximum number of alternating iterations (default 2000).

        Returns
        -------
        MCNNMResult
            Result with ``tau`` (ATT), ``baseline`` (M + FE panel),
            ``M`` (low-rank component), ``row_fixed_effects``,
            ``column_fixed_effects``, and ``beta`` (covariate coefficients).
        """
        M = M_init
        if M is None:
            M = np.zeros_like(O)

        for T in range(max_iter):
            res = self.FE_beta_solver.fit(O - M)
            M_new = util.SVD_soft((O-res.fitted_value) * self.Omega + M * (1-self.Omega), l)
            if (np.sum((M-M_new)**2) < eps * np.sum(M**2)):
                break
            M = M_new

        baseline = res.fitted_value + M
        tau = np.sum((O - baseline)*self.Z) / np.sum(self.Z)
        res_new = MCNNMResult(baseline = baseline, M = M, tau = tau, 
                          beta = res.beta,
                          row_fixed_effects = res.row_fixed_effects, 
                          column_fixed_effects = res.column_fixed_effects,
                          return_tau_scalar = self.return_tau_scalar)
        return res_new

    def solve_with_suggested_rank(self, O=None, suggest_r=1):
        suggest_r = min(suggest_r, O.shape[0])
        suggest_r = min(suggest_r, O.shape[1])
        coef = 1.1
        u, s, vh = np.linalg.svd(O*self.Omega, full_matrices = False)
        l = s[1]*coef    

        res = self.solve_with_regularizer(O=O, l=l)
        l = l / coef
        T = 2000
        for i in range(T):
            res_new = self.solve_with_regularizer(O=O, l=l, M_init=res.M)
            if (np.linalg.matrix_rank(res_new.M) > suggest_r): # we hope to minimize the l while keeping the rank of M to be suggest_r
                return res
            res = res_new
            l = l / coef
        return res

    def solve_with_cross_validation(self, O=None, K=2, list_l=None):
        """
        Implement the K-fold cross validation in https://arxiv.org/pdf/1710.10251.pdf
        """
        if list_l is None:
            list_l = []
        else:
            list_l = list(list_l)
        np.random.seed(42) #for reproducibility
        raw_Omega = self.raw_Omega
        def MSE_validate(res, valid_Ω):
            return np.sum((valid_Ω)*((O-res.baseline)**2)) / np.sum(valid_Ω)
            
        #K-fold cross validation
        train_list = []
        valid_list = []
        p = np.sum(self.Omega) / np.size(raw_Omega) # due to the treatment, the ratio of the missing entries
        for k in range(K):
            select = np.random.rand(O.shape[0], O.shape[1]) <= p
            train_list.append(raw_Omega * select)
            valid_list.append(raw_Omega * (1 - select)) 

        if (len(list_l) == 0):# auto-selection of a list of regularization parameters
            _, s, _ = np.linalg.svd(O*self.Omega, full_matrices = False)
            l = s[1] #large enough regularization parameter
            for i in range(5):
                list_l.append(l)
                l /= 2    

        error = np.ones((K, len(list_l))) * np.inf
        for k in range(K):
            #print(np.sum(self.Omega * (1-self.Z)))
            #print(np.sum((train_list[k]&self.Omega) * (1-self.Z)))

            solver = MCNNMPanelSolver(Z = self.Z, X=self.X, Omega=train_list[k], fixed_effects=self.fixed_effects)
            
            M = None
            for i, l in enumerate(list_l):
                res = solver.solve_with_regularizer(O=O, l=l, M_init=M)
                #import IPython; IPython.embed() 
                error[k, i] = MSE_validate(res, valid_list[k])
                M =res.M
        index = error.sum(axis=0).argmin()
        l_opt = list_l[index]
        res = self.solve_with_regularizer(O=O, l=l_opt)
        return res


#backward compatability
def MC_NNM_with_l(O, Omega, l):
    solver = MCNNMPanelSolver(Z = 1-Omega)
    res = solver.solve_with_regularizer(O, l)
    return res.M, res.row_fixed_effects, res.column_fixed_effects, res.tau

def MC_NNM_with_suggested_rank(O, Omega, suggest_r=1):
    solver = MCNNMPanelSolver(Z = 1-Omega)
    res = solver.solve_with_suggested_rank(O, suggest_r)
    return res.M, res.row_fixed_effects, res.column_fixed_effects, res.tau

def MC_NNM_with_cross_validation(O, Omega, K=5, list_l=None):
    solver = MCNNMPanelSolver(Z = 1-Omega)
    res = solver.solve_with_cross_validation(O, K, list_l)
    return res.M, res.row_fixed_effects, res.column_fixed_effects, res.tau