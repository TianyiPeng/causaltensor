import numpy as np
from causaltensor.matlib.util import transform_to_3D
import causaltensor.matlib.util as util


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

class Result():
    pass

class OLSPanelSolver():
    """Solve the OLS regression for panel data with covariates and missing data
    
        Y ~ X * beta
    """
    def __init__(self, X, Omega=None, is_sparse_X=True):
        """
        X: 3D float numpy array (n,m,p)
            The covariates matrix. The last dimension is the index of covariates.
        Omega: 2D bool numpy array (n,m)
            Indicator matrix (1: observed, 0: missing).
        is_sparse_X: bool
            if True, then remove the elements if all zero for covariates X
        """
        if is_sparse_X:
            relevant_index = (np.sum(np.abs(X) > 1e-9, axis=2) > 0) #if all zero for X,  then not included 
        else:
            relevant_index = np.ones_like(X[:, :, 0], dtype=bool)
        if Omega is not None: #if there are missing entries
            relevant_index = (relevant_index & Omega.astype(bool)) #if missing, then not included
        X = X[relevant_index, :].astype(float) # compress X to the shape of (l, p) where l is the number of relevant observations and p is the number of covariates 
        Xinv = np.linalg.inv(X.T @ X) #compute the inverse of the covariance matrix
        self.X = X
        self.relevant_index = relevant_index
        self.Xinv = Xinv

    def fit(self, O):
        """Solve the OLS regression for panel data with covariates and missing data

        Parameters
        ----------
        O: 2D float numpy array
            The observation matrix. 
        Returns
        -------
        res: Result
            The result of OLS regression.
            res.beta: 1D numpy array (p, )
                The estimated coefficients for Y~X*beta.
        """
        O = O[self.relevant_index] #select non-zero entries, the resulting shape of O is (l, )
        beta = self.Xinv @ (self.X.T @ O)
        res = Result()
        res.beta = beta
        return res
 
class FixedEffectPanelSolver():
    """ Solve the OLS regression for panel data with covariates, missing data, and fixed effects

        Y ~ X * beta + ai + bj
        
        The implementation is based on the partial regrerssion method (which speeds up the computation significantly comparing to naive OLS):
            Let demean_Y be the residule of Y ~ ai + bj
            Let demean_X be the residule of X ~ ai + bj
            Solve beta by demean_Y ~ demean_X * beta
    """

    def __init__(self, fixed_effects='two-way', X=None, Omega=None):
        """ 
        fixed_effects: ['two-way']
            two-way fixed effects or one-way fixed effects (to be implemented)
        X: 3D float numpy array (n,m,p)
            The covariates matrix. The last dimension is the index of covariates.
        Omega: 2D bool numpy array (n,m)
            Indicator matrix (1: observed, 0: missing).
        """
        self.fixed_effects = fixed_effects
        if fixed_effects != 'two-way':
            raise NotImplementedError('Only two-way fixed effects are implemented.')
        self.X = X
        self.Omega = Omega

        if (X is not None):
            demean_X = np.zeros_like(X)
            for i in range(X.shape[2]):
                demean_X[:, :, i], _, _ = self.demean(X[:, :, i])
            self.demean_X = demean_X
            self.OLS_solver = OLSPanelSolver(demean_X, Omega)

    def demean(self, O, eps=1e-7, max_iter=2000):
        """ demean O by row and column (regress O by ai + bj on self.Omega)
        """
        if self.Omega is None:
            self.Omega = np.ones_like(O)
        n1 = O.shape[0]
        n2 = O.shape[1]     
        one_row = np.ones((1, n2))
        one_col = np.ones((n1, 1))
        Ω_row_sum = np.sum(self.Omega, axis = 1).reshape((n1, 1))
        Ω_column_sum = np.sum(self.Omega, axis = 0).reshape((n2, 1))
        Ω_row_sum[Ω_row_sum==0] = 1
        Ω_column_sum[Ω_column_sum==0] = 1
        b = np.zeros((n2, 1))
        for T in range(max_iter):
            a = np.sum(self.Omega*(O-one_col.dot(b.T)), axis=1).reshape((n1, 1)) / Ω_row_sum

            b_new = np.sum(self.Omega*(O-a.dot(one_row)), axis=0).reshape((n2, 1)) / Ω_column_sum

            if (np.sum((b_new - b)**2) < eps * np.sum(b**2)):
                break
            b = b_new
        return O - a.dot(one_row) - one_col.dot(b.T), a, b

    def fit(self, O):
        """ Solve the OLS regression for panel data with covariates, missing data, and fixed effects
        Parameters
        ----------
        O: 2D numpy array
            The observation matrix.
        Returns
        -------
        res: Result
            The result of OLS regression.
            res.beta: 1D numpy array (p, ) if X is not None
            res.row_fixed_effects: 2D numpy array (n, 1)
            res.column_fixed_effects: 2D numpy array (m, 1)
            res.fitted_value: 2D numpy array (n, m)
                fitted_value of O ~ X * beta + ai + bj
        """
        res = Result()
        demean_O, a, b = self.demean(O)
        if (self.X is not None):
            res_OLS = self.OLS_solver.fit(demean_O)
            res.beta = res_OLS.beta

            residual, a, b = self.demean(O - np.sum(res.beta * self.X, axis=2))
            res.row_fixed_effects = a
            res.column_fixed_effects = b
            res.fitted_value = a + b.T + np.sum(res.beta * self.X, axis=2) 
        else:
            res.row_fixed_effects = a
            res.column_fixed_effects = b
            res.fitted_value = a + b.T
        return res
    
class MCNNMPanelSolver():
    """ 
    Solve the matrix completion problem with nuclear norm regularizer and fixed effects for panel data with covariates and missing data
        reference: https://arxiv.org/pdf/1710.10251.pdf
    """

    def __init__(self, O, Z, X=None, Omega=None, fixed_effects = 'two-way'):
        """
        O: 2D numpy array
            The observation matrix.
        Z: 2D bool numpy array
            The treatment matrix.
        X: 3D float numpy array (n,m,p) or 2D float numpy array (n,m) or a list of 2D float numpy array
            The covariates matrix. The last dimension is the index of covariates.
        Omega: 2D bool numpy array (n,m)
            Indicator matrix (1: observed, 0: missing).
        fixed_effects: ['two-way']
            two-way fixed effects or one-way fixed effects (to be implemented)
        """
        self.O = O
        if (Omega is None):
            Omega = np.ones_like(O, dtype=bool)
        Omega = Omega.astype(bool)
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

    def solve_with_regularizer(self, l, M_init=None, eps=1e-7, max_iter=2000):
        """ Solve the matrix completion problem with nuclear norm regularizer and fixed effects
        Parameters
        ----------
        l: float
            Nuclear norm regularizer.
        M_init: 2D numpy array or None
            Initial guess of the underlying low-rank matrix.
        eps: float 
            Convergence threshold.
        max_iter: int
            Maximum number of iterations.
        Returns
        -------
        res: Result
            res.M: 2D numpy array
                The estimated low-rank matrix.
            res.row_fixed_effects: 2D numpy array (n, 1)
            res.column_fixed_effects: 2D numpy array (m, 1)
            res.beta: 1D numpy array (p, ) if X is not None
            res.baseline_model: 2D numpy array
                The estimated baseline model (M+ai+bj+beta*X).
            res.tau: float
                The estimated treatment effect.
        """
        M = M_init
        if M is None:
            M = np.zeros_like(self.O)

        for T in range(max_iter):
            res = self.FE_beta_solver.fit(self.O - M)
            M_new = util.SVD_soft((self.O-res.fitted_value) * self.Omega + M * (1-self.Omega), l)
            if (np.sum((M-M_new)**2) < eps * np.sum(M**2)):
                break
            M = M_new
        res.M = M 
        res.baseline_model = res.fitted_value + M
        res.tau = np.sum((self.O - res.baseline_model)*self.Z) / np.sum(self.Z)
        return res

    def solve_with_suggested_rank(self, suggest_r=1):
        suggest_r = min(suggest_r, self.O.shape[0])
        suggest_r = min(suggest_r, self.O.shape[1])
        coef = 1.1
        u, s, vh = np.linalg.svd(self.O*self.Omega, full_matrices = False)
        l = s[1]*coef    

        res = self.solve_with_regularizer(l)
        l = l / coef
        while (True):
            res_new = self.solve_with_regularizer(l, M_init=res.M)
            if (np.linalg.matrix_rank(res_new.M) >= suggest_r):
                return res_new
            res = res_new
            l = l / coef

    def solve_with_cross_validation(self, K=3, list_l = []):
        """
        Implement the K-fold cross validation in https://arxiv.org/pdf/1710.10251.pdf
        """
        np.random.seed(42) #for reproducibility
        O = self.O
        Omega = self.Omega
        def MSE_validate(res, valid_Ω):
            return np.sum((valid_Ω)*((O-res.baseline_model)**2)) / np.sum(valid_Ω)
            
        #K-fold cross validation
        train_list = []
        valid_list = []
        p = np.sum(Omega) / np.size(Omega)
        for k in range(K):
            select = np.random.rand(O.shape[0], O.shape[1]) <= p
            train_list.append(Omega * select)
            valid_list.append(Omega * (1 - select)) 

        if (len(list_l) == 0):# auto-selection of a list of regularization parameters
            _, s, _ = np.linalg.svd(O*Omega, full_matrices = False)
            l = s[1]*1.1 #large enough regularization parameter
            for i in range(5):
                list_l.append(l)
                l /= 1.1       

        error = np.zeros((K, len(list_l)))   
        for k in range(K):
            solver = MCNNMPanelSolver(O = self.O, Z = self.Z, X=self.X, Omega=(train_list[k]&self.Omega), fixed_effects=self.fixed_effects)
            
            M = None
            for i, l in enumerate(list_l):
                res = solver.solve_with_regularizer(l, M_init=M)
                #import IPython; IPython.embed() 
                error[k, i] = MSE_validate(res, valid_list[k])
                M =res.M
        index = error.sum(axis=0).argmin()
        l_opt = list_l[index]
        res = self.solve_with_regularizer(l_opt)
        return res


    