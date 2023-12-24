import numpy as np
from causaltensor.cauest.result import Result
from causaltensor.matlib.util import transform_to_3D

class PanelSolver():
    def __init__(self, Z):
        if isinstance(Z, np.ndarray):
            if len(Z.shape) == 2:
                self.return_tau_scalar = True
        self.Z = transform_to_3D(Z)

class OLSResult():
    def __init__(self, beta=None):
        self.beta = beta

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
    
class FixedEffectResult():
    def __init__(self, beta=None, row_fixed_effects=None, column_fixed_effects=None, fitted_value=None):
        self.beta = beta
        self.row_fixed_effects = row_fixed_effects
        self.column_fixed_effects = column_fixed_effects
        self.fitted_value = fitted_value

 
class FixedEffectPanelSolver(PanelSolver):
    """ Solve the OLS regression for panel data with covariates, missing data, and fixed effects

        Y ~ X * beta + ai + bj
        
        The implementation is based on the partial regrerssion method (which speeds up the computation significantly comparing to naive OLS):
            Let demean_Y be the residule of Y ~ ai + bj
            Let demean_X be the residule of X ~ ai + bj
            Solve beta by demean_Y ~ demean_X * beta
    """

    def __init__(self, fixed_effects='two-way', X=None, Omega=None,
                 demean_eps=1e-7, demean_max_iter=2000):
        """ 
        fixed_effects: ['two-way']
            two-way fixed effects
            TODO: implement one-way fixed effects
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
        res = FixedEffectResult()
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
    