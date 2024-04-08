import numpy as np
import causaltensor.matlib.util as util
from causaltensor.matlib.util import transform_to_3D
from causaltensor.cauest.result import Result
from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.panel_solver import FixedEffectPanelSolver

class DCResult(Result):
    def __init__(self, baseline = None, tau=None, std=None, return_tau_scalar=False):
        super().__init__(baseline = baseline, tau = tau, return_tau_scalar = return_tau_scalar)
        self.std = std
        self.M = baseline # for backward compatability

class DCPanelSolver(PanelSolver):
    def __init__(self, Z=None, O=None, suggest_r=None):
        """
        De-biased Convex Panel Regression with the regularizer l. 

        Parameters
        -------------
        O : 2d float numpy array
            Observation matrix.
        Z : a list of 2d float numpy array or a single 2d/3d float numpy array
            Intervention matrices. If Z is a list, then each element of the list is a 2d numpy array. If Z is a single 2d numpy array, then Z is a single intervention matrix. If Z is a 3d numpy array, then Z is a collection of intervention matrices with the last dimension being the index of interventions.
        """
        super().__init__(Z)
        self.O = O
        self.Z = transform_to_3D(Z) ## Z is (n1 x n2 x num_treat) numpy array
        self.suggest_R = suggest_r
        self.small_index, self.X, self.Xinv = self.prepare_OLS()


    def prepare_OLS(self):
        ### Select non-zero entries for OLS (optmizing sparsity of Zs)
        small_index = (np.sum(np.abs(self.Z) > 1e-9, axis=2) > 0)
        X = self.Z[small_index, :].astype(float) # small X
        ## X.shape = (#non_zero entries of Zs, num_treat)
        Xinv = np.linalg.inv(X.T @ X)
        return small_index, X, Xinv
      

    def fit(self, auto_rank = True, suggest_r = 1, spectrum_cut = 0.002, method='convex'):
        if auto_rank:
            M, tau, std = self.DC_PR_auto_rank(spectrum_cut=spectrum_cut, method=method)
        else:
            M, tau, std = self.DC_PR_with_suggested_rank(suggest_r=suggest_r, method=method)
        res = DCResult(baseline=M, tau=tau, std=std)
        return res

    def solve_tau(self, O):
        y = O[self.small_index] #select non-zero entries
        tau = self.Xinv @ (self.X.T @ y)
        return tau 

    def als(self, tau, eps, l=None, r=None):
        for T in range(2000):
            #### SVD to find low-rank M
            if l:
                M = util.SVD_soft(self.O - np.tensordot(self.Z, tau,  axes=([2], [0])), l)
            elif r:
                M = util.SVD(self.O - np.tensordot(self.Z, tau,  axes=([2], [0])), r) #hard truncation

            #### OLS to get tau
            tau_new = self.solve_tau(self.O - M)

            #### Check convergence
            if (np.linalg.norm(tau_new - tau) < eps * np.linalg.norm(tau)):
                return M, tau
            tau = tau_new
        return M, tau
    
    def debias(self, M, tau, l):
        u, s, vh = util.svd_fast(M)
        r = np.sum(s / np.cumsum(s) >= 1e-6)
        u = u[:, :r]
        vh = vh[:r, :]

        PTperpZ = np.zeros_like(self.Z)
        for k in np.arange(self.Z.shape[2]):
            PTperpZ[:, :, k] = util.remove_tangent_space_component(u, vh, self.Z[:, :, k])

        D = np.zeros((self.Z.shape[2], self.Z.shape[2]))
        for k in np.arange(self.Z.shape[2]):
            for m in np.arange(k, self.Z.shape[2]):
                D[k, m] = np.sum(PTperpZ[:, :, k] * PTperpZ[:, :, m])
                D[m, k] = D[k, m]

        Delta = np.array([l * np.sum(self.Z[:, :, k]*(u.dot(vh))) for k in range(self.Z.shape[2])]) 

        tau_delta = np.linalg.pinv(D) @ Delta
        tau_debias = tau - tau_delta

        PTZ = self.Z - PTperpZ
        M_debias = M + l * u.dot(vh) + np.sum(PTZ * tau_delta.reshape(1, 1, -1), axis=2)
        return M_debias, tau_debias
    
    def projection_T_orthogonal(self, M):
        u, s, vh = np.linalg.svd(M, full_matrices = False)
        r = np.sum(s / np.cumsum(s) >= 1e-6)
        u = u[:, :r]
        vh = vh[:r, :]
        PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(self.Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
        return PTperpZ

    def DC_PR_with_l(self, l, initial_tau = None, eps = 1e-6):
        """
        De-biased Convex Panel Regression with the regularizer l. 

        Parameters
        -------------
        l : float
            Regularizer for the nuclear norm.
        intial_tau : (num_treat,) float numpy array 
            Initial value(s) for tau.
        eps : float
            Convergence threshold.
        
        Returns
        -------------
        M : 2d float numpy array
            Estimated matrix.
        tau : (num_treat,) float numpy array
            Estimated treatment effects.
        """
        if initial_tau is None:
            tau = np.zeros(self.Z.shape[2])
        else:
            tau = initial_tau

        M, tau = self.als(tau, eps, l=l)
        return M, tau
    

    
    def non_convex_PR(self, r, initial_tau = None, eps = 1e-6):
        """
        Non-Convex Panel Regression with the rank r

        Parameters
        -------------
        r : int rank constraint for the baseline matrix.
        intial_tau : (num_treat,) float numpy array 
            Initial value(s) for tau.
        eps : float
            Convergence threshold.
        
        Returns
        -------------
        M : 2d float numpy array
            Estimated baseline matrix.
        tau : (num_treat,) float numpy array
            Estimated treatment effects.
        """
        if initial_tau is None:
            tau = np.zeros(self.Z.shape[2])
        else:
            tau = initial_tau

        M, tau = self.als(tau, eps, r=r)
        return M, tau


    def DC_PR_with_suggested_rank(self, suggest_r = 1, method = 'convex'):
        """
            De-biased Convex Panel Regression with the suggested rank. Gradually decrease the nuclear-norm regularizer l until the rank of the next-iterated estimator exceeds r.
        
            :param O: observation matrix
            :param Z: intervention matrix
        
        """
        ## determine pre_tau
        pre_tau = self.solve_tau(self.O)

        if method == 'convex' or method == 'auto':
            ## determine l
            coef = 1.1
            _, s, _ = util.svd_fast(self.O-np.tensordot(self.Z, pre_tau,  axes=([2], [0])))
            l = s[1]*coef
            ##inital pre_M and pre_tau for current l
            pre_M, pre_tau = self.DC_PR_with_l(l, initial_tau = pre_tau)
            l = l / coef
            while (True):
                M, tau = self.DC_PR_with_l(l, initial_tau = pre_tau)
                if (np.linalg.matrix_rank(M) > suggest_r):
                    M_debias, tau_debias = self.debias(pre_M, pre_tau, l*coef)
                    M = util.SVD(M_debias, suggest_r)
                    tau = tau_debias 
                    break
                pre_M = M
                pre_tau = tau
                l = l / coef
        if method == 'non-convex':
            M, tau = self.non_convex_PR(suggest_r, initial_tau = pre_tau)

        if method == 'auto':
            M1, tau1 = self.non_convex_PR(suggest_r, initial_tau = self.solve_tau(self.O))
            if np.linalg.matrix_rank(M) != suggest_r or np.linalg.norm(self.O-M-np.tensordot(self.Z, tau,  axes=([2], [0]))) > np.linalg.norm(self.O-M1-np.tensordot(self.Z, tau1,  axes=([2], [0]))):
                M = M1
                tau = tau1

        CI = self.panel_regression_CI(M, self.O-M-np.tensordot(self.Z, tau,  axes=([2], [0])))
        standard_deviation = np.sqrt(np.diag(CI))
        if len(tau) == 1:
            return M, tau[0], standard_deviation[0]
        else:
            return M, tau, standard_deviation


    def DC_PR_auto_rank(self, spectrum_cut = 0.002, method='convex'):
        s = np.linalg.svd(self.O, full_matrices = False, compute_uv=False)
        suggest_r = np.sum(np.cumsum(s**2) / np.sum(s**2) <= 1-spectrum_cut)
        return self.DC_PR_with_suggested_rank(suggest_r = suggest_r, method=method)



    def panel_regression_CI(self, M, E):
        '''
        Compute the confidence interval of taus using the first-order approximation.
        
        Parameters:
        -------------
        M: the (approximate) baseline matrix
        E: the (approximate) noise matrix

        Returns
        -----------
        CI: a kxk matrix that charaterizes the asymptotic covariance matrix of treatment estimation from non-convex panel regression,
            where k is the number of treatments
        '''
        u, s, vh = util.svd_fast(M)
        r = np.sum(s / np.cumsum(s) >= 1e-6)
        u = u[:, :r]
        vh = vh[:r, :]

        X = np.zeros((self.Z.shape[0]*self.Z.shape[1], self.Z.shape[2]))
        for k in np.arange(self.Z.shape[2]):
            X[:, k] = util.remove_tangent_space_component(u, vh, self.Z[:, :, k]).reshape(-1)

        A = (np.linalg.inv(X.T@X)@X.T) 
        CI = (A * np.reshape(E**2, -1)) @ A.T
        return CI
    


# backward compatibility
def DC_PR_auto_rank(O, Z, spectrum_cut = 0.002, method='convex'):
    solver = DCPanelSolver(Z, O)
    res = solver.fit(spectrum_cut=spectrum_cut, method=method)
    return res.M, res.tau, res.std

def DC_PR_with_suggested_rank(O, Z, suggest_r = 1, method = 'convex'):
    solver = DCPanelSolver(Z, O)
    res = solver.fit(auto_rank=False, suggest_r=suggest_r, method=method)
    return res.M, res.tau, res.std