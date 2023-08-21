import numpy as np
import causaltensor.matlib.util as util

def debias(M, tau, Z, l):
    u, s, vh = util.svd_fast(M)
    r = np.sum(s / np.cumsum(s) >= 1e-6)
    u = u[:, :r]
    vh = vh[:r, :]

    PTperpZ = np.zeros_like(Z)
    for k in np.arange(Z.shape[2]):
        PTperpZ[:, :, k] = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z[:, :, k]).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))

    D = np.zeros((Z.shape[2], Z.shape[2]))
    for k in np.arange(Z.shape[2]):
        for m in np.arange(k, Z.shape[2]):
            D[k, m] = np.sum(PTperpZ[:, :, k] * PTperpZ[:, :, m])
            D[m, k] = D[k, m]

    Delta = np.array([l * np.sum(Z[:, :, k]*(u.dot(vh))) for k in range(Z.shape[2])]) 

    tau_delta =  np.linalg.pinv(D) @ Delta
    tau_debias = tau - tau_delta

    PTZ = Z - PTperpZ
    M_debias = M + l * u.dot(vh) + np.sum(PTZ * tau_delta.reshape(1, 1, -1), axis=2)
    return M_debias, tau_debias

def transform_Z(Z):
    """
        convert Z to a 3-dimension numpy array with the last dimension being the index of interventions
    """
    if isinstance(Z, list): #if Z is a list of numpy arrays
        Z = np.stack(Z, axis = 2) 
    elif Z.ndim == 2: #if a single Z
        Z = Z.reshape(Z.shape[0], Z.shape[1], 1)
    return Z.astype(float)

def prepare_OLS(Z):
    ### Select non-zero entries for OLS (optmizing sparsity of Zs)
    small_index = (np.sum(np.abs(Z) > 1e-9, axis=2) > 0)
    X = Z[small_index, :].astype(float) # small X
    ## X.shape = (#non_zero entries of Zs, num_treat)
    Xinv = np.linalg.inv(X.T @ X)
    return small_index, X, Xinv

def DC_PR_with_l(O, Z, l, initial_tau = None, eps = 1e-6):
    """
        De-biased Convex Panel Regression with the regularizer l. 

        Parameters
        -------------
        O : 2d float numpy array
            Observation matrix.
        Z : a list of 2d float numpy array or a single 2d/3d float numpy array
            Intervention matrices. If Z is a list, then each element of the list is a 2d numpy array. If Z is a single 2d numpy array, then Z is a single intervention matrix. If Z is a 3d numpy array, then Z is a collection of intervention matrices with the last dimension being the index of interventions.
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
    Z = transform_Z(Z) ## Z is (n1 x n2 x num_treat) numpy array
    if initial_tau is None:
        tau = np.zeros(Z.shape[2])
    else:
        tau = initial_tau

    small_index, X, Xinv = prepare_OLS(Z)
    
    for T in range(2000):
        #### SVD to find low-rank M
        M = util.SVD_soft(O - np.tensordot(Z, tau,  axes=([2], [0])), l)
        #### OLS to get tau
        y = (O - M)[small_index] #select non-zero entries
        tau_new = Xinv @ (X.T @ y)
        #### Check convergence
        if (np.linalg.norm(tau_new - tau) < eps * np.linalg.norm(tau)):
            return M, tau
        tau = tau_new
    return M, tau

def non_convex_PR(O, Z, r, initial_tau = None, eps = 1e-6):
    """
        Non-Convex Panel Regression with the rank r

        Parameters
        -------------
        O : 2d float numpy array
            Observation matrix.
        Z : a list of 2d float numpy array or a single 2d/3d float numpy array
            Intervention matrices. If Z is a list, then each element of the list is a 2d numpy array. If Z is a single 2d numpy array, then Z is a single intervention matrix. If Z is a 3d numpy array, then Z is a collection of intervention matrices with the last dimension being the index of interventions.
        r : int
            rank constraint for the baseline matrix.
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
    Z = transform_Z(Z) ## Z is (n1 x n2 x num_treat) numpy array
    if initial_tau is None:
        tau = np.zeros(Z.shape[2])
    else:
        tau = initial_tau

    small_index, X, Xinv = prepare_OLS(Z)
    
    for T in range(2000):
        #### SVD to find low-rank M
        M = util.SVD(O - np.tensordot(Z, tau,  axes=([2], [0])), r) #hard truncation
        #### OLS to get tau
        y = (O - M)[small_index] #select non-zero entries
        tau_new = Xinv @ (X.T @ y)
        #### Check convergence
        if (np.linalg.norm(tau_new - tau) < eps * np.linalg.norm(tau)):
            return M, tau
        tau = tau_new
    return M, tau

def solve_tau(O, Z):
    small_index, X, Xinv = prepare_OLS(Z)
    y = O[small_index] #select non-zero entries
    tau = Xinv @ (X.T @ y)
    return tau 

def DC_PR_with_suggested_rank(O, Z, suggest_r = 1, non_convex = False):
    """
        De-biased Convex Panel Regression with the suggested rank. Gradually decrease the nuclear-norm regularizer l until the rank of the next-iterated estimator exceeds r.
    
        :param O: observation matrix
        :param Z: intervention matrix
    
    """
    Z = transform_Z(Z) ## Z is (n1 x n2 x num_treat) numpy array
    ## determine pre_tau
    pre_tau = solve_tau(O, Z)

    if non_convex == False:
        ## determine l
        coef = 1.1
        _, s, _ = util.svd_fast(O-np.tensordot(Z, pre_tau,  axes=([2], [0])))
        l = s[1]*coef
        ##inital pre_M and pre_tau for current l
        pre_M, pre_tau = DC_PR_with_l(O, Z, l, initial_tau = pre_tau)
        l = l / coef
        while (True):
            M, tau = DC_PR_with_l(O, Z, l, initial_tau = pre_tau)
            if (np.linalg.matrix_rank(M) > suggest_r):
                M_debias, tau_debias = debias(pre_M, pre_tau, Z, l*coef)
                M = util.SVD(M_debias, suggest_r)
                tau = tau_debias 
                break 
            pre_M = M
            pre_tau = tau
            l = l / coef
    else:
        M, tau = non_convex_PR(O, Z, suggest_r, initial_tau = pre_tau)
        
    CI = panel_regression_CI(M, Z, O-M-np.tensordot(Z, tau,  axes=([2], [0])))
    standard_deviation = np.sqrt(np.diag(CI))
    if len(tau) == 1:
        return M, tau[0], standard_deviation[0]
    else:
        return M, tau, standard_deviation

def DC_PR_auto_rank(O, Z, spectrum_cut = 0.002):
    s = np.linalg.svd(O, full_matrices = False, compute_uv=False)
    suggest_r = np.sum(np.cumsum(s**2) / np.sum(s**2) <= 1-spectrum_cut)
    return DC_PR_with_suggested_rank(O, Z, suggest_r = suggest_r)


def projection_T_orthogonal(Z, M):
    u, s, vh = np.linalg.svd(M, full_matrices = False)
    r = np.sum(s / np.cumsum(s) >= 1e-6)
    u = u[:, :r]
    vh = vh[:r, :]
    PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
    return PTperpZ

def panel_regression_CI(M, Z, E):
    '''
    Compute the confidence interval of taus using the first-order approximation.
    
    Parameters:
    -------------
    M: the (approximate) baseline matrix
    Z: a list of intervention matrices
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

    X = np.zeros((Z.shape[0]*Z.shape[1], Z.shape[2]))
    for k in np.arange(Z.shape[2]):
        X[:, k] = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z[:, :, k]).dot(np.eye(vh.shape[1]) - vh.T.dot(vh)).reshape(-1)

    A = (np.linalg.inv(X.T@X)@X.T) 
    CI = (A * np.reshape(E**2, -1)) @ A.T
    return CI