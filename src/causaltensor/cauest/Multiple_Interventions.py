import numpy as np
from causaltensor.cauest import projection_T_orthogonal

def solve_OLS(O, Z, tau_init):
    '''
    Regress O on Z_1, Z_2, ...; the regression coefficient is tau_1, tau_2, ...
    Using alternating minimization to solve the OLS problem. 
    
    Parameters:
    -------------
    O: the matrix to be regressed
    Z: a list of intervention matrices
    tau_init: the initial values of tau

    Returns
    -----------
    tau: the vector of regression coefficients
    '''
    num_treat = len(Z)
    tau = np.copy(tau_init)
    for T in range(2000):
        O_sub = np.copy(O)
        for k in range(num_treat):
            O_sub -= tau[k] * Z[k]
        tau_new = np.zeros(len(tau))

        for k in range(num_treat): 
            tau_new[k] = tau[k] + np.sum(O_sub * Z[k]) / np.sum(Z[k]**2) #update tau[k]
            O_sub = O_sub + (tau[k]-tau_new[k]) * Z[k] # update O_sub for solving other taus

        if (np.linalg.norm(tau_new - tau) <= 1e-6 * np.linalg.norm(tau)):
            return tau_new
        tau = tau_new
    print('not converge')
    return tau_new

def non_convex_panel_regression(O, Z, r=2):
    '''
    Estimate the baseline matrix M and the treatment effect tau using non-convex panel regression.

    Parameters:
    -------------
    O: observation matrix
    Z: a list of intervention matrices
    r: the pre-defined rank for the baseline matrix M

    Returns
    -----------
    M: the baseline matrix 
    tau: the treatment effect (a vector)
    '''

    num_treat = len(Z)
    tau = np.zeros(num_treat)
    for T in range(2000):
        M = np.copy(O)
        for k in range(num_treat):
            M -= tau[k]*Z[k]
        u,s,vh = np.linalg.svd(M, full_matrices=False)
        s[r:] = 0
        M = (u*s).dot(vh)
        tau_new = solve_OLS(O-M, Z, tau) 
        if (np.linalg.norm(tau_new - tau) < 1e-6 * np.linalg.norm(tau)):
            return M, tau_new
        tau = tau_new
    return M, tau_new


def panel_regression_CI(M, Zs, E):
    '''
    Compute the confidence interval of taus using the first-order approximation.
    
    Parameters:
    -------------
    M: the (approximate) baseline matrix
    Zs: a list of intervention matrices
    E: the (approximate) noise matrix

    Returns
    -----------
    CI: a kxk matrix that charaterizes the asymptotic covariance matrix of treatment estimation from non-convex panel regression,
        where k is the number of treatments
    '''
    Z = []
    num_treat = len(Zs)
    for k in range(num_treat):
        Z.append(np.copy(Zs[k]))
    X = np.zeros((np.size(M), num_treat))
    for k in range(num_treat):
        X[:, k] = np.reshape(projection_T_orthogonal(Z[k], M), -1)

    A = (np.linalg.inv(X.T@X)@X.T) 
    CI = (A * np.reshape(E**2, -1)) @ A.T
    return CI