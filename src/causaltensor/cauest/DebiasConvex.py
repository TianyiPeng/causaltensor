import numpy as np

def debias(M, tau, Z, l):
    u, s, vh = np.linalg.svd(M, full_matrices = False)
    #r = np.linalg.matrix_rank(M)
    r = np.sum(s >= 1e-6)
    #print(s[:r+1], r)
    u = u[:, :r]
    vh = vh[:r, :]
    t1 = l * np.sum(Z*(u.dot(vh)))
    PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
    t2 = np.sum(PTperpZ**2)

    #print('PUVZ {}, PUZ {}, PVZ {}, PTperZ {}, Z {}'.format(np.sum((u.T.dot(Z).dot(vh.T))**2), np.sum((u.dot(u.T).dot(Z))**2), np.sum((Z.dot(vh.T.dot(vh)))**2), t2, np.sum(Z**2)))
    M_debias = M + l * u.dot(vh) + t1 / t2 * (Z - PTperpZ)
    #print(t1, t2)
    return tau - t1 / t2, M_debias

def DC_PR_with_l(O, Z, l, suggest_tau = 0, eps = 1e-6):
    tau = suggest_tau
    num_treat = np.sum(Z)
    for T in range(2000):
        ## update M
        u,s,vh = np.linalg.svd(O - tau*Z, full_matrices=False)
        s = np.maximum(s-l, 0)
        M = (u*s).dot(vh)

        #print(np.sum(s))
        #print(s)
        
        tau_new = np.sum(Z * (O - M)) / num_treat # update tau
        #print('tau(t) is {}, tau(t+1) is {}'.format(tau, tau_new))
        if (np.abs(tau_new - tau) < eps * np.abs(tau)):
            #print('iterations', T)
            return M, tau, 'successful'
        tau = tau_new
    return M, tau, 'fail'

def non_convex_algorithm(O, Z, r, tau = 0):
    M = O
    for T in range(2000):
        u,s,vh = np.linalg.svd(O - tau*Z, full_matrices=False)
        s[r:] = 0
        M = (u*s).dot(vh)
        tau_new = np.sum(Z*(O-M)) / np.sum(Z)
        if (np.abs(tau_new - tau) < 1e-6 * np.abs(tau)):
            return M, tau
        tau = tau_new
    return M, tau

#gradually decrease l until the rank of the output estimator is more than r
def DC_PR_with_suggested_rank(O, Z, suggest_r = 1):
    coef = 1.1
    pre_tau = np.sum(O*Z)/np.sum(Z)
    ## determine l
    
    _, s, _ = np.linalg.svd(O-pre_tau*Z, full_matrices = False)
    l = s[1]*coef
    ##inital pre_M and pre_tau for current l
    pre_M, pre_tau, info = DC_PR_with_l(O, Z, l, suggest_tau = pre_tau)
    l = l / coef
    while (True):
        M, tau, info = DC_PR_with_l(O, Z, l, suggest_tau = pre_tau)
        #print(info, l, np.linalg.matrix_rank(M))
        if (info != 'fail' and np.linalg.matrix_rank(M) > suggest_r):
            ###pre_M, pre_tau is the right choice
            tau_debias, M_debias = debias(pre_M, pre_tau, Z, l*coef)
            return M_debias, tau_debias, pre_M, pre_tau
        pre_M = M
        pre_tau = tau
        l = l / coef

def DC_PR_auto_rank(O, Z):
    s = np.linalg.svd(O, full_matrices = False, compute_uv=False)
    suggest_r = np.sum(np.cumsum(s**2) / np.sum(s**2) <= 0.998)
    return DC_PR_with_suggested_rank(O, Z, suggest_r = suggest_r)


def projection_T_orthogonal(Z, M):
    u, s, vh = np.linalg.svd(M, full_matrices = False)
    r = np.sum(s >= 1e-5)
    u = u[:, :r]
    vh = vh[:r, :]
    PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
    return PTperpZ


def std_debiased_convex(O, Z, M, tau):
    '''
    Estimate the standard deviation of the Debiased-Convex algorithm. Refer to FaraisLiPeng2021 for the formula.

    Parameters:
    -------------
    O: observation matrix
    Z: intervention matrix
    M_debias: debiased version of M
    tau_debias: debiased version of tau
    M: orignal M (non-debiased version)
    tau: orignal tau (non-debiased version) 

    Returns
    -----------
    estimated_sigma_level: estimated STD for tau_debias
    '''

    #E_hat = O - M_debias - tau_debias * Z
    E_hat = O - M - tau * Z # we find this formula seems more stable than the above

    projected_Z = projection_T_orthogonal(Z, M)

    estimated_sigma_level = np.sqrt(np.sum((projected_Z**2)*(E_hat**2))) / np.sum(projected_Z**2)
    
    #if mode == "homo":
    #    estimated_sigma_level = np.sqrt(np.mean(E_hat**2) / np.sum(projected_Z**2)) 
        
    return estimated_sigma_level
    