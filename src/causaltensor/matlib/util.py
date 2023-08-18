import numpy as np

def noise_to_signal(X, M, Ω):
    return np.sqrt(np.sum((Ω*X - Ω*M)**2) / np.sum((Ω*M)**2))

def abs_mean(X, M, Ω):
    return np.sum(np.abs((X-M)*Ω)) / np.sum(Ω)

def svd_fast(M):
    is_swap = False
    if M.shape[0] > M.shape[1]:
        is_swap = True
        M = M.T

    A = M @ M.T
    u, ss, uh = np.linalg.svd(A, full_matrices=False)
    ss[ss < 1e-7] = 0
    s = np.sqrt(ss)
    sinv = 1.0 / (s + 1e-7*(s<1e-7))
    vh = sinv.reshape(M.shape[0], 1) * (uh @ M)

    if is_swap:
        return vh.T, s, u.T
    else:
        return u, s, vh

## least-squares solved via single SVD
def SVD(M, r):
    """
        input matrix M, approximating with rank r
    """
    u, s, vh = svd_fast(M)
    s[r:] = 0
    return (u * s).dot(vh)

def SVD_soft(X, l):
    u, s, vh = svd_fast(X)
    s_threshold = np.maximum(0,s-l)
    return (u * s_threshold).dot(vh)
    

def L2_error(s, r):
    '''
        s: a vector
        compute the L2 norm for the vector s[r:] 
    '''
    return np.sqrt(np.mean(s[r:]**2))

def L1_error(s, r):
    '''
        s: a vector
        compute the L2 norm for the vector s[r:] 
    '''
    return np.mean(np.abs(s[r:]))

def error_metric(M, tau, M0, tau_star):
    return np.sum((M - M0)**2) / np.sum(M0**2), np.amax(np.abs(M-M0)) / np.amax(np.abs(M0)), np.abs(tau-tau_star) #/ tau_star

def metric_compute(M, tau, M0, tau_star, Z, metric_name = []):
    error_metrics = {}
    for metric in metric_name:
        if (metric == 'tau'):
            error_metrics[metric] = np.abs(tau-tau_star) / np.mean(np.abs(M0))
        if (metric == 'RMSE_treat_elements'):
            error_metrics[metric] = np.sqrt(np.sum(Z*((M-M0)**2))/np.sum(Z))
        if (metric == 'tau_diff'):
            error_metrics[metric] = tau-tau_star
    return error_metrics

def convex_condition_test(M, Z, r):
    u, s, vh = np.linalg.svd(M, full_matrices = False)
    u = u[:, :r]
    vh = vh[:r, :]

    t1 = np.sum(Z*(u.dot(vh)))
    PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))
    t2 = np.sum(PTperpZ**2)
    t3 = np.linalg.norm(PTperpZ, ord=2)
    return (t1*t3, t2)
    