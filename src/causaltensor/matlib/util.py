import numpy as np

def non_negative_decomposition(M, r=None, l=None, method='nnmf'):
    """
    Apply a non-negative decomposition with either a soft regularization (l) or a hard rank constraint (r).

    Parameters:
        M (ndarray): Input matrix.
        r (int, optional): Hard rank constraint.
        l (float, optional): Regularization parameter used as alpha in NNMF.
        method (str): Either 'nnmf' or 'svd'.

    Returns:
        M_approx (ndarray): Low-rank approximation of M.
    """
    if l is not None:
        if method == "nnmf":
            M_approx = nmf_decomposition(M=M, r=r, l=l)
        elif method == "svd":
            M_approx = SVD_soft_non_negative(X=M, l=l)
    elif r is not None:
        if method == "nnmf":
            M_approx = nmf_decomposition(M=M, r=r)
        elif method == "svd":
            M_approx = SVD_non_negative(M=M, r=r)

    return M_approx

def nmf_decomposition(M, r=None, l=None, init='random', max_iter=2000, random_state=None):
    """
    Perform Non-Negative Matrix Factorization (NMF) on matrix M with an option to include regularization.

    Parameters:
        M (ndarray): Input matrix.
        r (int, optional): Desired rank (number of components). Used as the number of components if l is provided;
                           if not provided, defaults to the number of columns of M.
        l (float, optional): Regularization parameter (alpha). When provided, it is passed to the NMF algorithm.
        init (str): Initialization method ('random' or 'nndsvd').
        max_iter (int): Maximum number of iterations.
        random_state (int): Seed for reproducibility.

    Returns:
        M_approx (ndarray): Reconstructed matrix.
    """
    from sklearn.decomposition import NMF

    M_nonneg = np.maximum(M, 0)

    n_components = r if r is not None else M.shape[1]

    # l as regularization parameter on H, W and L1 regularization somewhat similar to soft-thresholding
    if l is not None:
        model = NMF(n_components=n_components, init=init, max_iter=max_iter,
                    random_state=random_state, alpha_H=l, alpha_W=l, l1_ratio=1.0)
    elif r is not None:
        # Use the hard rank constraint.
        model = NMF(n_components=n_components, init=init, max_iter=max_iter,
                    random_state=random_state)
    else:
        # default to full factorization
        model = NMF(n_components=M.shape[1], init=init, max_iter=max_iter,
                    random_state=random_state)

    W = model.fit_transform(M_nonneg)
    H = model.components_
    M_approx = W @ H

    return M_approx


def SVD_non_negative(M, r):
    """
    Ensure nonnegative output in the reconstructred M matrix with r rank using 
    M_{ij} = max(M_{ij}, 0) 
    """
    M_approx = SVD(M=M, r=r)
    M_approx[M_approx < 0] = 0
    return M_approx

def SVD_soft_non_negative(X, l):
    X_approx = SVD_soft(X, l)
    X_approx[X_approx < 0] = 0
    return X_approx

def noise_to_signal(X, M, Ω):
    return np.sqrt(np.sum((Ω*X - Ω*M)**2) / np.sum((Ω*M)**2))

def abs_mean(X, M, Ω):
    return np.sum(np.abs((X-M)*Ω)) / np.sum(Ω)

def svd_fast(M):
    is_swap = False
    if M.shape[0] > M.shape[1]:
        is_swap = True
        M = M.T

    A = M @ M.T # this will speed up the calculation when M is asymmetric 
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


def transform_to_3D(Z):
    """
        Z is a list of 2D numpy arrays or a single 2D/3D numpy array
        convert Z to a 3D numpy array with the last dimension being the index of interventions
    """
    if isinstance(Z, list): #if Z is a list of numpy arrays
        Z = np.stack(Z, axis = 2) 
    elif Z.ndim == 2: #if a single Z
        Z = Z.reshape(Z.shape[0], Z.shape[1], 1)
    return Z.astype(float)


def remove_tangent_space_component(u, vh, Z):
    """
        Remove the projection of Z (a single treatment) onto the tangent space of M in memory-aware manner
    """

    # We conduct some checks for extremely wide or extremely long matrices, which may result in OOM errors with
    # naïve operation sequencing.  If BOTH dimensions are extremely large, there may still be an OOM error, but this
    # case is quite rare.
    treatment_matrix_shape = Z.shape
    if max(treatment_matrix_shape) > 1e4:

        if treatment_matrix_shape[0] > treatment_matrix_shape[1]:
            first_factor = (Z - u.dot(u.T.dot(Z)))
            second_factor = np.eye(vh.shape[1]) - vh.T.dot(vh)
        else:
            first_factor = (np.eye(u.shape[0]) - u.dot(u.T))
            second_factor = Z - (Z.dot(vh.T)).dot(vh)

        PTperpZ = first_factor.dot(second_factor)

    else:
        PTperpZ = (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(np.eye(vh.shape[1]) - vh.T.dot(vh))

    return PTperpZ
