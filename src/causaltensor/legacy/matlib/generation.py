import numpy as np

def low_rank_M0_normal(n1=50, n2=50, r = 10, loc = 0, scale = 1): 
    """
        Generate a random rank-r matrix M = U.dot(V.T) with shape (n1xn2), where U's shape is (n1xr) and V's shape is (n2xr).
        Here, entries of U and V are i.i.d Gaussian R.V.s drawn from N(loc, scale) where loc is the mean and scale is standard deviation.
    """
    
    U = np.random.normal(loc = loc, scale = scale, size = (n1, r))
    V = np.random.normal(loc = loc, scale = scale, size = (n2, r))
    M0 = U.dot(V.T)
    return M0
    
def low_rank_M0_Gamma(n1=50, n2=50, r = 10, mean_M = 1, shape = 1, scale = 2): 
    """
        Generate a random rank-r non-negative (n1 x n2) matrix with mean(M) = mean_M

        To do so, 
        (i) Generate U with shape (n1xr) and V with shape (n2xr). The entries of U and V are i.i.d Gamma R.V.s drawn from Gamma(shape, scale). 
        (ii) M0 = k * U.dot(V.T), where k is the scale to control the mean value of M0 such that np.mean(M0) = mean_M
    """
    U = np.random.gamma(shape = shape, scale = scale, size = (n1, r))
    V = np.random.gamma(shape = shape, scale = scale, size = (n2, r))
    M0 = U.dot(V.T)
    M0 = M0 / np.mean(M0) * mean_M
    return M0

def add_noise_normal(M0, noise_std=1):
    E = np.random.normal(loc=0, scale=noise_std, size=M0.shape)
    return M0 + E

def add_noise_Poisson(M0):
    return np.random.poisson(M0)