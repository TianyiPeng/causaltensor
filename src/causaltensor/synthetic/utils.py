import numpy as np


def generate_low_rank_M(N, T, rank=3, mean=0.0, scale=1.0, rng=None):
    """
    Generate a random low-rank baseline matrix M = U @ V.T

    Entries of U (N x rank) and V (T x rank) are drawn i.i.d from
    N(mean, scale).

    Parameters
    ----------
    N : int
        Number of units (rows).
    T : int
        Number of time periods (columns).
    rank : int, default 3
        Rank of the generated matrix.
    mean : float, default 0.0
        Mean of the Gaussian factors.
    scale : float, default 1.0
        Standard deviation of the Gaussian factors.
    rng : int or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    M : np.ndarray, shape (N, T)
    """
    rng = np.random.default_rng(rng)
    U = rng.normal(mean, scale, size=(N, rank))
    V = rng.normal(mean, scale, size=(T, rank))
    return U @ V.T


def generate_low_rank_M_nonneg(N, T, rank=3, mean_M=1.0, shape=1.0, scale=2.0, rng=None):
    """
    Generate a random non-negative low-rank baseline matrix M = k * U @ V.T

    Entries of U (N x rank) and V (T x rank) are drawn i.i.d from
    Gamma(shape, scale); M is then rescaled so that mean(M) = mean_M.
    Useful when outcomes must be positive (e.g. sales, prices).

    Parameters
    ----------
    N, T : int
        Panel dimensions.
    rank : int, default 3
        Rank of the generated matrix.
    mean_M : float, default 1.0
        Target mean of the output matrix.
    shape, scale : float
        Gamma distribution parameters.
    rng : int or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    M : np.ndarray, shape (N, T)
    """
    rng = np.random.default_rng(rng)
    U = rng.gamma(shape, scale, size=(N, rank))
    V = rng.gamma(shape, scale, size=(T, rank))
    M = U @ V.T
    return M / np.mean(M) * mean_M


def add_noise(M, noise_scale=1.0, rng=None):
    """
    Add i.i.d Gaussian noise: O = M + E, E ~ N(0, noise_scale).

    Parameters
    ----------
    M : np.ndarray
        Baseline panel.
    noise_scale : float, default 1.0
        Standard deviation of the noise.
    rng : int or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    O : np.ndarray, same shape as M
    """
    rng = np.random.default_rng(rng)
    return M + rng.normal(0.0, noise_scale, size=M.shape)


def add_noise_poisson(M, rng=None):
    """
    Add Poisson noise: O[i,t] ~ Poisson(M[i,t]).
    Requires M to be non-negative (use generate_low_rank_M_nonneg).

    Parameters
    ----------
    M : np.ndarray
        Non-negative rate matrix.
    rng : int or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    O : np.ndarray, same shape as M
    """
    rng = np.random.default_rng(rng)
    return rng.poisson(np.maximum(M, 0))
