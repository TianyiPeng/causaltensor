"""
Linear algebra utilities for panel estimators.

Centralises SVD helpers, soft-thresholding, non-negative decomposition,
and the 3-D treatment-mask reshape that are shared across all solver classes.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Treatment-mask helpers
# ---------------------------------------------------------------------------

def transform_to_3D(Z) -> np.ndarray:
    """
    Normalise a treatment mask to shape (N, T, K).

    Parameters
    ----------
    Z : ndarray (N, T) or (N, T, K), or list of ndarray (N, T)
        Single or multiple treatment masks.

    Returns
    -------
    ndarray, shape (N, T, K), dtype float
    """
    if isinstance(Z, list):
        Z = np.stack(Z, axis=2)
    elif Z.ndim == 2:
        Z = Z.reshape(Z.shape[0], Z.shape[1], 1)
    return Z.astype(float)


# ---------------------------------------------------------------------------
# Fast SVD
# ---------------------------------------------------------------------------

def svd_fast(M: np.ndarray):
    """
    Memory-efficient SVD that transposes M when rows > cols.

    Returns
    -------
    u, s, vh : ndarrays
        Thin SVD factors consistent with ``np.linalg.svd`` convention.
    """
    is_swap = False
    if M.shape[0] > M.shape[1]:
        is_swap = True
        M = M.T

    A = M @ M.T
    u, ss, uh = np.linalg.svd(A, full_matrices=False)
    ss[ss < 1e-7] = 0
    s = np.sqrt(ss)
    sinv = 1.0 / (s + 1e-7 * (s < 1e-7))
    vh = sinv.reshape(M.shape[0], 1) * (uh @ M)

    if is_swap:
        return vh.T, s, u.T
    return u, s, vh


# ---------------------------------------------------------------------------
# Rank-truncated and soft-threshold SVD
# ---------------------------------------------------------------------------

def SVD(M: np.ndarray, r: int) -> np.ndarray:
    """Hard-truncated rank-r approximation of M."""
    u, s, vh = svd_fast(M)
    s[r:] = 0
    return (u * s).dot(vh)


def SVD_soft(X: np.ndarray, l: float) -> np.ndarray:
    """Soft-threshold SVD: singular values shrunk by l toward zero."""
    u, s, vh = svd_fast(X)
    s_thresh = np.maximum(0, s - l)
    return (u * s_thresh).dot(vh)


# ---------------------------------------------------------------------------
# Non-negative SVD variants
# ---------------------------------------------------------------------------

def SVD_non_negative(M: np.ndarray, r: int) -> np.ndarray:
    """Rank-r approximation with negative entries clipped to zero."""
    M_approx = SVD(M, r)
    M_approx[M_approx < 0] = 0
    return M_approx


def SVD_soft_non_negative(X: np.ndarray, l: float) -> np.ndarray:
    """Soft-threshold SVD with negative entries clipped to zero."""
    X_approx = SVD_soft(X, l)
    X_approx[X_approx < 0] = 0
    return X_approx


def nmf_decomposition(
    M: np.ndarray,
    r: int | None = None,
    l: float | None = None,
    init: str = "random",
    max_iter: int = 2000,
    random_state=None,
) -> np.ndarray:
    """Non-Negative Matrix Factorization wrapper (scikit-learn backend)."""
    from sklearn.decomposition import NMF

    M_nonneg = np.maximum(M, 0)
    n_components = r if r is not None else M.shape[1]

    if l is not None:
        model = NMF(
            n_components=n_components, init=init, max_iter=max_iter,
            random_state=random_state, alpha_H=l, alpha_W=l, l1_ratio=1.0,
        )
    else:
        model = NMF(
            n_components=n_components, init=init, max_iter=max_iter,
            random_state=random_state,
        )
    W = model.fit_transform(M_nonneg)
    H = model.components_
    return W @ H


def non_negative_decomposition(
    M: np.ndarray,
    r: int | None = None,
    l: float | None = None,
    method: str = "nnmf",
) -> np.ndarray:
    """
    Low-rank non-negative approximation, dispatching to NMF or SVD variants.

    Parameters
    ----------
    M : ndarray
        Input matrix.
    r : int, optional
        Hard rank constraint.
    l : float, optional
        Soft regularisation strength.
    method : {'nnmf', 'svd'}
        Back-end algorithm.
    """
    if method not in ("nnmf", "svd"):
        raise ValueError(f"method must be 'nnmf' or 'svd', got '{method}'")

    if l is not None:
        return nmf_decomposition(M=M, r=r, l=l) if method == "nnmf" else SVD_soft_non_negative(X=M, l=l)
    if r is not None:
        return nmf_decomposition(M=M, r=r) if method == "nnmf" else SVD_non_negative(M=M, r=r)
    raise ValueError("At least one of r or l must be provided.")


# ---------------------------------------------------------------------------
# Tangent-space projection
# ---------------------------------------------------------------------------

def remove_tangent_space_component(
    u: np.ndarray, vh: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """
    Project Z onto the complement of the tangent space of a rank-r matrix
    with left/right singular vectors u, vh.

    Memory-aware: avoids forming large (N x N) or (T x T) identity matrices
    for very wide or very tall panels.
    """
    if max(Z.shape) > 1e4:
        if Z.shape[0] > Z.shape[1]:
            first_factor = Z - u.dot(u.T.dot(Z))
            second_factor = np.eye(vh.shape[1]) - vh.T.dot(vh)
        else:
            first_factor = np.eye(u.shape[0]) - u.dot(u.T)
            second_factor = Z - (Z.dot(vh.T)).dot(vh)
        return first_factor.dot(second_factor)
    return (np.eye(u.shape[0]) - u.dot(u.T)).dot(Z).dot(
        np.eye(vh.shape[1]) - vh.T.dot(vh)
    )
