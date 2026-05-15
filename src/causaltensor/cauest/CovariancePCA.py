"""
Covariance-based matrix completion with a PCA-style factor structure
(Xiong & Pelger, 2019).

Fitting matches MC-NNM: effective mask ``Omega_fit = (1 - Z) ⊙ Omega_all`` with
``Omega_all = 1`` if no extra missingness mask is passed. Treatment effect:
``tau = sum(Z * (O - M)) / sum(Z)`` (Readme Step 3).
"""

import warnings

import numpy as np

from causaltensor.cauest.panel_solver import PanelSolver
from causaltensor.cauest.result import Result


def random_subset(Ω, K, m):
    """
    Sample ``K`` random binary masks with exactly ``m`` observed entries each.

    Parameters
    ----------
    Ω : ndarray
        Binary observation mask (same shape as outcomes).
    K : int
        Number of mask replicates.
    m : int
        Number of positions set to 1 in each replicate.

    Returns
    -------
    list of ndarray
        Length ``K``, each the same shape as ``Ω``.

    Raises
    ------
    ValueError
        If there are fewer than ``m`` observed positions in ``Ω`` or ``m < 1``.
    """
    O_1 = np.reshape(Ω, -1)
    pos = np.arange(len(O_1))[O_1 == 1]
    if m < 1:
        raise ValueError("m must be >= 1.")
    if m > len(pos):
        raise ValueError(
            f"Cannot sample m={m} positions without replacement; only {len(pos)} observed entries in Ω."
        )
    O_list = []
    for _ in range(K):
        select_pos = np.random.choice(list(pos), m, replace=False)
        new_O_1 = np.zeros(len(O_1))
        new_O_1[select_pos] = 1
        new_O = np.reshape(new_O_1, Ω.shape)
        O_list.append(new_O)
    return O_list


def covariance_PCA(O, Z, Omega=None, suggest_r=-1, return_U=False, seed=None):
    """
    Covariance PCA: low-rank ``M`` from outcomes, fit using only control cells
    ``(1 - Z) ⊙ Omega`` (same spirit as :class:`MCNNMPanelSolver`).

    Parameters
    ----------
    O : ndarray
        Outcome matrix.
    Z : ndarray
        Binary treatment (same shape as ``O``).
    Omega : ndarray, optional
        Extra observation mask (1 = data present). Defaults to all ones.
        Fitting uses ``(1 - Z) * Omega``.
    suggest_r : int, optional
        Fixed rank if ``>= 1``. Use ``-1`` (default) for CV over ``r``.
    return_U : bool, optional
        If True, also return the left factor ``U``.
    seed : int or None, optional
        Random seed passed to :func:`random_subset` when ``suggest_r == -1``.
        Passing an integer makes the CV rank selection deterministic.
        Defaults to ``None`` (use current NumPy random state).

    Returns
    -------
    M : ndarray
        Low-rank reconstruction.
    tau : float
        ``sum(Z * (O - M)) / sum(Z)``.
    U : ndarray, optional
        Returned only if ``return_U`` is True.

    Raises
    ------
    ValueError
        Invalid masks, empty support, or bad ``suggest_r``.
    """
    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # Deprecation guard: the old API passed an observation mask Ω (mostly 1s)
    # as the second positional arg.  Treatment matrices are typically sparse,
    # so a dense Z (>50% ones) with no explicit Omega is a strong signal that
    # the caller is using the old convention.
    if Omega is None and Z.size > 0 and Z.mean() > 0.5:
        warnings.warn(
            "Z has >50% entries equal to 1 and no Omega was passed. "
            "If Z is an observation mask from the old API, note that the "
            "second argument is now the *treatment* matrix (sparse, mostly 0s). "
            "Pass the observation mask as Omega instead: "
            "covariance_PCA(O, Z=treatment, Omega=mask).",
            DeprecationWarning,
            stacklevel=2,
        )

    if Z.shape != O.shape:
        raise ValueError("Z must have the same shape as O.")
    if np.sum(Z) <= 0:
        raise ValueError("Z must have at least one treated entry.")

    if Omega is None:
        Omega_all = np.ones_like(O, dtype=float)
    else:
        Omega_all = np.asarray(Omega, dtype=float)

    omega_fit = (1.0 - Z) * Omega_all
    if np.sum(omega_fit) <= 0:
        raise ValueError("No control observations: (1-Z)*Omega is all zero.")

    n_obs = int(np.sum(omega_fit))
    O_ob = O * omega_fit
    denom = omega_fit.dot(omega_fit.T)
    denom = np.where(np.abs(denom) < 1e-15, 1.0, denom)
    A = O_ob.dot(O_ob.T) / denom
    u, s, _vh = np.linalg.svd(A, full_matrices=False)
    if len(s) == 0:
        raise ValueError("SVD returned no singular values.")

    def recover(O_train, Ω_train, r):
        r = int(min(max(r, 1), len(s)))
        U = u[:, :r] * np.sqrt(O.shape[0])

        col_sum = np.sum(Ω_train, axis=0).reshape((omega_fit.shape[1], 1))
        col_sum = np.maximum(col_sum, 1e-15)
        Y = O_train.T.dot(U) / col_sum

        M = U.dot(Y.T)

        mse = np.sum(((omega_fit - Ω_train) * (M - O)) ** 2)
        return mse, M, U

    if suggest_r == -1:
        K = 2
        p = float(np.sum(omega_fit)) / np.size(omega_fit)
        m = int(np.sum(omega_fit) * p) + 1
        m = min(m, n_obs)
        if seed is not None:
            np.random.seed(seed)
        Ω_list = random_subset(omega_fit, K, m)

        energy = float(np.sum(s))
        if energy <= 0:
            raise ValueError("Sum of singular values is zero.")

        opt_mse = np.inf
        opt_r = 1

        for r in range(1, len(s)):
            if (np.sum(s[r - 1 :]) / energy) <= 1e-3:
                break
            train_mse = []
            for i in range(K):
                mse, _, _ = recover(O * Ω_list[i], Ω_list[i], r)
                train_mse.append(mse)
            mse_mean = float(np.mean(train_mse))
            if mse_mean < opt_mse:
                opt_mse = mse_mean
                opt_r = r
    else:
        opt_r = int(suggest_r)
        if opt_r < 1 or opt_r > len(s):
            raise ValueError(
                f"suggest_r must be in [1, {len(s)}] or -1 for CV; got {suggest_r!r}."
            )

    _, M, U = recover(O_ob, omega_fit, opt_r)

    tau = float(np.sum(Z * (O - M)) / np.sum(Z))

    if return_U:
        return M, tau, U
    return M, tau


class CovariancePCAResult(Result):
    """Result container for :class:`CovariancePCAPanelSolver`."""

    def __init__(self, baseline=None, tau=None, U=None):
        super().__init__(baseline=baseline, tau=tau)
        self.M = baseline  # low-rank reconstruction
        self.U = U         # left factor matrix (N × r)

    def _summary_internals(self):
        lines = []
        if self.U is not None:
            lines.append(f"{'num_factors':<24s}: {self.U.shape[1]}")
            lines.append(f"{'factor_matrix (U) shape':<24s}: {self.U.shape}  (units x factors)")
        return lines


class CovariancePCAPanelSolver(PanelSolver):
    """
    Covariance PCA estimator (Xiong & Pelger, 2019).

    Fits a low-rank counterfactual matrix using only control cells
    ``(1 - Z) ⊙ Omega``. Rank ``r`` is either supplied or chosen by CV.

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Observed outcome panel (units × time).
    Z : ndarray, shape (N, T)
        Binary treatment mask (1 = treated).
    Omega : ndarray, shape (N, T), optional
        Extra observation mask (1 = data present). Defaults to all ones.
    suggest_r : int, optional
        Fixed rank if ``>= 1``; use ``-1`` (default) for CV.
    seed : int or None, optional
        Random seed for the cross-validation rank selection (only used when
        ``suggest_r == -1``). Defaults to ``2`` for reproducible results.
        Pass ``None`` to use the current NumPy random state.

    Notes
    -----
    When ``suggest_r == -1`` the CV uses two random observation-mask splits to
    score each candidate rank. The result is sensitive to the random subsets on
    small panels; ``seed`` pins the subsets for reproducibility.

    Examples
    --------
    >>> solver = CovariancePCAPanelSolver(O, Z)
    >>> result = solver.fit()
    >>> result.tau        # ATT estimate
    >>> result.baseline   # low-rank counterfactual panel
    """

    def __init__(self, O, Z, Omega=None, suggest_r=-1, seed=2):
        self.O = np.asarray(O, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.Omega = Omega
        self.suggest_r = suggest_r
        self.seed = seed

    def fit(self):
        """
        Run the Covariance PCA algorithm.

        Returns
        -------
        CovariancePCAResult
            ``.tau``     — ATT estimate (float)
            ``.baseline`` / ``.M`` — low-rank counterfactual panel (ndarray)
            ``.U``       — left factor matrix (ndarray, N × r)
        """
        M, tau, U = covariance_PCA(
            self.O, self.Z,
            Omega=self.Omega,
            suggest_r=self.suggest_r,
            return_U=True,
            seed=self.seed,
        )
        res = CovariancePCAResult(baseline=M, tau=tau, U=U)
        res.O = self.O
        res.Z = self.Z
        return res
