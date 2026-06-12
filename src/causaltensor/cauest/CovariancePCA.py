"""
Covariance-based matrix completion with a PCA-style factor structure
(Xiong & Pelger, 2019).

Fitting matches MC-NNM: effective mask ``Omega_fit = (1 - Z) ⊙ Omega_all`` with
``Omega_all = 1`` if no extra missingness mask is passed.
:class:`CovariancePCAPanelSolver` defaults to two-way fixed effects on the
control mask (like :class:`MCNNMPanelSolver`); :func:`covariance_PCA` defaults
to ``fixed_effects='none'`` for the legacy low-rank-only fit.
Treatment effect: ``tau = sum(Z * (O - M)) / sum(Z)``.
"""

import warnings

import numpy as np

from causaltensor.cauest.panel_solver import FixedEffectPanelSolver, PanelSolver
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
        If fewer than ``m`` observed positions in ``Ω`` or ``m < 1``.
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
        select_pos = np.random.choice(pos, size=m, replace=False)
        new_O_1 = np.zeros(len(O_1))
        new_O_1[select_pos] = 1
        new_O = np.reshape(new_O_1, Ω.shape)
        O_list.append(new_O)
    return O_list


def _rank_cap_from_energy_tail(s, energy_tail_tol):
    """Largest candidate rank scanned when tail mass of singular values drops."""
    energy = float(np.sum(s))
    if energy <= 0:
        raise ValueError("Sum of singular values is zero.")
    r_cap = len(s)
    for rk in range(1, len(s)):
        if (float(np.sum(s[rk - 1 :])) / energy) <= float(energy_tail_tol):
            r_cap = rk
            break
    return max(1, r_cap)


def _derived_rng_seed(seed: int, offset: int) -> int:
    """Deterministic nonnegative seed mixer for replicated CV masks."""
    return int((int(seed) + int(offset) * 1_000_003) & 0x7FFFFFFF)


def _rank_from_spectral_energy(s, spectrum_cut, energy_tail_tol):
    """
    Discrete rank like :meth:`DCPanelSolver.DC_PR_auto_rank`: count of prefixes
    whose cumulative squared-singular-value ratio stays below ``1 - spectrum_cut``,
    applied to covariance singular values ``s``, then cap by the covariance
    tail-scan bound.
    """
    stot_sq = float(np.sum(s**2))
    if stot_sq <= 0:
        raise ValueError("Squared singular-value sum is zero on covariance.")
    ratios = np.cumsum(s**2) / stot_sq
    r_pick = int(np.sum(ratios <= 1.0 - float(spectrum_cut)))
    r_pick = max(1, min(r_pick, len(s)))
    r_cap = _rank_cap_from_energy_tail(s, energy_tail_tol)
    return min(r_pick, r_cap)


def covariance_PCA(
    O,
    Z,
    Omega=None,
    suggest_r=-1,
    return_U=False,
    *,
    rank_selection="spectral",
    spectrum_cut=0.002,
    cv_folds=5,
    cv_replicates=1,
    rank_penalty_lambda=0.0,
    energy_tail_tol=1e-3,
    seed=None,
    fixed_effects="none",
):
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
        Fixed rank if ``>= 1``. Use ``-1`` (default) for automatic rank.
    return_U : bool, optional
        If True, also return the left factor ``U``.
    rank_selection : {'spectral', 'cv'}, optional
        How to choose rank when ``suggest_r == -1``:

        - ``'spectral'`` (default) — deterministic rank from covariance singular
          values (``spectrum_cut``, ``energy_tail_tol``); ignores ``seed``.
        - ``'cv'`` — random-mask cross-validation using ``seed``, ``cv_folds``,
          ``cv_replicates``, ``rank_penalty_lambda``.
    spectrum_cut : float, optional
        Energy threshold when ``rank_selection=='spectral'``: retain prefixes with
        cumulative ``sum(s^2) / sum(s^2) <= 1 - spectrum_cut`` (default ``0.002``).
    cv_folds : int, optional
        Number of random holdout masks per CV replicate when
        ``rank_selection=='cv'``.
    cv_replicates : int, optional
        Independent mask draws averaged when ``rank_selection=='cv'`` (each
        replicate evaluates all candidate ranks using one shared mask suite).
        Default ``1``.
    rank_penalty_lambda : float, optional
        If positive, minimise ``average_CV_MSE + lambda * r`` instead of CV MSE
        alone when ``rank_selection=='cv'``.
    energy_tail_tol : float, optional
        Relative tail cutoff on covariance singular values ``s``: stop scanning CV
        candidate ranks when ``sum(s[r-1:]) / sum(s)`` drops below this, and clip
        ``rank_selection=='spectral'`` to ranks no larger than allowed by this rule.
    seed : int or None, optional
        If ``int``, temporarily seeds ``numpy.random`` only while constructing
        CV masks (per replicate offsets when ``cv_replicates>1``), then restores.
    fixed_effects : {'two-way', 'none'}, optional
        ``'none'`` (default) — low-rank factorisation of ``O`` on controls only.
        ``'two-way'`` — subtract interactive fixed effects on ``Omega_fit`` via
        alternating projections, fit low rank on residuals, then add effects
        back .

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
        Invalid masks, empty support, unknown ``rank_selection``, or bad ``suggest_r``.
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

    fe_mode = str(fixed_effects).strip().lower()
    if fe_mode not in ("none", "two-way"):
        raise ValueError("fixed_effects must be 'none' or 'two-way'; " f"got {fixed_effects!r}.")

    fe_row = np.zeros((O.shape[0], 1))
    fe_col = np.zeros((O.shape[1], 1))
    O_fit = O
    if fe_mode == "two-way":
        fe_solver = FixedEffectPanelSolver(
            fixed_effects="two-way",
            Omega=omega_fit.astype(float),
        )
        O_fit, fe_row, fe_col = fe_solver.demean(O)

    n_obs = int(np.sum(omega_fit))
    O_ob = O_fit * omega_fit
    denom = omega_fit.dot(omega_fit.T)
    denom = np.where(np.abs(denom) < 1e-15, 1.0, denom)
    A = O_ob.dot(O_ob.T) / denom
    A = (A + A.T) * 0.5
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

        mse = np.sum(((omega_fit - Ω_train) * (M - O_fit)) ** 2)
        return mse, M, U

    if suggest_r == -1:
        energy_tail_tol = float(energy_tail_tol)
        mode = str(rank_selection).strip().lower()
        if mode == "spectral":
            sc = float(spectrum_cut)
            if not 0 < sc < 1:
                raise ValueError(f"spectrum_cut must lie in (0, 1); got {spectrum_cut}.")
            opt_r = _rank_from_spectral_energy(s, sc, energy_tail_tol)
        elif mode == "cv":
            cv_folds = int(cv_folds)
            cv_replicates = max(1, int(cv_replicates))
            if cv_folds < 1:
                raise ValueError(f"cv_folds must be >= 1; got {cv_folds}.")
            lam = float(rank_penalty_lambda)
            if lam < 0:
                raise ValueError(f"rank_penalty_lambda must be >= 0; got {rank_penalty_lambda}.")

            p = float(np.sum(omega_fit)) / np.size(omega_fit)
            m_draw = int(np.sum(omega_fit) * p) + 1
            m_draw = min(m_draw, n_obs)

            energy_mass = float(np.sum(s))
            if energy_mass <= 0:
                raise ValueError("Sum of singular values is zero.")

            opt_score = np.inf
            opt_r = 1

            mask_suites = []
            for rep in range(cv_replicates):
                prev_state = None
                if seed is not None:
                    prev_state = np.random.get_state()
                    np.random.seed(_derived_rng_seed(int(seed), rep))
                try:
                    mask_suites.append(random_subset(omega_fit, cv_folds, m_draw))
                finally:
                    if prev_state is not None:
                        np.random.set_state(prev_state)

            for r in range(1, len(s)):
                if (float(np.sum(s[r - 1 :])) / energy_mass) <= energy_tail_tol:
                    break
                rep_scores = []
                for Ω_list in mask_suites:
                    train_mse = []
                    for i in range(cv_folds):
                        mse_fold, _, _ = recover(O_fit * Ω_list[i], Ω_list[i], r)
                        train_mse.append(mse_fold)
                    rep_scores.append(float(np.mean(train_mse)))
                mse_mean = float(np.mean(rep_scores))
                score = mse_mean + lam * float(r)
                if score < opt_score:
                    opt_score = score
                    opt_r = r
        else:
            raise ValueError(
                "rank_selection must be 'cv' or 'spectral'; "
                f"got {rank_selection!r}."
            )
    else:
        opt_r = int(suggest_r)
        if opt_r < 1 or opt_r > len(s):
            raise ValueError(
                f"suggest_r must be in [1, {len(s)}] or -1 for CV; got {suggest_r!r}."
            )

    _, M_resid, U = recover(O_ob, omega_fit, opt_r)
    M = M_resid + fe_row + fe_col.T

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
    ``(1 - Z) ⊙ Omega``. Rank ``r`` is supplied or chosen automatically
    (default: **spectral**, seed-stable).

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Observed outcome panel (units × time).
    Z : ndarray, shape (N, T)
        Binary treatment mask (1 = treated).
    Omega : ndarray, shape (N, T), optional
        Extra observation mask (1 = data present). Defaults to all ones.
    suggest_r : int, optional
        Fixed rank if ``>= 1``; use ``-1`` for automatic rank (via
        :attr:`rank_selection`).
    seed : int or None, optional
        Passed through when ``rank_selection=='cv'`` with ``suggest_r == -1``.
        Default ``2``. ``None`` uses the legacy global RNG for mask draws.
    rank_selection : {'spectral', 'cv'}, optional
        Automatic rank strategy when ``suggest_r==-1``. Default ``'spectral'``;
        use ``'cv'`` for mask-based cross-validation (see :func:`covariance_PCA`).
    spectrum_cut : float, optional
        Squared-energy threshold for ``rank_selection=='spectral'``.
    cv_folds : int, optional
        Masks per CV replicate when ``rank_selection=='cv'``.
    cv_replicates : int, optional
        Replicates averaged for CV rank selection when ``rank_selection=='cv'``.
    rank_penalty_lambda : float, optional
        Penalty on ``r`` added to averaged CV loss when ``rank_selection=='cv'``.
    energy_tail_tol : float, optional
        Covariance spectral tail cutoff (CV scan + spectral clipping).
    fixed_effects : {'two-way', 'none'}, optional
        Whether to subtract two-way FE on ``(1-Z) ⊙ Omega`` before the
        low-rank covariance step. Default ``'two-way'`` aligns with
        :class:`MCNNMPanelSolver`.

    Notes
    -----
    The default spectral rule matches a DC-PR-style cumulative energy cut on the
    covariance spectrum and is deterministic in the data (ignores CV ``seed``).
    For stochastic mask CV comparison, pass ``rank_selection='cv'`` and optionally
    increase ``cv_folds`` / ``cv_replicates``.

    Examples
    --------
    >>> solver = CovariancePCAPanelSolver(O, Z)
    >>> result = solver.fit()
    >>> result.tau        # ATT estimate
    >>> result.baseline   # low-rank counterfactual panel
    """

    def __init__(
        self,
        O,
        Z,
        Omega=None,
        suggest_r=-1,
        seed=2,
        *,
        rank_selection="spectral",
        spectrum_cut=0.002,
        cv_folds=5,
        cv_replicates=1,
        rank_penalty_lambda=0.0,
        energy_tail_tol=1e-3,
        fixed_effects="two-way",
    ):
        self.O = np.asarray(O, dtype=float)
        self.Z = np.asarray(Z, dtype=float)
        self.Omega = Omega
        self.suggest_r = suggest_r
        self.seed = seed
        self.rank_selection = str(rank_selection).strip().lower()
        self.spectrum_cut = float(spectrum_cut)
        self.cv_folds = int(cv_folds)
        self.cv_replicates = int(cv_replicates)
        self.rank_penalty_lambda = float(rank_penalty_lambda)
        self.energy_tail_tol = float(energy_tail_tol)
        self.fixed_effects = str(fixed_effects).strip().lower()

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
            self.O,
            self.Z,
            Omega=self.Omega,
            suggest_r=self.suggest_r,
            return_U=True,
            rank_selection=self.rank_selection,
            spectrum_cut=self.spectrum_cut,
            cv_folds=self.cv_folds,
            cv_replicates=self.cv_replicates,
            rank_penalty_lambda=self.rank_penalty_lambda,
            energy_tail_tol=self.energy_tail_tol,
            seed=self.seed,
            fixed_effects=self.fixed_effects,
        )
        res = CovariancePCAResult(baseline=M, tau=tau, U=U)
        res.O = self.O
        res.Z = self.Z
        return res
