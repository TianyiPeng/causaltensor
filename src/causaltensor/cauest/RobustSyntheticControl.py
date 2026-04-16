"""
Robust synthetic control (Abadie et al.–style): low-rank denoising on donor units
followed by linear extrapolation for treated trajectories.

Convention: ``O`` and ``Z`` are ``(N, T)`` — units × time — consistent with
``robust_synthetic_control`` usage elsewhere in the package.
"""

import numpy as np


def stagger_pattern_RSC(O, Z, suggest_r=1):
    """
    Variant that builds a rank-``suggest_r`` approximation from never-treated
    units (donors) and projects treated units onto that subspace.

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Outcome panel.
    Z : ndarray, shape (N, T)
        Treatment indicators (1 after adoption).
    suggest_r : int, optional
        Rank for the truncated SVD step.

    Returns
    -------
    Mhat : ndarray, shape (N, T)
        Fitted counterfactual mean.
    tau : float
        Average treatment effect on the treated (ATT), weighted by ``Z``.
    """
    starting_time = O.shape[1] - np.sum(Z, axis=1).astype(int)
    donor_units = np.arange(O.shape[0])[starting_time == O.shape[1]]
    if len(donor_units) == 0:
        raise ValueError("No donor units (never-treated rows) for stagger_pattern_RSC.")

    M = O[donor_units, :]

    u, s, vh = np.linalg.svd(M, full_matrices=False)
    r = int(suggest_r)
    if r < 1 or r > len(s):
        raise ValueError(
            f"suggest_r must satisfy 1 <= suggest_r <= min(N_donor, T); got suggest_r={suggest_r!r} with {len(s)} singular values."
        )
    Mnew = (u[:, :r] * s[:r]).dot(vh[:r, :])
    Mhat = np.zeros_like(O)
    Mhat[donor_units, :] = Mnew

    for i in range(O.shape[0]):
        start = starting_time[i]
        if start == O.shape[1]:
            continue
        coef = np.linalg.pinv(Mnew[:, :start].T).dot(O[i, :start].T)
        Mhat[i, :] = Mnew.T.dot(coef)

    tau = np.sum(Z * (O - Mhat)) / np.sum(Z)
    return Mhat, tau


def robust_synthetic_control(O, Z, suggest_r=-1):
    """
    Robust synthetic control with optional cross-validation over the SVD rank.

    Donors are units that are never treated. A rank-``r`` approximation of the
    donor panel is built; treated units are fit by regression on pre-treatment
    columns. If ``suggest_r == -1``, ``r`` is chosen by minimizing pre-period
    prediction error on treated units over a grid, subject to a cumulative
    singular-value energy cutoff.

    Parameters
    ----------
    O : ndarray, shape (N, T)
        Outcome panel (units × time).
    Z : ndarray, shape (N, T)
        Treatment matrix; entries 1 in post-treatment windows for treated units.
    suggest_r : int, optional
        Fixed rank if ``>= 1``. Use ``-1`` (default) to select ``r`` by CV.

    Returns
    -------
    Mhat : ndarray, shape (N, T)
        Fitted counterfactuals.
    tau : float
        ATT: ``sum(Z * (O - Mhat)) / sum(Z)``.

    Raises
    ------
    ValueError
        If there are no treated units, treatment starts at the first period
        (no pre-period), singular values are empty, or ``suggest_r`` is invalid.
    """
    treat_units = np.where(np.any(Z == 1, axis=1))[0]
    treat_set = set(treat_units.tolist())

    if len(treat_units) == 0:
        raise ValueError("No treated units found in Z (need at least one row with some Z==1).")

    treated_sums = np.sum(Z[treat_units, :], axis=1).astype(int)
    starting_time = O.shape[1] - int(np.min(treated_sums))

    if starting_time == 0:
        raise ValueError("Treatment starts at t=0 for at least one treated unit; need a positive pre-treatment window.")

    donor_units = [i for i in range(O.shape[0]) if i not in treat_set]

    if len(donor_units) == 0:
        raise ValueError("No donor (never-treated) units available.")

    M = O[donor_units, :]

    u, s, vh = np.linalg.svd(M, full_matrices=False)
    if len(s) == 0:
        raise ValueError("SVD produced no singular values (empty donor panel?).")

    def recover(r, start, end):
        r = int(min(max(r, 1), len(s)))
        Mnew = (u[:, :r] * s[:r]).dot(vh[:r, :])
        Mhat = np.zeros_like(O)
        Mhat[donor_units, :] = Mnew

        Mminus = Mnew[:, :start]
        for i in treat_units:
            coef = np.linalg.pinv(Mminus.T).dot(O[i, :start].T)
            Mhat[i, :] = Mnew.T.dot(coef)

        mse = np.sum((Mhat - O)[treat_units, start:end] ** 2)
        return mse, Mhat

    if suggest_r == -1:
        energy = float(np.sum(s))
        if energy <= 0:
            raise ValueError("Sum of singular values is zero; cannot run rank CV.")

        valid_start = int(starting_time / 2 + 0.5)

        opt_mse = np.inf
        opt_r = min(2, len(s))

        # Cross-validation over r: minimize pre-period MSE on treated units.
        for r in range(1, len(s)):
            if (np.sum(s[r - 1 :]) / energy) <= 0.03:
                break
            mse, _ = recover(r, valid_start, starting_time)
            if mse < opt_mse:
                opt_mse = mse
                opt_r = r
    else:
        opt_r = int(suggest_r)
        if opt_r < 1 or opt_r > len(s):
            raise ValueError(
                f"suggest_r must be in [1, {len(s)}] or -1 for CV; got {suggest_r!r}."
            )

    _, mhat = recover(opt_r, starting_time, O.shape[1])

    z_at = np.zeros_like(O)
    z_at[treat_units, starting_time:] = 1
    tau = np.sum(z_at * (O - mhat)) / np.sum(z_at)
    return mhat, tau
