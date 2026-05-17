import numpy as np
# -----------------------------
# 1) Treatment pattern generators
# -----------------------------

def Z_iid(M, p_treat=0.2, rng=None):
    """
    IID treatment: each cell (i,t) independently treated with prob p_treat.
    Returns a binary mask Z ∈ {0,1}^{n×T}.
    """
    rng = np.random.default_rng(rng)
    n, T = M.shape
    Z = (rng.random((n, T)) < p_treat).astype(int)
    return Z



def Z_block(M, m1, m2, rng=None):
    """Simultaneous adoption: choose m1 random units; treat from time index m2 onward."""
    rng = np.random.default_rng(rng)
    n, T = M.shape
    Z = np.zeros_like(M, dtype=int)
    treat_units = rng.choice(n, size=m1, replace=False)
    Z[treat_units, m2:] = 1
    return Z

def Z_stagger(M, m1, min_start, rng=None):
    """Staggered adoption: choose m1 random units; each gets random start >= min_start."""
    rng = np.random.default_rng(rng)
    n, T = M.shape
    Z = np.zeros_like(M, dtype=int)
    treat_units = rng.choice(n, size=m1, replace=False)
    for i in treat_units:
        j = rng.integers(min_start, T)  # start ∈ [min_start, T-1]
        Z[i, j:] = 1
    return Z

def Z_adaptive(
    M,
    lookback_a,
    duration_b,
    rng=None,
    ap=0.9,
    randomize_duration=True
):
    """
    Adaptive rule (endogenous): for each unit i, scan time j.
    If M[i,j] is strictly the minimum over the previous 'a' periods
    AND there was no treatment in that lookback window,
    then mark the NEXT 'b' periods as treated (with stochastic rules).

    Parameters
    ----------
    M : ndarray
        Baseline panel (used only to evaluate the adaptive condition).
    lookback_a : int
        Length of the pre-window compared against M[i,j].
    duration_b : int
        Max length of a treatment episode; see ``randomize_duration``.
    rng : numpy.random.Generator or compatible seed, optional
        Required when ``adopt_prob < 1`` or ``randomize_duration`` is True.
    """

    rng = np.random.default_rng(rng)

    n, T = M.shape
    Z = np.zeros((n, T), dtype=int)
    for i in range(n):
        j = 0
        while j < T:
            ok = True
            for k in range(1, lookback_a + 1):
                if j - k < 0:
                    ok = False; break
                if Z[i, j - k] == 1:
                    ok = False; break
                if M[i, j] > M[i, j - k]:
                    ok = False; break
            if ok:
                if ap < 1.0 and rng.random() >= ap:
                    j += 1
                    continue
                b_eff = int(duration_b)
                if randomize_duration:
                    b_eff = int(rng.integers(1, int(duration_b) + 1))
                end = min(T, j + b_eff + 1)
                Z[i, j + 1 : end] = 1
                j += lookback_a + b_eff
            else:
                j += 1
    return Z
