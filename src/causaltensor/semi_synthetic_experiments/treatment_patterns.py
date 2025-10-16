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

def Z_adaptive(M, lookback_a, duration_b):
    """
    Adaptive rule (endogenous): for each unit i, scan time j.
    If M[i,j] is strictly the minimum over the previous 'a' periods
    AND there was no treatment in that lookback window,
    then mark the NEXT 'b' periods as treated.
    """
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
                # treat NEXT b periods
                end = min(T, j + duration_b + 1)
                Z[i, j+1:end] = 1
                j += lookback_a + duration_b
            else:
                j += 1
    return Z
