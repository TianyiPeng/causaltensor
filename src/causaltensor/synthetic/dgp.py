"""
Synthetic panel data generator.

Always returns a 3-tuple ``(O, Z, tau_true)``.

- When ``treatment_level=None`` (default), no effect is injected and
  ``tau_true=0.0``. Pass ``(O, Z)`` to ``run_experiment``, which sweeps
  over treatment levels internally::

      O, Z, _ = generate(N=30, T=50, treatment_pattern="Block", seed=0)
      df = run_experiment(O, Z, methods=["DID"], treatment_levels=[0.1, 0.2])

- When ``treatment_level`` is set, the effect is injected into ``O`` and
  ``tau_true`` is the known ATT. Pass to ``estimate`` for ground-truth
  evaluation::

      O, Z, tau_true = generate(N=30, T=50, treatment_pattern="Block",
                                treatment_level=0.3, seed=0)
      tau_hat = estimate(O, Z, "MC-NNM_CV")
      print(tau_hat, tau_true)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from causaltensor.semi_synthetic.utils import (
    inject_treatment_centered,
    sample_treatment_parameters,
)
from causaltensor.synthetic.utils import (
    add_noise,
    add_noise_poisson,
    generate_low_rank_M,
    generate_low_rank_M_nonneg,
)
from causaltensor.utils.treatment_patterns import Z_adaptive, Z_block, Z_iid, Z_stagger

VALID_PATTERNS = ["IID", "Block", "Staggered", "Adaptive"]


def generate(
    N: int,
    T: int,
    rank: int = 3,
    noise_scale: float = 1.0,
    noise_type: Literal["normal", "poisson"] = "normal",
    M_type: Literal["normal", "nonneg"] = "normal",
    mean_M: float = 0.0,
    scale_M: float = 1.0,
    treatment_pattern: Optional[str] = "Block",
    treatment_level: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a synthetic panel from a low-rank factor model.

    Parameters
    ----------
    N : int
        Number of units (rows).
    T : int
        Number of time periods (columns).
    rank : int, default 3
        Rank of the latent baseline matrix M.
    noise_scale : float, default 1.0
        Standard deviation of additive Gaussian noise (ignored for Poisson).
    noise_type : {'normal', 'poisson'}, default 'normal'
        Noise distribution. Use ``'poisson'`` with ``M_type='nonneg'`` for
        count data.
    M_type : {'normal', 'nonneg'}, default 'normal'
        Factor distribution for M.

        - ``'normal'``  — Gaussian factors; M can be negative.
        - ``'nonneg'``  — Gamma factors; M is non-negative (suitable for
          counts or prices). ``mean_M`` controls the target mean.
    mean_M : float, default 0.0
        For ``M_type='normal'``: mean of the Gaussian factors.
        For ``M_type='nonneg'``: target mean of M after rescaling.
    scale_M : float, default 1.0
        Standard deviation of the Gaussian factors (``M_type='normal'`` only).
    treatment_pattern : {'IID', 'Block', 'Staggered', 'Adaptive'} or None
        Synthetic treatment pattern. ``None`` returns Z = all zeros.
    treatment_level : float or None, optional
        When provided, a treatment effect of size ``treatment_level * mean(|M|)``
        is injected into ``O`` using :func:`inject_treatment_centered` and the
        function returns a **3-tuple** ``(O, Z, tau_true)`` so you can evaluate
        an estimator against the known ground truth.
        When ``None`` (default), ``O`` is the pure baseline + noise and the
        function returns a **2-tuple** ``(O, Z)``.
    seed : int or None, optional
        Random seed for full reproducibility.

    Returns
    -------
    O : np.ndarray, shape (N, T)
        Observed outcome panel. Includes injected treatment effect when
        ``treatment_level`` is set; otherwise pure baseline + noise.
    Z : np.ndarray, shape (N, T)
        Binary treatment mask.
    tau_true : float
        True ATT. Equals the injected effect when ``treatment_level`` is set,
        or ``0.0`` when no treatment is injected.

    Examples
    --------
    Use with run_experiment (treatment injected later at multiple levels):

    >>> O, Z, _ = generate(30, 50, rank=3, treatment_pattern="Block", seed=0)
    >>> df = run_experiment(O, Z, methods=["DID"], treatment_levels=[0.1, 0.2])

    Use with estimate (single known treatment level, compare to ground truth):

    >>> O, Z, tau_true = generate(30, 50, treatment_pattern="Block",
    ...                           treatment_level=0.2, seed=0)
    >>> tau_hat = estimate(O, Z, "DID")
    >>> print(f"true={tau_true:.4f}  hat={tau_hat:.4f}")

    Non-negative / count panel:

    >>> O, Z, _ = generate(50, 60, M_type="nonneg", noise_type="poisson", mean_M=10, seed=0)

    No treatment (load testing / A/A tests, tau_true=0.0):

    >>> O, Z, tau_true = generate(100, 200, treatment_pattern=None, seed=1)
    >>> assert tau_true == 0.0
    """
    if treatment_pattern is not None and treatment_pattern not in VALID_PATTERNS:
        raise ValueError(
            f"Unknown treatment_pattern '{treatment_pattern}'. "
            f"Valid: {VALID_PATTERNS} or None."
        )

    rng = np.random.default_rng(seed)

    # --- Generate baseline M ---
    if M_type == "nonneg":
        M = generate_low_rank_M_nonneg(N, T, rank=rank, mean_M=max(mean_M, 1.0), rng=rng)
    else:
        M = generate_low_rank_M(N, T, rank=rank, mean=mean_M, scale=scale_M, rng=rng)

    # --- Generate treatment mask ---
    if treatment_pattern is None:
        Z = np.zeros((N, T), dtype=int)
    else:
        m1, m2, lookback_a, duration_b = sample_treatment_parameters(N, T, rng)

        if treatment_pattern == "IID":
            Z = Z_iid(M, p_treat=0.2, rng=rng)
        elif treatment_pattern == "Block":
            Z = Z_block(M, m1=m1, m2=m2, rng=rng)
        elif treatment_pattern == "Staggered":
            Z = Z_stagger(M, m1=m1, min_start=m2, rng=rng)
        elif treatment_pattern == "Adaptive":
            Z = Z_adaptive(M, lookback_a=lookback_a, duration_b=duration_b)

        # Guard: ensure at least one treated cell
        if Z.sum() == 0:
            i = int(rng.integers(0, N))
            Z[i, int(rng.integers(0, T))] = 1

    Z = Z.astype(int)

    # --- Optionally inject treatment effect into the mean surface ---
    tau_true = 0.0
    if treatment_level is not None:
        # inject_treatment_centered works on M (noiseless), returns M + Tmat*Z
        M, tau_true = inject_treatment_centered(
            M, Z, treatment_level=treatment_level, rng=rng
        )

    # --- Add noise (always after treatment injection so rng state is consistent) ---
    if noise_type == "poisson":
        O = add_noise_poisson(M)
    else:
        O = add_noise(M, noise_scale=noise_scale, rng=rng)

    return O, Z, tau_true
