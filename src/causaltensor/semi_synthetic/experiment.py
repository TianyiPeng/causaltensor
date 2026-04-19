"""
User-facing semi-synthetic experiment runner.

Pass your own panel ``O`` and treatment mask ``Z``; the function builds a
clean baseline matrix ``M``, injects a known treatment effect, runs the
requested estimators, and returns a tidy results DataFrame — no file I/O.

Quickstart
----------
>>> import numpy as np
>>> from causaltensor.semi_synthetic import run_experiment
>>> rng = np.random.default_rng(0)
>>> O = rng.standard_normal((20, 30))
>>> Z = np.zeros((20, 30)); Z[0, 15:] = 1   # one treated unit, block treatment
>>> df = run_experiment(O, Z, methods=["DID", "SDID"], patterns=["Block"], n_trials=5)
>>> df.head()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from causaltensor.semi_synthetic.utils import (
    build_baseline_M,
    inject_treatment_centered,
    print_summary_table,
    sample_treatment_parameters,
)
from causaltensor.utils.treatment_patterns import Z_adaptive, Z_block, Z_iid, Z_stagger
from causaltensor.utils.common import get_tau_from_method

# Valid pattern and method names (mirrors analysis/semi_synthetic.py defaults)
VALID_PATTERNS: List[str] = ["IID", "Block", "Staggered", "Adaptive"]

DEFAULT_METHODS: Dict[str, List[str]] = {
    "DC_PR_auto_rank": ["IID", "Block", "Staggered", "Adaptive"],
    "MC_NNM_CV":       ["IID", "Block", "Staggered", "Adaptive"],
    "CovariancePCA":   ["IID", "Block", "Staggered", "Adaptive"],
    "DID":             ["Block", "Staggered"],
    "SDID":            ["Block", "Staggered"],
    "SC":              ["Block"],
    "RobustSyntheticControl": ["Block"],
}


def _treated_info_from_Z(Z: np.ndarray):
    """Derive (treated_states, treat_start_years) from a numpy treatment mask."""
    treated_mask = Z.any(axis=1)
    treated_states = list(np.where(treated_mask)[0])
    treat_start_years = [int(np.argmax(Z[i, :])) for i in treated_states]
    return treated_states, treat_start_years


def _normalise_methods(
    methods: Optional[Union[List[str], Dict[str, List[str]]]],
    patterns: List[str],
) -> Dict[str, List[str]]:
    """
    Convert the ``methods`` argument to the canonical dict form.

    Parameters
    ----------
    methods : None | list[str] | dict[str, list[str]]
        - None   → use DEFAULT_METHODS (same as analysis/semi_synthetic.py)
        - list   → each method is valid for all requested ``patterns``
        - dict   → used as-is
    patterns : list[str]
        The patterns that will be run (used when methods is a plain list).
    """
    if methods is None:
        return DEFAULT_METHODS
    if isinstance(methods, dict):
        return methods
    return {m: list(patterns) for m in methods}


def run_experiment(
    O: np.ndarray,
    Z: np.ndarray,
    methods: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    patterns: Optional[List[str]] = None,
    baseline_type: str = "control",
    treatment_levels: Optional[List[float]] = None,
    n_trials: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a semi-synthetic experiment on user-supplied panel data.

    A *semi-synthetic* experiment proceeds as follows for each
    ``(treatment_level, pattern, trial)`` combination:

    1. Build a clean baseline matrix ``M`` from ``O`` and ``Z`` using
       ``baseline_type`` (control units, or pre-treatment period).
    2. Generate a *synthetic* treatment mask according to ``pattern``.
    3. Inject a known treatment effect ``tau_star`` into ``M``.
    4. Run each estimator and record ``tau_hat`` and relative error.

    Parameters
    ----------
    O : np.ndarray, shape (n, T)
        Observed outcome panel.
    Z : np.ndarray, shape (n, T)
        Binary treatment mask for the *real* data. Used only to identify
        which rows are treated (and when) so that ``build_baseline_M`` can
        construct the baseline matrix ``M``. A fresh synthetic mask is
        generated for each trial.
    methods : None | list[str] | dict[str, list[str]], optional
        Estimators to evaluate.

        - ``None`` (default) — all seven methods with their default valid
          patterns (same as ``analysis/semi_synthetic.py``):
          DC_PR_auto_rank, MC_NNM_CV, CovariancePCA, DID, SDID, SC,
          RobustSyntheticControl.
        - ``list[str]`` — run those methods against every pattern in
          ``patterns``.
        - ``dict[str, list[str]]`` — explicit mapping of method → valid
          patterns (same format as the internal ``analysis`` module).
    patterns : list[str] or None, optional
        Subset of ``['IID', 'Block', 'Staggered', 'Adaptive']`` to test.
        ``None`` (default) runs all four.
    baseline_type : {'control', 'pre-treatment'}, default 'control'
        How to build the baseline matrix ``M``:

        - ``'control'``       — use never-treated rows of ``O``.
        - ``'pre-treatment'`` — use the pre-treatment columns of ``O``
          (requires at least one treated unit with start index > 0).
    treatment_levels : list[float], default [0.2]
        Fraction of ``mean(M)`` injected as ``tau_star``.  Multiple values
        produce one block of results per level.
    n_trials : int, default 10
        Number of independent trials per ``(pattern, treatment_level)``
        combination.  Each trial uses a different random seed.
    verbose : bool, default True
        Print progress and per-trial results.

    Returns
    -------
    pd.DataFrame
        One row per ``(method, pattern, treatment_level, trial)`` with
        columns:

        ============== =====================================================
        method         estimator name
        pattern        synthetic treatment pattern used in this trial
        treatment_level fraction of mean(M) used as tau_star
        trial          0-based trial index
        tau_star       ground-truth injected effect
        tau_hat        estimated effect (NaN if estimator failed)
        error          |tau_star - tau_hat| / |tau_star| (NaN on failure)
        ============== =====================================================
    """
    if treatment_levels is None:
        treatment_levels = [0.2]

    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if O.shape != Z.shape:
        raise ValueError(
            f"O and Z must have the same shape; got O={O.shape}, Z={Z.shape}."
        )

    patterns = list(patterns) if patterns is not None else VALID_PATTERNS
    unknown = set(patterns) - set(VALID_PATTERNS)
    if unknown:
        raise ValueError(f"Unknown pattern(s): {unknown}. Valid: {VALID_PATTERNS}")

    methods_dict = _normalise_methods(methods, patterns)

    # Derive treatment structure from the observed Z
    treated_states, treat_start_years = _treated_info_from_Z(Z)

    if verbose:
        print(f"Data shape: {O.shape}")
        print(f"Treated states: {treated_states}, Treatment start years: {treat_start_years}")

    # Build baseline matrix M
    M, n, T = build_baseline_M(O, treated_states, treat_start_years, baseline_type)

    if verbose:
        print(f"Baseline matrix M shape: ({n}, {T})")
        print(f"Baseline type: {baseline_type}\n")

    results = []

    for t_level in treatment_levels:
        if verbose:
            print(f"--- Treatment level: {t_level} ---")

        for pattern_name in patterns:
            if verbose:
                print(f"  {pattern_name}:")

            for trial in range(n_trials):
                rng = np.random.default_rng(trial)

                n_local, T_local = M.shape
                m1, m2, lookback_a, duration_b = sample_treatment_parameters(
                    n_local, T_local, rng
                )

                # Generate synthetic treatment pattern for this trial
                if pattern_name == "IID":
                    Z_syn = Z_iid(M, p_treat=0.2, rng=rng)
                elif pattern_name == "Block":
                    Z_syn = Z_block(M, m1=m1, m2=m2, rng=rng)
                elif pattern_name == "Staggered":
                    Z_syn = Z_stagger(M, m1=m1, min_start=m2, rng=rng)
                elif pattern_name == "Adaptive":
                    Z_syn = Z_adaptive(M, lookback_a=lookback_a, duration_b=duration_b)
                else:
                    raise ValueError(f"Unknown pattern: {pattern_name}")

                # Guard: ensure at least one treated cell
                if Z_syn.sum() == 0:
                    i = int(rng.integers(0, n_local))
                    j = int(rng.integers(0, max(1, T_local)))
                    Z_syn[i, j % max(T_local, 1)] = 1

                # Inject known treatment effect
                O_syn, tau_star = inject_treatment_centered(
                    M, Z_syn, treatment_level=t_level, rng=rng
                )

                # Run each method (only against its valid patterns)
                for method_name, valid_patterns in methods_dict.items():
                    if pattern_name not in valid_patterns:
                        if verbose:
                            print(
                                f"    {method_name}: SKIPPED (not valid for {pattern_name})"
                            )
                        continue

                    tau_hat = get_tau_from_method(method_name, O_syn, Z_syn)

                    if not np.isnan(tau_hat):
                        error = abs(tau_star - tau_hat) / abs(tau_star) if tau_star != 0 else np.nan
                        if verbose:
                            print(
                                f"    {method_name} (trial {trial + 1}/{n_trials}): "
                                f"tau_star={tau_star:.4f}, tau_hat={tau_hat:.4f}, "
                                f"error={error:.4f}"
                            )
                    else:
                        error = np.nan
                        if verbose:
                            print(f"    {method_name}: FAILED")

                    results.append(
                        {
                            "method": method_name,
                            "pattern": pattern_name,
                            "treatment_level": t_level,
                            "trial": trial,
                            "tau_star": tau_star,
                            "tau_hat": tau_hat,
                            "error": error,
                        }
                    )

            if verbose:
                print()

    df = pd.DataFrame(results)

    if verbose and not df.empty:
        print_summary_table(df)

    return df
