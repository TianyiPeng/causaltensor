"""
User-facing A/A test runner.

An A/A test verifies that an estimator returns ~0 when there is no true
treatment effect.  Pass your own panel ``O`` and treatment mask ``Z``; the
function carves out a "clean" baseline window (control units or pre-treatment
period), repeatedly assigns *random* synthetic treatment patterns within it,
runs each estimator on the untouched baseline, and reports how often the
estimated effect is non-negligible (false-positive rate).

Quickstart
----------
>>> import numpy as np
>>> from causaltensor.semi_synthetic import run_aa_test
>>> rng = np.random.default_rng(0)
>>> O = rng.normal(100, 10, (20, 40))
>>> Z = np.zeros((20, 40)); Z[0, 20:] = 1
>>> df = run_aa_test(O, Z, methods=["DID", "SDID"], n_trials=20)
>>> df.groupby(["method", "pattern"])["is_false_positive"].mean()
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from causaltensor.semi_synthetic.utils import (
    build_baseline_M,
    sample_treatment_parameters,
)
from causaltensor.utils.treatment_patterns import Z_adaptive, Z_block, Z_iid, Z_stagger
from causaltensor.utils.common import get_tau_from_method, treated_states_and_starts_from_Z

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


def _normalise_methods(
    methods: Optional[List[str]],
    patterns: List[str],
) -> Dict[str, List[str]]:
    if methods is None:
        return DEFAULT_METHODS
    return {m: list(patterns) for m in methods}


def run_aa_test(
    O: np.ndarray,
    Z: np.ndarray,
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    baseline_type: str = "control",
    n_trials: int = 10,
    seed: int = 0,
    fpr_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run an A/A test on observed panel data.

    Builds a clean baseline ``M`` where the true treatment effect is zero,
    then repeatedly assigns random synthetic treatment masks and checks
    whether each estimator falsely detects a non-zero effect.

    Parameters
    ----------
    O : np.ndarray, shape (n, T)
        Observed outcome panel.
    Z : np.ndarray, shape (n, T)
        Binary treatment mask for the *real* data. Used only to identify
        which rows/columns form the clean baseline.
    methods : list of str, optional
        Estimator names to evaluate. Defaults to all seven methods.
    patterns : list[str] or None, optional
        Subset of ``['IID', 'Block', 'Staggered', 'Adaptive']``.
        ``None`` (default) runs all four.
    baseline_type : {'control', 'pre-treatment'}, default 'control'
        How to carve out the clean baseline:

        - ``'control'``       — use never-treated rows of ``O``.
        - ``'pre-treatment'`` — use pre-treatment columns of ``O``.
    n_trials : int, default 10
        Number of random treatment assignments per ``(pattern, method)``
        combination.
    seed : int, default 0
        Base random seed. Each ``(pattern, trial)`` uses a distinct derived seed.
    fpr_threshold : float, default 0.05
        A trial is a *false positive* when
        ``|tau_hat| / std(M) > fpr_threshold``.
    verbose : bool, default True
        Print progress and per-trial results.

    Returns
    -------
    pd.DataFrame
        One row per ``(method, pattern, trial)`` with columns:

        ================== ================================================
        method             estimator name
        pattern            synthetic treatment pattern used in this trial
        baseline_type      'control' or 'pre-treatment'
        trial              0-based trial index
        tau_hat            estimated effect (NaN if estimator failed)
        std_M              std(M) — used to scale tau_hat
        relative_tau       ``|tau_hat| / std(M)`` (NaN on failure)
        is_false_positive  True when ``relative_tau > fpr_threshold``
        ================== ================================================
    """
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

    treated_states, treat_start_years = treated_states_and_starts_from_Z(Z)

    if verbose:
        print(f"Data shape: {O.shape}")
        print(f"Treated states: {treated_states}, Treatment start years: {treat_start_years}")

    # Build clean baseline M (true tau = 0 within this window)
    M, n, T = build_baseline_M(O, treated_states, treat_start_years, baseline_type)

    std_M = float(np.std(M)) if np.std(M) > 0 else 1.0

    if verbose:
        print(f"Baseline matrix M shape: ({n}, {T})")
        print(f"Baseline type: {baseline_type}, std(M): {std_M:.4f}\n")

    results = []

    for p_idx, pattern_name in enumerate(patterns):
        if verbose:
            print(f"  {pattern_name}:")

        for trial in range(n_trials):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, p_idx, trial])
            )

            m1, m2, lookback_a, duration_b = sample_treatment_parameters(n, T, rng)

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
                i = int(rng.integers(0, n))
                Z_syn[i, int(rng.integers(0, max(T, 1)))] = 1

            for method_name, valid_patterns in methods_dict.items():
                if pattern_name not in valid_patterns:
                    continue

                # Run estimator on pure baseline — no injection, true tau = 0
                tau_hat = get_tau_from_method(method_name, M, Z_syn)

                if not np.isnan(tau_hat):
                    relative_tau = abs(tau_hat) / std_M
                    is_fp = relative_tau > fpr_threshold
                    if verbose:
                        fp_flag = " [FP]" if is_fp else ""
                        print(
                            f"    {method_name} (trial {trial + 1}/{n_trials}): "
                            f"tau_hat={tau_hat:.4f}, rel={relative_tau:.4f}{fp_flag}"
                        )
                else:
                    relative_tau = np.nan
                    is_fp = False
                    if verbose:
                        print(f"    {method_name}: FAILED")

                results.append(
                    {
                        "method":           method_name,
                        "pattern":          pattern_name,
                        "baseline_type":    baseline_type,
                        "trial":            trial,
                        "tau_hat":          tau_hat,
                        "std_M":            std_M,
                        "relative_tau":     relative_tau,
                        "is_false_positive": is_fp,
                    }
                )

        if verbose:
            print()

    df = pd.DataFrame(results)

    if verbose and not df.empty:
        _print_aa_summary(df, fpr_threshold)

    return df


def _print_aa_summary(df: pd.DataFrame, fpr_threshold: float) -> None:
    """Print FPR + mean |tau_hat| grouped by method / pattern."""
    summary = (
        df.groupby(["method", "pattern"])
        .agg(
            fpr=("is_false_positive", "mean"),
            mean_abs_tau=("tau_hat", lambda x: x.abs().mean()),
            std_abs_tau=("tau_hat", lambda x: x.abs().std()),
            n_trials=("trial", "count"),
        )
        .reset_index()
    )

    col_widths = {
        "method":  max(len("method"),  summary["method"].str.len().max()),
        "pattern": max(len("pattern"), summary["pattern"].str.len().max()),
        "fpr":     len("FPR"),
        "tau":     len("mean |tau_hat| +/- std"),
    }

    header = (
        f"{'method':<{col_widths['method']}}  "
        f"{'pattern':<{col_widths['pattern']}}  "
        f"{'FPR':>{col_widths['fpr']}}  "
        f"{'mean |tau_hat| +/- std':>{col_widths['tau']}}"
    )
    sep = "-" * len(header)

    print(f"\n=== A/A Summary (fpr_threshold={fpr_threshold}) ===")
    print(sep)
    print(header)
    print(sep)

    prev_method = None
    for _, row in summary.iterrows():
        if row["method"] != prev_method:
            if prev_method is not None:
                print(sep)
            prev_method = row["method"]
        std_s = f"{row['std_abs_tau']:.4f}" if not np.isnan(row["std_abs_tau"]) else "n/a"
        tau_cell = f"{row['mean_abs_tau']:.4f} +/- {std_s}"
        print(
            f"{row['method']:<{col_widths['method']}}  "
            f"{row['pattern']:<{col_widths['pattern']}}  "
            f"{row['fpr']:>{col_widths['fpr']}.2f}  "
            f"{tau_cell:>{col_widths['tau']}}"
        )

    print(sep)
