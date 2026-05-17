"""
User-facing A/A test runner.

An A/A test checks estimators on a **fixed baseline** panel ``M`` (true effect zero),
with **independent random synthetic** treatment masks ``Z_syn`` each trial — Monte
Carlo variation comes from assignment geometry, not from many field experiments.

Use :func:`plot_aa_null_figure` (this module) for null histograms, and
:func:`run_empirical_power_grid` / :func:`plot_empirical_power_figure` in
``causaltensor.semi_synthetic.empirical_power`` for empirical critical values
and power curves (TestOps-style workflow without ad-hoc |τ|/std(M) cutoffs).

Quickstart
----------
>>> import numpy as np
>>> from causaltensor.semi_synthetic import run_aa_test
>>> rng = np.random.default_rng(0)
>>> O = rng.normal(100, 10, (20, 40))
>>> Z = np.zeros((20, 40)); Z[0, 20:] = 1
>>> df = run_aa_test(O, Z, methods=["DID", "SDID"], n_trials=20)
>>> df.groupby(["method", "pattern"])["tau_hat"].describe()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
    "SC":              ["Block", "Staggered"],
    "RobustSyntheticControl": ["Block", "Staggered"],
}


def _normalise_methods(
    methods: Optional[List[str]],
    patterns: List[str],
) -> Dict[str, List[str]]:
    if methods is None:
        return DEFAULT_METHODS
    return {m: list(patterns) for m in methods}


def draw_synthetic_z(
    M: np.ndarray,
    pattern_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one random binary treatment mask on the baseline ``M`` (same logic as ``run_aa_test``).
    """
    M = np.asarray(M, dtype=float)
    n, T = M.shape
    m1, m2, lookback_a, duration_b = sample_treatment_parameters(n, T, rng)

    if pattern_name == "IID":
        Z_syn = Z_iid(M, p_treat=0.2, rng=rng)
    elif pattern_name == "Block":
        Z_syn = Z_block(M, m1=m1, m2=m2, rng=rng)
    elif pattern_name == "Staggered":
        Z_syn = Z_stagger(M, m1=m1, min_start=m2, rng=rng)
    elif pattern_name == "Adaptive":
        Z_syn = Z_adaptive(M, lookback_a=lookback_a, duration_b=duration_b, rng=rng)
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    if Z_syn.sum() == 0:
        i = int(rng.integers(0, n))
        Z_syn[i, int(rng.integers(0, max(T, 1)))] = 1
    return Z_syn


def run_aa_test(
    O: np.ndarray,
    Z: np.ndarray,
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    baseline_type: str = "control",
    n_trials: int = 10,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run an A/A test on observed panel data.

    Builds a clean baseline ``M`` where the true treatment effect is zero,
    then repeatedly draws random synthetic ``Z_syn`` and records ``tau_hat``
    from each estimator (no treatment injected into ``M``).

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
    verbose : bool, default True
        Print progress and per-trial results.

    Returns
    -------
    pd.DataFrame
        One row per ``(method, pattern, trial)`` with columns:

        method, pattern, baseline_type, trial, tau_hat, std_M

        ``std_M`` is ``std(M)`` of the baseline (same for all rows in a run);
        useful as a scale reference, not as a formal test statistic.
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

            Z_syn = draw_synthetic_z(M, pattern_name, rng)

            for method_name, valid_patterns in methods_dict.items():
                if pattern_name not in valid_patterns:
                    continue

                tau_hat = get_tau_from_method(method_name, M, Z_syn)

                if verbose:
                    if not np.isnan(tau_hat):
                        print(
                            f"    {method_name} (trial {trial + 1}/{n_trials}): "
                            f"tau_hat={tau_hat:.4f}"
                        )
                    else:
                        print(f"    {method_name}: FAILED")

                results.append(
                    {
                        "method":        method_name,
                        "pattern":       pattern_name,
                        "baseline_type": baseline_type,
                        "trial":         trial,
                        "tau_hat":       tau_hat,
                        "std_M":         std_M,
                    }
                )

        if verbose:
            print()

    df = pd.DataFrame(results)

    if verbose and not df.empty:
        _print_aa_summary(df)

    return df


def _print_aa_summary(df: pd.DataFrame) -> None:
    """Print mean / std of ``tau_hat`` under the null, by method and pattern."""
    summary = (
        df.groupby(["method", "pattern"], sort=True)
        .agg(
            mean_tau=("tau_hat", "mean"),
            std_tau=("tau_hat", "std"),
            mean_abs_tau=("tau_hat", lambda x: x.abs().mean()),
            n_trials=("trial", "count"),
        )
        .reset_index()
    )

    col_widths = {
        "method":  max(len("method"), int(summary["method"].str.len().max() or 6)),
        "pattern": max(len("pattern"), int(summary["pattern"].str.len().max() or 8)),
        "mean":    len("mean(tau)"),
        "sd":      len("sd(|tau|)"),
    }

    header = (
        f"{'method':<{col_widths['method']}}  "
        f"{'pattern':<{col_widths['pattern']}}  "
        f"{'mean(tau)':>{col_widths['mean']}}  "
        f"{'std(tau)':>{col_widths['mean']}}  "
        f"{'mean|tau|':>{col_widths['sd']}}"
    )
    sep = "-" * len(header)

    print("\n=== A/A Summary (null distribution of tau_hat) ===")
    print(sep)
    print(header)
    print(sep)

    prev_method = None
    for _, row in summary.iterrows():
        if row["method"] != prev_method:
            if prev_method is not None:
                print(sep)
            prev_method = row["method"]
        mt = row["mean_tau"]
        st = row["std_tau"]
        mat = row["mean_abs_tau"]
        mt_s = f"{mt:.4f}" if pd.notna(mt) else "n/a"
        st_s = f"{st:.4f}" if pd.notna(st) else "n/a"
        mat_s = f"{mat:.4f}" if pd.notna(mat) else "n/a"
        print(
            f"{row['method']:<{col_widths['method']}}  "
            f"{row['pattern']:<{col_widths['pattern']}}  "
            f"{mt_s:>{col_widths['mean']}}  "
            f"{st_s:>{col_widths['mean']}}  "
            f"{mat_s:>{col_widths['sd']}}"
        )

    print(sep)


def plot_aa_null_figure(
    null_df: pd.DataFrame,
    *,
    bins: int = 20,
    figsize: Tuple[float, float] = (12, 10),
):
    """
    Histogram of ``tau_hat`` under the null for each pattern (subplot) and method (overlaid).

    Expects columns ``method``, ``pattern``, ``tau_hat`` as returned by
    :func:`run_aa_test`. Requires ``matplotlib``.
    """
    import matplotlib.pyplot as plt

    patterns = list(dict.fromkeys(null_df["pattern"].tolist()))
    methods = list(dict.fromkeys(null_df["method"].tolist()))
    n_p = len(patterns)
    fig, axes = plt.subplots(
        n_p, 1, figsize=(figsize[0], max(2.5, 2.8 * n_p)), squeeze=False
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 2)))

    for ax, pat in zip(axes.flat, patterns):
        sub = null_df[null_df["pattern"] == pat]
        for k, meth in enumerate(methods):
            x = sub[sub["method"] == meth]["tau_hat"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                continue
            ax.hist(
                x,
                bins=bins,
                density=True,
                alpha=0.35,
                color=colors[k % len(colors)],
                label=meth,
            )
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(f"Null distribution — pattern = {pat}")
        ax.set_xlabel(r"$\hat\tau$")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        r"A/A: $\hat\tau$ when true effect is 0 (fixed $M$, random $Z_\mathrm{syn}$)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig, axes
