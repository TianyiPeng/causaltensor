"""
Power analysis pipeline on a built-in dataset: A/A null simulations, empirical
thresholds, Monte Carlo power, plots, and CSVs.

All outputs go under one folder per dataset:

    analysis/results/power_analysis/<dataset_name>/

If you pass more than one ``--baseline``, filenames are prefixed (e.g.
``control_null_trials.csv``) so nothing is overwritten.

CLI
---
    python -m causaltensor.analysis.power_analysis basque
    python -m causaltensor.analysis.power_analysis basque --methods OLS_DID SDID \\
        --patterns Block Staggered
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import (
    DEFAULT_METHODS,
    VALID_PATTERNS,
    plot_aa_null_figure,
    run_aa_test,
)
from causaltensor.semi_synthetic.empirical_power import (
    empirical_critical_abs_tau,
    plot_empirical_power_figure,
    run_empirical_power_grid,
)
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

_BASELINE_TYPES: Tuple[str, ...] = ("control", "pre-treatment")
_RESULTS_SUBDIR = "power_analysis"

# Estimator keys for ``--methods`` (same as semi-synthetic registry).
_METHOD_KEYS: Tuple[str, ...] = tuple(DEFAULT_METHODS.keys())

DEFAULT_RELATIVE_EFFECTS = tuple(float(x) for x in np.linspace(0.0, 0.2, 9))


def default_run_root(dataset_name: str) -> Path:
    return (
        Path(__file__).resolve().parent
        / "results"
        / _RESULTS_SUBDIR
        / dataset_name
    )


def run_power_analysis_for_baseline(
    dataset_name: str,
    O: np.ndarray,
    Z: np.ndarray,
    baseline_type: str,
    *,
    output_dir: Path,
    file_prefix: str = "",
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    n_trials_null: int = 40,
    n_trials_power: int = 120,
    relative_effects: Sequence[float] = DEFAULT_RELATIVE_EFFECTS,
    alpha: float = 0.05,
    seed_null: int = 0,
    seed_power: int = 1,
    verbose: bool = True,
    plot_dpi: int = 120,
) -> Dict[str, Any]:
    """
    Run null A/A, save CSVs, empirical thresholds, power grid, both plots.

    Parameters
    ----------
    output_dir
        Directory for outputs (e.g. .../power_analysis/basque/).
    file_prefix
        Optional stem prefix for files when multiple baselines share one folder
        (e.g. ``"control_"``). Empty for the usual single-baseline names.
    seed_null, seed_power
        Seeds for ``run_aa_test`` and ``run_empirical_power_grid`` respectively.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n--- {dataset_name} | baseline={baseline_type} -> {output_dir} ---")

    df_null = run_aa_test(
        O,
        Z,
        methods=methods,
        patterns=patterns,
        baseline_type=baseline_type,
        n_trials=n_trials_null,
        seed=seed_null,
        verbose=verbose,
    )
    if df_null.empty:
        logger.warning("Empty null dataframe for %s / %s", dataset_name, baseline_type)
        return {"output_dir": output_dir, "df_null": df_null, "paths": {}}

    df_null = df_null.copy()
    df_null.insert(0, "dataset", dataset_name)

    path_null = output_dir / f"{file_prefix}null_trials.csv"
    df_null.to_csv(path_null, index=False)
    logger.info("Wrote %s", path_null)

    thr_df = empirical_critical_abs_tau(df_null, alpha=alpha)
    path_thr = output_dir / f"{file_prefix}empirical_thresholds.csv"
    thr_df.to_csv(path_thr, index=False)
    logger.info("Wrote %s", path_thr)

    _, df_power = run_empirical_power_grid(
        O,
        Z,
        df_null,
        relative_effects,
        baseline_type=baseline_type,
        alpha=alpha,
        n_trials_per_effect=n_trials_power,
        seed=seed_power,
        methods=methods,
        patterns=patterns,
        verbose=verbose,
    )
    path_power = output_dir / f"{file_prefix}empirical_power.csv"
    df_power.to_csv(path_power, index=False)
    logger.info("Wrote %s", path_power)

    fig_null, _ = plot_aa_null_figure(df_null, bins=min(20, max(8, n_trials_null // 2)))
    path_fig_null = output_dir / f"{file_prefix}null_tau_distribution.png"
    fig_null.savefig(path_fig_null, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig_null)
    logger.info("Wrote %s", path_fig_null)

    fig_p, _ = plot_empirical_power_figure(df_power)
    path_fig_p = output_dir / f"{file_prefix}power_curves.png"
    fig_p.savefig(path_fig_p, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig_p)
    logger.info("Wrote %s", path_fig_p)

    return {
        "output_dir": output_dir,
        "df_null": df_null,
        "thresholds_df": thr_df,
        "power_df": df_power,
        "paths": {
            "null_trials": path_null,
            "empirical_thresholds": path_thr,
            "empirical_power": path_power,
            "null_tau_distribution_png": path_fig_null,
            "power_curves_png": path_fig_p,
        },
    }


def run_power_analysis(
    dataset_name: str,
    *,
    baseline_types: Sequence[str] = _BASELINE_TYPES,
    root_out: Optional[Path] = None,
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    n_trials_null: int = 40,
    n_trials_power: int = 120,
    relative_effects: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
    plot_dpi: int = 120,
) -> List[Dict[str, Any]]:
    """
    Full pipeline for each ``baseline_type``.

    Output layout::

        results/power_analysis/<dataset_name>/

    With multiple baselines, filenames are prefixed so outputs do not overwrite.
    Returns one result dict per baseline (see ``run_power_analysis_for_baseline``).
    """
    if root_out is None:
        root_out = default_run_root(dataset_name)
    root_out = Path(root_out)

    rel = relative_effects if relative_effects is not None else DEFAULT_RELATIVE_EFFECTS

    datasets_path = default_raw_datasets_path()
    Y_df, Z_df, _ = load_dataset(dataset_name, datasets_path=datasets_path)
    O, Z = prepare_panel(Y_df, Z_df)

    if Z is None or not np.any(Z):
        logger.warning("No treatment matrix Z for %s.", dataset_name)
        return []

    results: List[Dict[str, Any]] = []
    multi = len(baseline_types) > 1
    for b_idx, baseline_type in enumerate(baseline_types):
        prefix = (
            f"{baseline_type.replace('-', '_')}_" if multi else ""
        )
        try:
            out = run_power_analysis_for_baseline(
                dataset_name,
                O,
                Z,
                baseline_type,
                output_dir=root_out,
                file_prefix=prefix,
                methods=methods,
                patterns=patterns,
                n_trials_null=n_trials_null,
                n_trials_power=n_trials_power,
                relative_effects=rel,
                alpha=alpha,
                seed_null=seed + b_idx * 10_007,
                seed_power=seed + b_idx * 10_007 + 1,
                verbose=verbose,
                plot_dpi=plot_dpi,
            )
            results.append(out)
        except Exception as exc:
            logger.warning(
                "Skipping %s / baseline=%s: %s",
                dataset_name,
                baseline_type,
                exc,
            )
    return results


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="A/A null + empirical power + plots for one built-in dataset."
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g. basque, smoking) for load_dataset.",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=list(_BASELINE_TYPES),
        choices=list(_BASELINE_TYPES),
        help=(
            "Baseline type(s). Multiple values write prefixed files in the same "
            "results/power_analysis/<dataset>/ folder."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        choices=_METHOD_KEYS,
        help=(
            "Estimator keys to run (default: all). Example: --methods OLS_DID SDID SC"
        ),
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        metavar="PATTERN",
        choices=list(VALID_PATTERNS),
        help=(
            "Synthetic treatment assignment patterns (default: all). "
            "Example: --patterns Block Staggered"
        ),
    )
    parser.add_argument(
        "--n-trials-null",
        type=int,
        default=40,
        help="Monte Carlo draws per (pattern, method) for null A/A (default: 40).",
    )
    parser.add_argument(
        "--n-trials-power",
        type=int,
        default=120,
        help="Monte Carlo replications per (delta, pattern, method) for power (default: 120).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target size for empirical |tau| threshold (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0).",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help=(
            "Optional root directory for this dataset "
            "(default: analysis/results/power_analysis/<dataset>/)."
        ),
    )
    parser.add_argument(
        "--rel-effects",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Relative effect grid δ; default: nine values from 0 to 0.2 (see "
            "DEFAULT_RELATIVE_EFFECTS). Example: --rel-effects 0 0.05 0.1 0.2"
        ),
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=120,
        help="PNG resolution (default: 120).",
    )
    args = parser.parse_args(argv)

    root = Path(args.out_root) if args.out_root else default_run_root(args.dataset)
    rel = args.rel_effects if args.rel_effects else None

    outs = run_power_analysis(
        args.dataset,
        baseline_types=args.baseline,
        root_out=root,
        methods=args.methods,
        patterns=args.patterns,
        n_trials_null=args.n_trials_null,
        n_trials_power=args.n_trials_power,
        relative_effects=rel,
        alpha=args.alpha,
        seed=args.seed,
        verbose=True,
        plot_dpi=args.plot_dpi,
    )
    for o in outs:
        print("\nSaved:")
        for k, p in o.get("paths", {}).items():
            print(f"  {k}: {p}")
    return outs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
