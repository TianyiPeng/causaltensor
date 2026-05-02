"""
A/A test reports on built-in datasets.

Runs :func:`causaltensor.semi_synthetic.run_aa_test` on every built-in
dataset that ships with a treatment matrix ``Z``, for both baseline types
and all requested methods/patterns, and saves the results as a CSV.

CLI usage
---------
    python -m causaltensor.analysis.aa_tests
    python -m causaltensor.analysis.aa_tests california_prop99 basque
    python -m causaltensor.analysis.aa_tests --baseline control --n-trials 50
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from causaltensor.analysis.real_dataset_report import datasets_with_treatment_pattern
from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import run_aa_test
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

_BASELINE_TYPES: Tuple[str, ...] = ("control", "pre-treatment")


def run_aa_report(
    dataset_names: Optional[Iterable[str]] = None,
    baseline_types: Sequence[str] = _BASELINE_TYPES,
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    n_trials: int = 10,
    fpr_threshold: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run A/A tests across multiple datasets and baseline types.

    Parameters
    ----------
    dataset_names : iterable of str, optional
        Dataset names to test. Defaults to all built-in datasets with ``Z``.
    baseline_types : sequence of str, default ('control', 'pre-treatment')
        Which baseline window(s) to use per dataset.
    methods : list of str, optional
        Estimator names. Defaults to all seven methods.
    patterns : list of str, optional
        Treatment patterns to test. Defaults to all four patterns.
    n_trials : int, default 10
        Trials per (dataset, baseline_type, pattern, method).
    fpr_threshold : float, default 0.05
        False-positive threshold: ``|tau_hat| / std(M) > this``.
    seed : int, default 0
        Base random seed for :func:`~causaltensor.semi_synthetic.run_aa_test` calls.
        Each dataset and baseline type uses a derived seed so runs are reproducible
        and independent across conditions.
    verbose : bool, default True
        Print progress per dataset.

    Returns
    -------
    pd.DataFrame
        All trial-level rows concatenated, with an added ``dataset`` column.
    """
    datasets_path = default_raw_datasets_path()
    names = list(dataset_names) if dataset_names is not None else list(datasets_with_treatment_pattern())
    frames = []

    for ds_idx, dataset_name in enumerate(names):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*60}")

        Y_df, Z_df, _ = load_dataset(dataset_name, datasets_path=datasets_path)
        O, Z = prepare_panel(Y_df, Z_df)

        if Z is None or not np.any(Z):
            logger.warning("Skipping %s: no treatment matrix Z.", dataset_name)
            continue

        for b_idx, baseline_type in enumerate(baseline_types):
            if verbose:
                print(f"\n-- baseline_type: {baseline_type} --")
            try:
                sub_seed = seed + ds_idx * 100_003 + b_idx * 10_007
                df = run_aa_test(
                    O, Z,
                    methods=methods,
                    patterns=patterns,
                    baseline_type=baseline_type,
                    n_trials=n_trials,
                    fpr_threshold=fpr_threshold,
                    seed=sub_seed,
                    verbose=verbose,
                )
                df.insert(0, "dataset", dataset_name)
                frames.append(df)
            except Exception as exc:
                logger.warning(
                    "Skipping %s / baseline_type=%s: %s",
                    dataset_name, baseline_type, exc,
                )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_aa_report(
    df: pd.DataFrame,
    output_dir=None,
    prefix: str = "aa_test_report",
) -> Path:
    """Save A/A report DataFrame as CSV; returns the path written."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results" / "aa_tests"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)
    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="A/A test report on built-in datasets.")
    parser.add_argument(
        "datasets", nargs="*", default=None,
        help="Dataset names (default: all datasets with treatment Z).",
    )
    parser.add_argument(
        "--baseline", nargs="+", default=list(_BASELINE_TYPES),
        choices=list(_BASELINE_TYPES),
        help="Baseline type(s) to run (default: both).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Trials per (dataset, baseline, pattern, method).",
    )
    parser.add_argument(
        "--fpr-threshold", type=float, default=0.05,
        help="False-positive threshold: |tau_hat| / std(M) > this (default: 0.05).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for A/A trials (default: 0).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for CSV (default: analysis/results/aa_tests).",
    )
    args = parser.parse_args(argv)

    df = run_aa_report(
        dataset_names=args.datasets if args.datasets else None,
        baseline_types=args.baseline,
        n_trials=args.n_trials,
        fpr_threshold=args.fpr_threshold,
        seed=args.seed,
    )
    csv_path = save_aa_report(df, output_dir=args.out_dir)
    print(f"\nReport saved to {csv_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
