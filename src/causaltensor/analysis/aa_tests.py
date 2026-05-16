"""
A/A test reports on a single built-in dataset.

Runs :func:`causaltensor.semi_synthetic.run_aa_test` for both baseline types
(control and pre-treatment) and all requested methods/patterns, and saves the
results as ``aa_test_report_<dataset>.csv``.

CLI usage
---------
    python -m causaltensor.analysis.aa_tests basque
    python -m causaltensor.analysis.aa_tests smoking --baseline control --n-trials 50
"""

from __future__ import annotations

import argparse
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import run_aa_test
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

_BASELINE_TYPES: Tuple[str, ...] = ("control", "pre-treatment")


def run_aa_report(
    dataset_name: str,
    baseline_types: Sequence[str] = _BASELINE_TYPES,
    methods: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    n_trials: int = 10,
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run A/A tests for one dataset and one or more baseline types.

    Parameters
    ----------
    dataset_name : str
        Name accepted by ``load_dataset``.
    baseline_types : sequence of str, default ('control', 'pre-treatment')
        Which baseline window(s) to use.
    methods : list of str, optional
        Estimator names. Defaults to all seven methods.
    patterns : list of str, optional
        Treatment patterns to test. Defaults to all four patterns.
    n_trials : int, default 10
        Trials per (baseline_type, pattern, method).
    seed : int, default 0
        Base random seed for :func:`~causaltensor.semi_synthetic.run_aa_test` calls.
        Each baseline type uses a derived seed so runs are reproducible.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    pd.DataFrame
        All trial-level rows concatenated, with an added ``dataset`` column.
    """
    datasets_path = default_raw_datasets_path()
    frames: list[pd.DataFrame] = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

    Y_df, Z_df, _ = load_dataset(dataset_name, datasets_path=datasets_path)
    O, Z = prepare_panel(Y_df, Z_df)

    if Z is None or not np.any(Z):
        logger.warning("No treatment matrix Z for %s.", dataset_name)
        return pd.DataFrame()

    for b_idx, baseline_type in enumerate(baseline_types):
        if verbose:
            print(f"\n-- baseline_type: {baseline_type} --")
        try:
            sub_seed = seed + b_idx * 10_007
            df = run_aa_test(
                O, Z,
                methods=methods,
                patterns=patterns,
                baseline_type=baseline_type,
                n_trials=n_trials,
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
    dataset_name: Optional[str] = None,
    prefix: str = "aa_test_report",
) -> Path:
    """Save A/A report as CSV; uses ``aa_test_report_<dataset>.csv`` when ``dataset_name`` is set."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results" / "aa_tests"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{dataset_name}" if dataset_name else prefix
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)
    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="A/A test report on one built-in dataset.")
    parser.add_argument(
        "dataset",
        help=(
            "Dataset name (required), e.g. basque, smoking — same as load_dataset. "
            "Output: aa_test_report_<dataset>.csv"
        ),
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
        "--seed", type=int, default=0,
        help="Random seed for A/A trials (default: 0).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for CSV (default: analysis/results/aa_tests).",
    )
    args = parser.parse_args(argv)

    df = run_aa_report(
        dataset_name=args.dataset,
        baseline_types=args.baseline,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    csv_path = save_aa_report(df, output_dir=args.out_dir, dataset_name=args.dataset)
    print(f"\nReport saved to {csv_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
