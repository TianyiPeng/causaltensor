import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import VALID_PATTERNS
from causaltensor.semi_synthetic.experiment import run_experiment
from causaltensor.utils.common import extract_treatment_info_from_Z
from causaltensor.utils.panel import default_raw_datasets_path

logger = logging.getLogger(__name__)


def run_semi_synthetic_experiment(O, treated_states, treat_start_years,
                                   baseline_type='control',
                                   treatment_levels=None,
                                   methods=None,
                                   patterns=None,
                                   n_trials=10,
                                   seed=0,
                                   verbose=True):
    """
    Run semi-synthetic experiments with different treatment patterns, levels, and methods.

    Thin wrapper around :func:`causaltensor.semi_synthetic.run_experiment` that
    accepts the pre-derived ``treated_states`` / ``treat_start_years`` lists
    (the internal format used by ``analysis.semi_synthetic``) instead of a raw
    ``Z`` array.

    Parameters
    ----------
    O : np.ndarray
        Observed panel data (n × T).
    treated_states : list of int
        Row indices of treated units in ``O``.
    treat_start_years : list of int
        Column index of the first treated period for each treated unit
        (same order as ``treated_states``).
    baseline_type : {'control', 'pre-treatment'}, default 'control'
        How to build the baseline matrix ``M``.
    treatment_levels : list of float, optional
        Fraction of mean(|M|) injected as tau_star.
        Defaults to ``[0.2, 0.1, 0.05, 0.01]``.
    methods : None | list[str] | dict[str, list[str]], optional
        Estimators to evaluate. See :func:`~causaltensor.semi_synthetic.run_experiment`.
    patterns : list[str] or None, optional
        Subset of ``VALID_PATTERNS`` to simulate. ``None`` runs all four.
    n_trials : int, default 10
        Trials per (pattern, treatment_level) combination.
    seed : int, default 0
        Passed to :func:`~causaltensor.semi_synthetic.run_experiment`.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: method, pattern, treatment_level, trial, tau_star, tau_hat, error.
    """
    if treatment_levels is None:
        treatment_levels = [0.2, 0.1, 0.05, 0.01]

    # Reconstruct a block-treatment Z from the treated_states / treat_start_years
    # lists so we can delegate to the user-facing run_experiment(O, Z, ...).
    Z_real = np.zeros(O.shape, dtype=float)
    for state, start in zip(treated_states, treat_start_years):
        Z_real[state, start:] = 1.0

    return run_experiment(
        O, Z_real,
        methods=methods,
        patterns=patterns,
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        n_trials=n_trials,
        seed=seed,
        verbose=verbose,
    )


def _aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summaries per (method, pattern, treatment_level), on finite estimates."""
    rows = []
    for (method, pattern, level), g in results_df.groupby(
        ["method", "pattern", "treatment_level"], sort=True
    ):
        ok = g[np.isfinite(g["tau_hat"]) & np.isfinite(g["tau_star"])]
        diff = ok["tau_hat"] - ok["tau_star"]
        abs_diff = diff.abs()
        rel_err = ok["error"].dropna()

        n_ok, n_trials = len(ok), len(g)
        rows.append({
            "method": method,
            "pattern": pattern,
            "treatment_level": level,
            "mean_error": rel_err.mean(),
            "std_error": rel_err.std(),
            "signed_bias": diff.mean(),
            "mae": abs_diff.mean(),
            "rmse": float(np.sqrt(np.mean(np.square(diff)))) if n_ok else np.nan,
            "abs_error_p50": abs_diff.quantile(0.5),
            "abs_error_p90": abs_diff.quantile(0.9),
            "n_ok": n_ok,
            "n_trials": n_trials,
            "success_rate": n_ok / n_trials if n_trials else np.nan,
        })
    return pd.DataFrame(rows)


def run_experiments(
    O,
    treated_states,
    treat_start_years,
    treatment_levels,
    baseline_type,
    dataset_name: str,
    methods=None,
    patterns=None,
    n_trials=10,
    seed=0,
):
    """
    Run experiments for a given baseline type, print a summary, and save results.

    Parameters
    ----------
    O : np.ndarray
        Observed panel data.
    treated_states : list of int
        Row indices of treated units.
    treat_start_years : list of int
        First treatment column index per treated unit.
    treatment_levels : list of float
        Treatment levels to test.
    baseline_type : str
        'control' or 'pre-treatment'.
    dataset_name : str
        Dataset key (same as :func:`load_dataset`); used for results subdirectory and plots.
    methods : optional
        Passed through to :func:`run_semi_synthetic_experiment`.
    patterns : list[str] or None, optional
        Synthetic patterns to run; ``None`` means all ``VALID_PATTERNS``.
    n_trials : int, default 10
        Trials per combination.
    seed : int, default 0
        Random seed for :func:`run_semi_synthetic_experiment`.
    """
    results_df = run_semi_synthetic_experiment(
        O=O,
        treated_states=treated_states,
        treat_start_years=treat_start_years,
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        methods=methods,
        patterns=patterns,
        n_trials=n_trials,
        seed=seed,
        verbose=True,
    )

    aggregated = _aggregate_results(results_df)

    # Save detailed results (all trials) to CSV
    _base = Path(__file__).resolve().parent / "results" / "semi_synthetic_data"
    results_dir = _base / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"semi_synthetic_{baseline_type}_results_detailed.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Detailed results (all trials) saved to: {output_path}")

    output_path_agg = results_dir / f"semi_synthetic_{baseline_type}_results_aggregated.csv"
    aggregated.to_csv(output_path_agg, index=False)
    print(f"Aggregated results (mean ± std) saved to: {output_path_agg}")

    return results_df, aggregated


def main(
    dataset_name="smoking",
    methods: Optional[List[str]] = None,
    baseline_type: str = "control",
    patterns: Optional[List[str]] = None,
):
    """
    Load a built-in dataset and run the full semi-synthetic comparison study.

    Runs estimators across synthetic treatment patterns, multiple treatment levels,
    and a chosen baseline type.

    Parameters
    ----------
    dataset_name : str, default "smoking"
        Any name accepted by :func:`causaltensor.datasets.load_dataset`.
    methods : None or list[str], optional
        ``None`` runs all default estimators (see ``DEFAULT_METHODS``).
        A list runs only those methods on every pattern.
    baseline_type : {'control', 'pre-treatment'}, default 'control'
        Which baseline ``M`` to use. ``pre-treatment`` requires observed treatment
        in ``Z`` (non-empty ``treat_start_years``).
    patterns : None or list[str], optional
        Subset of ``VALID_PATTERNS``. ``None`` defaults to ``['Block', 'Staggered']``.
    """
    print(f"Loading dataset: {dataset_name}")
    Y_df, Z_df, _X_df = load_dataset(dataset_name, datasets_path=default_raw_datasets_path())

    O = Y_df.values

    treated_states, treat_start_years = extract_treatment_info_from_Z(Y_df, Z_df)

    treated_entities = Y_df.index[treated_states].tolist() if treated_states else []
    treat_start_labels = Y_df.columns[treat_start_years].tolist() if treat_start_years else []

    if patterns is None:
        patterns = ["Block", "Staggered"]

    n_trials = 100
    treatment_levels = [0.2, 0.1, 0.05]

    print("="*80)
    print("Semi-Synthetic Causal Inference Experiments - Comparison Study")
    print("="*80)
    print(f"Dataset: {dataset_name} (shape: {O.shape})")
    print(f"Treated entities: {treated_entities} (indices: {treated_states})")
    print(f"Treatment start: {treat_start_labels} (indices: {treat_start_years})")
    print(f"Treatment levels: {treatment_levels}")
    print(f"Number of trials per method/pattern: {n_trials}")
    print(f"Baseline type: {baseline_type}")
    print(f"Methods: {methods if methods is not None else 'all (DEFAULT_METHODS)'}")
    print(f"Treatment patterns: {patterns}")
    print("="*80)
    print()

    output = {}

    if baseline_type == "control":
        results_control, agg_control = run_experiments(
            O,
            treated_states,
            treat_start_years,
            treatment_levels,
            "control",
            dataset_name,
            methods,
            patterns,
            n_trials,
            seed=0,
        )
        output["control"] = {"detailed": results_control, "aggregated": agg_control}
    elif baseline_type == "pre-treatment":
        if not treat_start_years:
            print(
                "\nSkipping: pre-treatment baseline requires at least one treated unit "
                "with treatment start after the first period."
            )
        else:
            print("\n" + "="*80 + "\n")
            results_pretreatment, agg_pretreatment = run_experiments(
                O,
                treated_states,
                treat_start_years,
                treatment_levels,
                "pre-treatment",
                dataset_name,
                methods,
                patterns,
                n_trials,
            )
            output["pre-treatment"] = {
                "detailed": results_pretreatment,
                "aggregated": agg_pretreatment,
            }
    else:
        raise ValueError(f"baseline_type must be 'control' or 'pre-treatment', got {baseline_type!r}")
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

    return output


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Semi-synthetic estimator comparison.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="smoking",
        help="Dataset name (default: smoking).",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help='Comma-separated method keys (DEFAULT_METHODS), or "all" (default) for every entry.',
    )
    parser.add_argument(
        "--baseline-type",
        choices=("control", "pre-treatment"),
        default="control",
        help="How to build baseline M: control units (default) or pre-treatment columns.",
    )
    parser.add_argument(
        "--treatment-patterns",
        default="Block,Staggered",
        help='Comma-separated pattern names (must match VALID_PATTERNS). Use "all" to run every pattern. Default: Block,Staggered.',
    )
    args = parser.parse_args()
    ms = args.methods.strip()
    methods = (
        None
        if ms.lower() in ("", "all")
        else [m.strip() for m in ms.split(",") if m.strip()]
    )
    tp = args.treatment_patterns.strip()
    patterns = (
        list(VALID_PATTERNS)
        if tp.lower() == "all"
        else [p.strip() for p in tp.split(",") if p.strip()]
    )
    print(f"Running semi-synthetic experiments with dataset: {args.dataset}")
    main(
        args.dataset,
        methods=methods,
        baseline_type=args.baseline_type,
        patterns=patterns,
    )
