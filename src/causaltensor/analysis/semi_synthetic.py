import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import DEFAULT_METHODS, VALID_PATTERNS
from causaltensor.semi_synthetic.experiment import run_experiment
from causaltensor.utils.common import extract_treatment_info_from_Z
from causaltensor.utils.panel import default_raw_datasets_path

logger = logging.getLogger(__name__)

METHODS_ORDER = list(DEFAULT_METHODS.keys())


def save_semi_synthetic_error_boxplot(
    results_df: pd.DataFrame,
    path: Path,
    *,
    treatment_level: float,
    baseline_type: str,
) -> Path:
    """
    Relative-error box plots by estimator and pattern (as in tutorial 03); saves PNG via Plotly.
    """
    tl = float(treatment_level)
    sub = results_df.loc[results_df["treatment_level"] == tl].dropna(subset=["error"])
    if sub.empty:
        raise ValueError(f"no result rows for treatment_level={tl}")

    fig = px.box(
        sub,
        x="method",
        y="error",
        color="pattern",
        category_orders={"method": METHODS_ORDER, "pattern": VALID_PATTERNS},
        labels={
            "error": "Relative error  |τ* − τ̂| / |τ*|",
            "method": "Estimator",
            "pattern": "Pattern",
        },
        title=f"Error distribution by estimator and pattern  (treatment_level = {tl}, baseline = {baseline_type})",
    )
    fig.update_layout(
        height=480,
        xaxis_tickangle=-25,
        margin=dict(l=60, r=20, t=60, b=120),
        legend=dict(title="Pattern", orientation="h", y=-0.35, x=0.5, xanchor="center"),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=2)
    logger.info("Wrote semi-synthetic error box plot: %s", path)
    return path


def run_semi_synthetic_experiment(O, treated_states, treat_start_years,
                                   baseline_type='control',
                                   treatment_levels=None,
                                   methods=None,
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
        patterns=None,          # run all four patterns
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        n_trials=n_trials,
        seed=seed,
        verbose=verbose,
    )


def run_experiments(
    O,
    treated_states,
    treat_start_years,
    treatment_levels,
    baseline_type,
    methods=None,
    n_trials=10,
    results_dataset_subdir=None,
    seed=0,
    save_plots: bool = False,
    boxplot_treatment_level: Optional[float] = None,
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
    methods : optional
        Passed through to :func:`run_semi_synthetic_experiment`.
    n_trials : int, default 10
        Trials per combination.
    results_dataset_subdir : str, optional
        Subdirectory under ``results/semi_synthetic_data/`` for CSVs and plots.
    seed : int, default 0
        Random seed for :func:`run_semi_synthetic_experiment`.
    save_plots : bool, default False
        If True, save a PNG box plot of relative error alongside the CSVs.
    boxplot_treatment_level : float, optional
        Treatment level slice for the box plot; defaults to the first value in
        ``treatment_levels``.
    """
    results_df = run_semi_synthetic_experiment(
        O=O,
        treated_states=treated_states,
        treat_start_years=treat_start_years,
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        methods=methods,
        n_trials=n_trials,
        seed=seed,
        verbose=True,
    )

    aggregated = results_df.groupby(['method', 'pattern', 'treatment_level'])['error'].agg(['mean', 'std']).reset_index()
    aggregated.columns = ['method', 'pattern', 'treatment_level', 'mean_error', 'std_error']

    # Save detailed results (all trials) to CSV
    _base = Path(__file__).resolve().parent / "results" / "semi_synthetic_data"
    results_dir = _base / results_dataset_subdir if results_dataset_subdir else _base
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"semi_synthetic_{baseline_type}_results_detailed.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Detailed results (all trials) saved to: {output_path}")

    output_path_agg = results_dir / f"semi_synthetic_{baseline_type}_results_aggregated.csv"
    aggregated.to_csv(output_path_agg, index=False)
    print(f"Aggregated results (mean ± std) saved to: {output_path_agg}")

    if save_plots:
        tl_plot = float(boxplot_treatment_level) if boxplot_treatment_level is not None else float(treatment_levels[0])
        tl_tag = str(tl_plot).replace(".", "p")
        box_path = results_dir / f"semi_synthetic_{baseline_type}_error_boxplot_tl{tl_tag}.png"
        save_semi_synthetic_error_boxplot(
            results_df,
            box_path,
            treatment_level=tl_plot,
            baseline_type=baseline_type,
        )
        print(f"Error box plot saved to: {box_path}")

    return results_df, aggregated


def main(
    dataset_name="smoking",
    save_plots: bool = False,
    boxplot_treatment_level: Optional[float] = None,
):
    """
    Load a built-in dataset and run the full semi-synthetic comparison study.

    Runs all seven estimators across four treatment patterns (IID, Block,
    Staggered, Adaptive), multiple treatment levels, and both baseline types
    (control and pre-treatment where applicable).

    Parameters
    ----------
    dataset_name : str, default "smoking"
        Any name accepted by :func:`causaltensor.datasets.load_dataset`.
    save_plots : bool, default False
        If True, :func:`run_experiments` also saves PNG box plots.
    boxplot_treatment_level : float, optional
        Treatment level for the box plot; defaults to the first entry in ``treatment_levels``.
    """
    print(f"Loading dataset: {dataset_name}")
    Y_df, Z_df, _X_df = load_dataset(dataset_name, datasets_path=default_raw_datasets_path())

    O = Y_df.values

    treated_states, treat_start_years = extract_treatment_info_from_Z(Y_df, Z_df)

    treated_entities = Y_df.index[treated_states].tolist() if treated_states else []
    treat_start_labels = Y_df.columns[treat_start_years].tolist() if treat_start_years else []

    n_trials = 3
    treatment_levels = [0.2, 0.1, 0.05, 0.01][:1]
    methods = None
    # Or customize, e.g.:
    # methods = {
    #     'DC_PR_auto_rank': ['IID', 'Block', 'Staggered', 'Adaptive'],
    #     'DID': ['Block'],
    #     'SDID': ['Block', 'Staggered']
    # }

    print("="*80)
    print("Semi-Synthetic Causal Inference Experiments - Comparison Study")
    print("="*80)
    print(f"Dataset: {dataset_name} (shape: {O.shape})")
    print(f"Treated entities: {treated_entities} (indices: {treated_states})")
    print(f"Treatment start: {treat_start_labels} (indices: {treat_start_years})")
    print(f"Treatment levels: {treatment_levels}")
    print(f"Number of trials per method/pattern: {n_trials}")
    print("="*80)
    print()

    results_control, agg_control = run_experiments(
        O,
        treated_states,
        treat_start_years,
        treatment_levels,
        "control",
        methods,
        n_trials,
        dataset_name,
        seed=0,
        save_plots=save_plots,
        boxplot_treatment_level=boxplot_treatment_level,
    )
    output = {'control': {'detailed': results_control, 'aggregated': agg_control}}

    if treat_start_years:
        print("\n" + "="*80 + "\n")
        results_pretreatment, agg_pretreatment = run_experiments(
            O,
            treated_states,
            treat_start_years,
            treatment_levels,
            "pre-treatment",
            methods,
            n_trials,
            dataset_name,
            save_plots=save_plots,
            boxplot_treatment_level=boxplot_treatment_level,
        )
        output['pre-treatment'] = {
            'detailed': results_pretreatment,
            'aggregated': agg_pretreatment,
        }
    else:
        print("\nSkipping pre-treatment baseline: no observed treatment in this dataset.")
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

    return output


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    for _name in ("kaleido", "choreographer"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Semi-synthetic estimator comparison.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="smoking",
        help="Dataset name (default: smoking).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save PNG box plots of relative error next to the result CSVs.",
    )
    parser.add_argument(
        "--boxplot-treatment-level",
        type=float,
        default=None,
        help="Treatment level for the box plot (default: first level in this script's list).",
    )
    args = parser.parse_args()
    print(f"Running semi-synthetic experiments with dataset: {args.dataset}")
    main(
        args.dataset,
        save_plots=args.plots,
        boxplot_treatment_level=args.boxplot_treatment_level,
    )
