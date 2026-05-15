import pandas as pd
import numpy as np
from pathlib import Path
from causaltensor.semi_synthetic.experiment import run_experiment
from causaltensor.utils.common import extract_treatment_info_from_Z
from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.utils.panel import default_raw_datasets_path


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


def run_experiments(O, treated_states, treat_start_years, treatment_levels, baseline_type,
                    methods=None, n_trials=10, dataset_name=None, seed=0):
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
    dataset_name : str, optional
        Used for organising output files.
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
        n_trials=n_trials,
        seed=seed,
        verbose=True,
    )

    aggregated = results_df.groupby(['method', 'pattern', 'treatment_level'])['error'].agg(['mean', 'std']).reset_index()
    aggregated.columns = ['method', 'pattern', 'treatment_level', 'mean_error', 'std_error']

    # Save detailed results (all trials) to CSV
    _base = Path(__file__).resolve().parent / "results" / "semi_synthetic_data"
    results_dir = _base / dataset_name if dataset_name else _base
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"semi_synthetic_{baseline_type}_results_detailed.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Detailed results (all trials) saved to: {output_path}")

    output_path_agg = results_dir / f"semi_synthetic_{baseline_type}_results_aggregated.csv"
    aggregated.to_csv(output_path_agg, index=False)
    print(f"Aggregated results (mean ± std) saved to: {output_path_agg}")

    return results_df, aggregated


def main(dataset_name="smoking"):
    """
    Load a built-in dataset and run the full semi-synthetic comparison study.

    Runs all seven estimators across four treatment patterns (IID, Block,
    Staggered, Adaptive), multiple treatment levels, and both baseline types
    (control and pre-treatment where applicable).

    Parameters
    ----------
    dataset_name : str, default "smoking"
        Any name accepted by :func:`causaltensor.datasets.load_dataset`.
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
        O, treated_states, treat_start_years, treatment_levels,
        "control", methods, n_trials, dataset_name,
    )
    output = {'control': {'detailed': results_control, 'aggregated': agg_control}}

    if treat_start_years:
        print("\n" + "="*80 + "\n")
        results_pretreatment, agg_pretreatment = run_experiments(
            O, treated_states, treat_start_years, treatment_levels,
            "pre-treatment", methods, n_trials, dataset_name,
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
    import sys

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "smoking"
    print(f"Running semi-synthetic experiments with dataset: {dataset_name}")
    main(dataset_name)
