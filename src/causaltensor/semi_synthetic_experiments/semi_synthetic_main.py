import pandas as pd
import numpy as np
import os
from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.RobustSyntheticControl import robust_synthetic_control
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.semi_synthetic_experiments.semi_synthetic_utils import *
from causaltensor.semi_synthetic_experiments.treatment_patterns import *
from causaltensor.datasets.dataset_loader import load_dataset


def extract_treatment_info(dataset_name):
    """
    Extract treatment information (treated entity and treatment start year) for each dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
        
    Returns
    -------
    tuple
        (treated_entity, treatment_start_year)
    """
    treatment_info = {
        'smoking': ('California', 1988),
        'basque': ('Basque Country (Pais Vasco)', 1975),
        'german_reunification': ('West Germany', 1990),
        'texas': ('Texas', 1993),
        'pwt_spain_eu': ('Spain', 1986),
        'pwt_chile_trade': ('Chile', 1976),
        'pwt_korea_democracy': ('Republic of Korea', 1988),
        'pwt_norway_oil': ('Norway', 1971)
    }
    
    if dataset_name not in treatment_info:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(treatment_info.keys())}")
    
    return treatment_info[dataset_name]


def get_tau_from_method(method_name, O_syn, Z):
    """
    Extract tau estimate from different methods that have different APIs.
    
    Parameters
    ----------
    method_name : str
        Name of the method to use
    O_syn : np.array
        Synthetic observed panel data
    Z : np.array
        Treatment mask
        
    Returns
    -------
    float
        Estimated treatment effect (tau_hat)
    """
    try:
        if method_name == 'DC_PR_auto_rank':
            _, tau_hat, _ = DC_PR_auto_rank(O_syn, Z)
        elif method_name == 'MC_NNM_CV':
            _, _, _, tau_hat = MC_NNM_with_cross_validation(O_syn, 1-Z)
        elif method_name == 'DID':
            _, tau_hat = DID(O_syn, Z)
        elif method_name == 'SDID':
            tau_hat = SDID(O_syn, Z)
        elif method_name == 'SC':
            # Synthetic control expects transposed inputs
            _, tau_hat = ols_synthetic_control(O_syn.T, Z.T)
        elif method_name == 'RobustSyntheticControl':
            _, tau_hat = robust_synthetic_control(O_syn, Z)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        return tau_hat
    except Exception as e:
        print(f"    Warning: {method_name} failed with error: {e}")
        return np.nan


def run_semi_synthetic_experiment(O, treated_states, treat_start_years, 
                                   baseline_type='control', 
                                   treatment_levels=[0.2, 0.1, 0.05, 0.01],
                                   methods=None,
                                   n_trials=10,
                                   verbose=True):
    """
    Run semi-synthetic experiments with different treatment patterns, levels, and methods.
    
    Parameters
    ----------
    O : np.array
        Observed panel data
    treated_states : list
        List of treated state indices
    treat_start_years : list
        List of treatment start year indices
    baseline_type : str, default='control'
        Type of baseline matrix to build ('control' or 'pre-treatment')
    treatment_levels : list, default=[0.2, 0.1, 0.05, 0.01]
        List of treatment levels to test
    methods : dict, optional
        List of method names to test. If None, tests all methods.
    n_trials : int, default=10
        Number of trials to run for each method/pattern combination
    verbose : bool, default=True
        Whether to print results during execution
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with columns: method, pattern, treatment_level, trial, tau_star, tau_hat, error
    """
    if methods is None:
        # Default: specify which patterns each method is valid for
        methods = {
            'DC_PR_auto_rank': ['IID', 'Block', 'Staggered', 'Adaptive'],
            'MC_NNM_CV': ['IID', 'Block', 'Staggered', 'Adaptive'],
            'DID': ['Block', 'Staggered'],  # DID typically works with simple treatment patterns
            'SDID': ['Block', 'Staggered'],
            'SC': ['Block'],  #TODO: Add stagger
            'RobustSyntheticControl': ['Block'] #TODO: Add stagger
        }
    
    if verbose:
        print(f"Data shape: {O.shape}")
        print(f"Treated states: {treated_states}, Treatment start years: {treat_start_years}")
    
    # Build baseline matrix
    M, n, T = build_baseline_M(O, treated_states, treat_start_years, baseline_type)
    
    if verbose:
        print(f"Baseline matrix M shape: ({n}, {T})")
        print(f"Baseline type: {baseline_type}\n")
    
    results = []
    
    # Patterns considered (parameters sampled per trial below)
    if baseline_type in ('control', 'pre-treatment'):
        patterns = ['IID', 'Block', 'Staggered', 'Adaptive']
    else:
        raise ValueError(f"Invalid baseline type: {baseline_type}")
    
    for t_level in treatment_levels:
        if verbose:
            print(f"--- Treatment level: {t_level} ---")
        
        for pattern_name in patterns:
            if verbose:
                print(f"  {pattern_name}:")
            
            # Run multiple trials
            for trial in range(n_trials):
                # Set seeds for reproducibility
                # np.random.seed(trial)
                rng = np.random.default_rng(trial)

                # Sample parameters for this trial
                n_local, T_local = M.shape
                m1, m2, lookback_a, duration_b = sample_treatment_parameters(n_local, T_local, rng)

                # Generate treatment pattern deterministically for this trial
                if pattern_name == 'IID':
                    Z = Z_iid(M, p_treat=0.2, rng=rng)
                elif pattern_name == 'Block':
                    Z = Z_block(M, m1=m1, m2=m2, rng=rng)
                elif pattern_name == 'Staggered':
                    Z = Z_stagger(M, m1=m1, min_start=m2, rng=rng)
                elif pattern_name == 'Adaptive':
                    Z = Z_adaptive(M, lookback_a=lookback_a, duration_b=duration_b)
                else:
                    raise ValueError(f"Unknown pattern: {pattern_name}")
                # Guard: ensure at least one treated cell to avoid assertion failure
                if Z.sum() == 0:
                    i = int(rng.integers(0, n_local))
                    j = int(rng.integers(0, max(1, T_local)))
                    Z[i, j if T_local == 0 else j % T_local] = 1

                # Inject treatment deterministically for this trial
                O_syn, tau_star = inject_treatment_centered(M, Z, treatment_level=t_level, rng=rng)
                
                # Test each method (only if valid for this pattern)
                for method_name, valid_patterns in methods.items():
                    # Skip if this method is not valid for the current pattern
                    if pattern_name not in valid_patterns:
                        if verbose: 
                            print(f"    {method_name}: SKIPPED (not valid for {pattern_name})")
                        continue
                    
                    tau_hat = get_tau_from_method(method_name, O_syn, Z)
                    
                    if not np.isnan(tau_hat):
                        error = abs(tau_star - tau_hat) / tau_star
                        if verbose: 
                            print(f"    {method_name} (trial {trial+1}/{n_trials}): tau_star={tau_star:.4f}, tau_hat={tau_hat:.4f}, error={error:.4f}")
                    else:
                        error = np.nan
                        if verbose:
                            print(f"    {method_name}: FAILED")
                    
                    results.append({
                        'method': method_name,
                        'pattern': pattern_name,
                        'treatment_level': t_level,
                        'trial': trial,
                        'tau_star': tau_star,
                        'tau_hat': tau_hat,
                        'error': error
                    })
            
            if verbose:
                print()
    
    results_df = pd.DataFrame(results)
    return results_df


def run_experiments(O, treated_states, treat_start_years, treatment_levels, baseline_type, 
                    methods=None, n_trials=10, dataset_name=None):
    """
    Run experiments for a given baseline type and save results.
    
    Parameters
    ----------
    O : np.array
        Observed panel data
    treated_states : list
        List of treated state indices
    treat_start_years : list
        List of treatment start year indices
    treatment_levels : list
        List of treatment levels to test
    baseline_type : str
        Type of baseline ('control' or 'pre-treatment')
    methods : list, optional
        List of methods to test
    n_trials : int, default=10
        Number of trials to run for each method/pattern combination
    dataset_name : str, optional
        Name of the dataset for organizing results
    """
    results_df = run_semi_synthetic_experiment(
        O=O,
        treated_states=treated_states,
        treat_start_years=treat_start_years,
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        methods=methods,
        n_trials=n_trials,
        verbose=True
    )
    
    # Display results summary
    print("="*80)
    print(f"Results Summary for Baseline Type: {baseline_type}")
    print("="*80)
    
    # Aggregate results: compute mean and std for each method/pattern combination
    aggregated = results_df.groupby(['method', 'pattern', 'treatment_level'])['error'].agg(['mean', 'std']).reset_index()
    aggregated.columns = ['method', 'pattern', 'treatment_level', 'mean_error', 'std_error']
    
    # Pivot table: method vs pattern (showing mean ± std)
    pivot_mean = aggregated.pivot_table(
        values='mean_error', 
        index='method', 
        columns='pattern', 
        aggfunc='mean'
    )
    
    pivot_std = aggregated.pivot_table(
        values='std_error', 
        index='method', 
        columns='pattern', 
        aggfunc='mean'
    )
    
    print("\nAverage Error (Mean ± Std): Method vs Pattern")
    print("-" * 80)
    for method in pivot_mean.index:
        print(f"\n{method}:")
        for pattern in pivot_mean.columns:
            mean_val = pivot_mean.loc[method, pattern]
            std_val = pivot_std.loc[method, pattern]
            if pd.notna(mean_val):
                print(f"  {pattern}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"  {pattern}: N/A")
    print()
    
    # Save detailed results (all trials) to CSV
    if dataset_name:
        results_dir = f'src/causaltensor/semi_synthetic_experiments/results/{dataset_name}'
    else:
        results_dir = 'src/causaltensor/semi_synthetic_experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    output_path = f"{results_dir}/semi_synthetic_{baseline_type}_results_detailed.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Detailed results (all trials) saved to: {output_path}")
    
    # Save aggregated results to CSV
    output_path_agg = f"{results_dir}/semi_synthetic_{baseline_type}_results_aggregated.csv"
    aggregated.to_csv(output_path_agg, index=False)
    print(f"Aggregated results (mean ± std) saved to: {output_path_agg}")
    
    return results_df, aggregated



def main(dataset_name="smoking"):
    """
    Main function to run the semi-synthetic experiments.
    
    This function:
    1. Loads the specified dataset using dataset_loader
    2. Runs experiments with all methods across different:
       - Treatment patterns (IID, Block, Staggered, Adaptive)
       - Treatment levels (0.2, 0.1, 0.05, 0.01)
       - Baseline types (control vs pre-treatment)
       for multiple trials
    3. Compares performance of different causal inference methods:
       - DC_PR_auto_rank: Debiased Convex Panel Regression (auto rank)
       - MC_NNM_CV: Matrix Completion with Nuclear Norm (cross-validation)
       - DID: Difference-in-Differences
       - SDID: Synthetic Difference-in-Differences
       - SC: Synthetic Control (OLS)
       - RobustSyntheticControl: Robust Synthetic Control
    
    Parameters
    ----------
    dataset_name : str, default="smoking"
        Name of the dataset to load. Available datasets:
        smoking, basque, german_reunification, texas, pwt_spain_eu, 
        pwt_chile_trade, pwt_korea_democracy, pwt_norway_oil
    """
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    Y_df, Z_df, X_df = load_dataset(dataset_name)
    
    # Convert to numpy array for compatibility with existing code
    O = Y_df.values
    
    # Extract treatment information
    treated_entity, treatment_start_year = extract_treatment_info(dataset_name)
    
    # Find treated state index and treatment start year index
    treated_states = [Y_df.index.get_loc(treated_entity)]
    treat_start_years = [Y_df.columns.get_loc(treatment_start_year)]
    
    # Configuration
    n_trials = 1
    treatment_levels = [0.2, 0.1, 0.05, 0.01][:1]
    
    # Methods to test: dict mapping method names to valid patterns
    # None means use default (all methods with their valid patterns)
    methods = None  
    # Or customize: methods = {
    #     'DC_PR_auto_rank': ['IID', 'Block', 'Staggered', 'Adaptive'],
    #     'DID': ['Block'],
    #     'SDID': ['Block', 'Staggered']
    # }
    
    print("="*80)
    print("Semi-Synthetic Causal Inference Experiments - Comparison Study")
    print("="*80)
    print(f"Dataset: {dataset_name} (shape: {O.shape})")
    print(f"Treated entity: {treated_entity} (index: {treated_states[0]})")
    print(f"Treatment start year: {treatment_start_year} (index: {treat_start_years[0]})")
    print(f"Treatment levels: {treatment_levels}")
    print(f"Number of trials per method/pattern: {n_trials}")
    print("="*80)
    print()
    
    # Run experiments for both baseline types
    results_control, agg_control = run_experiments(
        O, treated_states, treat_start_years, treatment_levels, 
        "control", methods, n_trials, dataset_name
    )
    
    print("\n" + "="*80 + "\n")
    
    results_pretreatment, agg_pretreatment = run_experiments(
        O, treated_states, treat_start_years, treatment_levels, 
        "pre-treatment", methods, n_trials, dataset_name
    )
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    
    return {
        'control': {'detailed': results_control, 'aggregated': agg_control},
        'pre-treatment': {'detailed': results_pretreatment, 'aggregated': agg_pretreatment}
    }
 
 
if __name__ == "__main__":
    import sys
    
    # Allow dataset name to be passed as command line argument
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "smoking"  # default dataset
    
    print(f"Running semi-synthetic experiments with dataset: {dataset_name}")
    results = main(dataset_name)

