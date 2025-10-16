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
    
    # Define treatment patterns
    if baseline_type == 'control':
        treatment_patterns = {
            'IID': lambda M: Z_iid(M, p_treat=0.2),
            'Block': lambda M: Z_block(M, m1=10, m2=20),
            'Staggered': lambda M: Z_stagger(M, m1=10, min_start=20),
            'Adaptive': lambda M: Z_adaptive(M, lookback_a=5, duration_b=5)
        }
    elif baseline_type == 'pre-treatment':
        treatment_patterns = {
            'IID': lambda M: Z_iid(M, p_treat=0.2),
            'Block': lambda M: Z_block(M, m1=10, m2=10),
            'Staggered': lambda M: Z_stagger(M, m1=10, min_start=10),
            'Adaptive': lambda M: Z_adaptive(M, lookback_a=4, duration_b=2)
        }
    else:
        raise ValueError(f"Invalid baseline type: {baseline_type}")
    
    for t_level in treatment_levels:
        if verbose:
            print(f"--- Treatment level: {t_level} ---")
        
        for pattern_name in treatment_patterns.keys():
            if verbose:
                print(f"  {pattern_name}:")
            
            # Run multiple trials
            for trial in range(n_trials):
                # Set seed for reproducibility while ensuring different patterns per trial
                np.random.seed(trial)
                
                # Generate treatment pattern (different for each trial due to seed)
                Z = treatment_patterns[pattern_name](M)
                
                # Inject treatment
                O_syn, tau_star = inject_treatment_centered(M, Z, treatment_level=t_level)
                
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
                    methods=None, n_trials=10):
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



def main():
    """
    Main function to run the semi-synthetic experiments.
    
    This function:
    1. Loads the California Smoke dataset
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
    """
    # Configuration
    O = np.loadtxt("tests/MLAB_data.txt")
    O = O[8:, :]  # California Smoke Dataset preprocessing
    O = O.T
    n_trials = 5
    treated_states = [38]
    treat_start_years = [19]
    treatment_levels = [0.2, 0.1, 0.05, 0.01]
    
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
    print(f"Dataset: California Smoke (shape: {O.shape})")
    print(f"Treated states: {treated_states}")
    print(f"Treatment start years: {treat_start_years}")
    print(f"Treatment levels: {treatment_levels}")
    print(f"Number of trials per method/pattern: {n_trials}")
    print("="*80)
    print()
    
    # Run experiments for both baseline types
    results_control, agg_control = run_experiments(
        O, treated_states, treat_start_years, treatment_levels, 
        "control", methods, n_trials
    )
    
    print("\n" + "="*80 + "\n")
    
    results_pretreatment, agg_pretreatment = run_experiments(
        O, treated_states, treat_start_years, treatment_levels, 
        "pre-treatment", methods, n_trials
    )
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)
    
    return {
        'control': {'detailed': results_control, 'aggregated': agg_control},
        'pre-treatment': {'detailed': results_pretreatment, 'aggregated': agg_pretreatment}
    }
 
 
if __name__ == "__main__":
    results = main()

