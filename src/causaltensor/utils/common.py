"""
Common utilities shared across analysis modules.

These functions are used by both semi_synthetic (experiment runner) and
real_dataset_report (observed-data report), and any future analysis modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank
from causaltensor.cauest.MCNNM import MC_NNM_with_cross_validation
from causaltensor.cauest.DID import DID
from causaltensor.cauest.SDID import SDID
from causaltensor.cauest.RobustSyntheticControl import robust_synthetic_control
from causaltensor.cauest.OLSSyntheticControl import ols_synthetic_control
from causaltensor.cauest.CovariancePCA import covariance_PCA


def extract_treatment_info_from_Z(Y_df: pd.DataFrame, Z_df: pd.DataFrame | None):
    """
    Derive treated states and treatment start times as integer indices from Z_df.

    Works for any dataset that provides a treatment matrix, including datasets
    with multiple treated units or staggered adoption.

    Parameters
    ----------
    Y_df : pd.DataFrame
        Outcome panel (entity × time).
    Z_df : pd.DataFrame or None
        Binary treatment indicator panel (same shape as Y_df); 1 = treated.

    Returns
    -------
    treated_states : list of int
        Row integer positions in Y_df.values that are ever treated.
        Empty list if Z_df is None or all zeros (no observed treatment).
    treat_start_years : list of int
        Column integer positions of the first treatment period for each
        treated state (same order as treated_states).
        Empty list when treated_states is empty.
    """
    if Z_df is None or not Z_df.values.any():
        return [], []
    Z = Z_df.values
    treated_mask = Z.any(axis=1)
    treated_states = list(np.where(treated_mask)[0])
    treat_start_years = [int(np.argmax(Z[i, :])) for i in treated_states]
    return treated_states, treat_start_years


def treated_states_and_starts_from_Z(Z: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Derive treated row indices and first-treated column indices from a binary mask.

    Same indexing convention as :func:`extract_treatment_info_from_Z`, but for a
    NumPy array ``Z`` (rows = units, columns = time).
    """
    treated_mask = np.asarray(Z, dtype=float).any(axis=1)
    treated_states = list(np.where(treated_mask)[0])
    treat_start_years = [int(np.argmax(Z[i, :])) for i in treated_states]
    return treated_states, treat_start_years


def get_tau_from_method_with_error(method_name: str, O_syn: np.ndarray, Z: np.ndarray):
    """
    Run a single estimator and return (tau_hat, error_or_None).

    Provides a uniform interface over all estimators regardless of their
    individual return-value conventions.

    Parameters
    ----------
    method_name : str
        One of: 'DC_PR_auto_rank', 'MC_NNM_CV', 'CovariancePCA', 'DID',
        'SDID', 'SC', 'RobustSyntheticControl'.
    O_syn : np.ndarray
        Observed panel (n × T).
    Z : np.ndarray
        Binary treatment mask (same shape as O_syn).

    Returns
    -------
    tau_hat : float or np.ndarray
        Point estimate(s). np.nan on failure.
    error : str or None
        Error message if estimation failed, else None.
    """
    try:
        if method_name == 'DC_PR_auto_rank':
            _, tau_hat, _ = DC_PR_auto_rank(O_syn, Z)
        elif method_name == 'MC_NNM_CV':
            _, _, _, tau_hat = MC_NNM_with_cross_validation(O_syn, 1 - Z)
        elif method_name == 'DID':
            _, tau_hat = DID(O_syn, Z)
        elif method_name == 'SDID':
            tau_hat = SDID(O_syn, Z)
        elif method_name == 'SC':
            # Synthetic control expects transposed inputs
            _, tau_hat = ols_synthetic_control(O_syn.T, Z.T)
        elif method_name == 'RobustSyntheticControl':
            _, tau_hat = robust_synthetic_control(O_syn, Z)
        elif method_name == 'CovariancePCA':
            _, tau_hat = covariance_PCA(O_syn, Z, suggest_r=-1)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        return tau_hat, None
    except Exception as e:
        return np.nan, str(e)


def get_tau_from_method(method_name: str, O_syn: np.ndarray, Z: np.ndarray) -> float:
    """
    Run a single estimator and return tau_hat; print a warning on failure.

    Parameters
    ----------
    method_name : str
        Estimator key (see :func:`get_tau_from_method_with_error`).
    O_syn : np.ndarray
        Observed panel (n × T).
    Z : np.ndarray
        Binary treatment mask (same shape as O_syn).

    Returns
    -------
    float
        Estimated treatment effect (np.nan if estimation failed).
    """
    tau_hat, err = get_tau_from_method_with_error(method_name, O_syn, Z)
    if err is not None:
        print(f"    Warning: {method_name} failed with error: {err}")
    return tau_hat
