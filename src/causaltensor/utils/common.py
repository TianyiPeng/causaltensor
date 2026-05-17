"""
Common utilities shared across analysis modules.

These functions are used by both semi_synthetic (experiment runner) and
real_dataset_report (observed-data report), and any future analysis modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from causaltensor.cauest.CovariancePCA import CovariancePCAPanelSolver
from causaltensor.cauest.DebiasConvex import DC_PR_auto_rank, DCPanelSolver
from causaltensor.cauest.DID import DID, DIDPanelSolver
from causaltensor.cauest.MCNNM import MCNNMPanelSolver, MC_NNM_with_cross_validation
from causaltensor.cauest.OLSSyntheticControl import OLSSCPanelSolver, ols_synthetic_control
from causaltensor.cauest.RobustSyntheticControl import robust_synthetic_control
from causaltensor.cauest.SDID import SDID, SDIDPanelSolver
from causaltensor.cauest.result import Result

# Estimator string keys for CSVs, plots, CLI, and :func:`~causaltensor.real.estimate.estimate`.
CANONICAL_ESTIMATOR_METHODS: tuple[str, ...] = (
    "DCPR",
    "MC_NNM_CV",
    "CovPCA",
    "OLS_DID",
    "SDID",
    "SC",
    "RSC",
)


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
        One of: ``DCPR``, ``MC_NNM_CV``, ``CovPCA``, ``OLS_DID``, ``SDID``, ``SC``, ``RSC``.
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
        if method_name == "DCPR":
            _, tau_hat, _ = DC_PR_auto_rank(O_syn, Z)
        elif method_name == 'MC_NNM_CV':
            _, _, _, tau_hat = MC_NNM_with_cross_validation(O_syn, 1 - Z)
        elif method_name == "OLS_DID":
            _, tau_hat = DID(O_syn, Z)
        elif method_name == 'SDID':
            tau_hat = SDID(O_syn, Z)
        elif method_name == 'SC':
            # Synthetic control expects transposed inputs
            _, tau_hat = ols_synthetic_control(O_syn, Z)
        elif method_name == "RSC":
            _, tau_hat = robust_synthetic_control(O_syn, Z)
        elif method_name == "CovPCA":
            res = CovariancePCAPanelSolver(O_syn, Z).fit()
            tau_hat = res.tau
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


def get_fit_result_from_method(
    method_name: str,
    O: np.ndarray,
    Z: np.ndarray,
) -> Tuple[Optional[Result], Optional[str]]:
    """
    Fit one estimator and return the :class:`~causaltensor.cauest.result.Result`
    (for diagnostics and :meth:`Result.plot_actual_vs_counterfactual`).

    Returns
    -------
    result : Result or None
    error : str or None
    """
    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)
    try:
        if method_name == "DCPR":
            res = DCPanelSolver(O, Z).fit(
                spectrum_cut=0.002, method="convex", method_non_neg=None
            )
        elif method_name == "MC_NNM_CV":
            Omega = 1.0 - Z
            solver = MCNNMPanelSolver(Z=1.0 - Omega)
            res = solver.solve_with_cross_validation(O, K=5, list_l=None)
            res.O = O
            res.Z = Z
        elif method_name == "CovPCA":
            res = CovariancePCAPanelSolver(O, Z).fit()
        elif method_name == "OLS_DID":
            res = DIDPanelSolver(O, Z).fit()
        elif method_name == "SDID":
            res = SDIDPanelSolver(O, Z).fit()
        elif method_name == "SC":
            res = OLSSCPanelSolver(O, Z).fit()
        elif method_name == "RSC":
            Mhat, tau = robust_synthetic_control(O, Z)
            res = Result(baseline=Mhat, tau=tau, return_tau_scalar=True)
            res.O = O
            res.Z = Z
        else:
            return None, f"Unknown method: {method_name}"
        return res, None
    except Exception as e:
        return None, str(e)
