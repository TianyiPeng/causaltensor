"""
User-facing estimation interface for real (observed) panel data.

Pass your own (O, Z) arrays together with one or more estimator names;
get back the estimated treatment effect(s).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from causaltensor.utils.common import (
    CANONICAL_ESTIMATOR_METHODS,
    get_tau_from_method_with_error,
)

VALID_METHODS: Tuple[str, ...] = CANONICAL_ESTIMATOR_METHODS


def _scalar(tau: Union[float, np.ndarray]) -> float:
    """Reduce a vector/matrix estimate to a single float."""
    if tau is None:
        return float("nan")
    arr = np.asarray(tau, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def estimate(
    O: np.ndarray,
    Z: np.ndarray,
    method: Union[str, Sequence[str]],
    verbose: bool = True,
) -> Union[float, Dict[str, float]]:
    """
    Estimate the treatment effect on observed panel data.

    Parameters
    ----------
    O : np.ndarray, shape (n, T)
        Observed outcome panel.
    Z : np.ndarray, shape (n, T)
        Binary treatment mask (1 = treated, 0 = control).
    method : str or list of str
        Estimator name(s).  Valid choices::

            ``DCPR``, ``MC_NNM_CV``, ``CovPCA``, ``OLS_DID``, ``SDID``, ``SC``, ``RSC``

        (same as :data:`~causaltensor.utils.common.CANONICAL_ESTIMATOR_METHODS`).

    verbose : bool, default True
        Print each estimator's result (or failure reason).

    Returns
    -------
    float
        Estimated ATT when a single ``method`` string is passed.
    dict[str, float]
        ``{method: tau_hat}`` when a list of methods is passed (deduplicated order preserved).

    Examples
    --------
    >>> import numpy as np
    >>> from causaltensor.real import estimate

    >>> O = np.random.randn(20, 40)
    >>> Z = np.zeros((20, 40)); Z[0, 20:] = 1

    >>> tau = estimate(O, Z, "OLS_DID")

    >>> results = estimate(O, Z, ["OLS_DID", "SDID", "DCPR"])
    """
    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if O.shape != Z.shape:
        raise ValueError(
            f"O and Z must have the same shape; got O={O.shape}, Z={Z.shape}."
        )

    single = isinstance(method, str)
    raw_methods: List[str] = [method] if single else list(method)

    unknown = set(raw_methods) - set(VALID_METHODS)
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}.\nValid methods: {VALID_METHODS}"
        )

    methods: List[str] = []
    seen: set[str] = set()
    for m in raw_methods:
        if m not in seen:
            seen.add(m)
            methods.append(m)

    if verbose:
        print(f"Panel shape: {O.shape}")

    results: Dict[str, float] = {}
    for m in methods:
        tau_raw, err = get_tau_from_method_with_error(m, O, Z)
        tau_hat = _scalar(tau_raw) if err is None else float("nan")
        results[m] = tau_hat
        if verbose:
            if err:
                print(f"  {m}: FAILED ({err})")
            else:
                print(f"  {m}: tau_hat = {tau_hat:.6g}")

    return results[raw_methods[0]] if single else results

