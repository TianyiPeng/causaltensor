"""
Real-dataset treatment-effect reports using the same estimators as
``utils.common.get_tau_from_method`` (DC-PR, MC-NNM CV, Covariance PCA,
DID, SDID, OLS-SC, Robust SC).

Only datasets that ship with a treatment matrix ``Z`` can produce a full report
(classic case studies and PWT benchmarks). Large recommendation-style panels
(retailrocket, dunnhumby, truus, movielens) have loader implementations but are
not exposed in :func:`~causaltensor.datasets.load_dataset` until a sampling
strategy is in place.

The CLI requires one dataset name and writes ``real_data_report_<dataset>.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from causaltensor.datasets.dataset_loader import available_datasets, load_dataset
from causaltensor.utils.common import (
    extract_treatment_info_from_Z,
    get_tau_from_method_with_error,
)
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

# Match default methods in semi_synthetic (main() docstring and get_tau_from_method).
DEFAULT_METHODS: Tuple[str, ...] = (
    "DC_PR_auto_rank",
    "MC_NNM_CV",
    "CovariancePCA",
    "DID",
    "SDID",
    "SC",
    "RobustSyntheticControl",
)

_DATASETS_WITHOUT_Z = frozenset({"retailrocket", "truus", "movielens"})


def datasets_with_treatment_pattern() -> Tuple[str, ...]:
    """Built-in dataset names that include a treatment matrix ``Z``."""
    return tuple(n for n in available_datasets() if n not in _DATASETS_WITHOUT_Z)

def _tau_to_report_scalar(tau: Union[float, np.ndarray]) -> float:
    """Reduce vector/matrix tau estimates to a single float for tabular reports."""
    if tau is None:
        return float("nan")
    arr = np.asarray(tau, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])
    return float(np.nanmean(arr))


def run_report_for_dataset(
    dataset_name: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    datasets_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run each estimator on observed ``(Y, Z)`` and return one row per method.

    Parameters
    ----------
    dataset_name : str
        Name accepted by ``load_dataset``.
    methods : sequence of str
        Estimator keys (same as ``utils.common.get_tau_from_method``).
    datasets_path : str, optional
        Path to ``datasets/raw``. Defaults to package ``causaltensor/datasets/raw``.

    Returns
    -------
    pd.DataFrame
        Columns include ``dataset``, ``method``, ``tau_hat``, ``ok``, ``error``,
        panel shape, and treatment summary.
    """
    if datasets_path is None:
        datasets_path = default_raw_datasets_path()

    Y_df, Z_df, _X_df = load_dataset(dataset_name, datasets_path=datasets_path)
    O, Z = prepare_panel(Y_df, Z_df)

    if Z is None or not np.any(Z):
        treated_states, treat_start_years = [], []
        n_treated = 0
    else:
        Z_info = (Z_df.reindex(index=Y_df.index, columns=Y_df.columns).fillna(0) > 0).astype(int)
        Z_info_df = pd.DataFrame(Z_info, index=Y_df.index, columns=Y_df.columns)
        treated_states, treat_start_years = extract_treatment_info_from_Z(Y_df, Z_info_df)
        n_treated = int(Z.sum())

    n, T = O.shape
    rows: List[dict] = []

    if Z is None:
        for method in methods:
            rows.append(
                {
                    "dataset": dataset_name,
                    "n": n,
                    "T": T,
                    "n_treated_cells": np.nan,
                    "n_treated_units": np.nan,
                    "method": method,
                    "tau_hat": np.nan,
                    "ok": False,
                    "error": "No treatment matrix Z for this dataset.",
                }
            )
        return pd.DataFrame(rows)

    for method in methods:
        tau_raw, err = get_tau_from_method_with_error(method, O, Z)
        tau_hat = _tau_to_report_scalar(tau_raw) if err is None else float("nan")
        ok = err is None and np.isfinite(tau_hat)
        err_out = err or ("" if ok else "non-finite tau_hat")
        rows.append(
            {
                "dataset": dataset_name,
                "n": n,
                "T": T,
                "n_treated_cells": n_treated,
                "n_treated_units": len(treated_states),
                "treated_unit_indices": str(treated_states),
                "treatment_start_col_indices": str(treat_start_years),
                "method": method,
                "tau_hat": tau_hat,
                "ok": ok,
                "error": err_out,
            }
        )

    return pd.DataFrame(rows)


def run_real_data_report(
    dataset_name: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    datasets_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run each estimator on observed ``(Y, Z)`` for one dataset (one row per method).

    Parameters
    ----------
    dataset_name : str
        Name accepted by ``load_dataset``.
    methods : sequence of str
        Estimator keys (same as ``utils.common.get_tau_from_method``).
    datasets_path : str, optional
        Path to ``datasets/raw``. Defaults to package ``causaltensor/datasets/raw``.
    """
    return run_report_for_dataset(
        dataset_name, methods=methods, datasets_path=datasets_path
    )


def print_report_table(df: pd.DataFrame) -> None:
    """Print tau_hat per dataset/method as an aligned table."""
    if df.empty:
        print("(no results)")
        return

    col_widths = {
        "dataset": max(len("dataset"), df["dataset"].str.len().max()),
        "method":  max(len("method"),  df["method"].str.len().max()),
        "tau_hat": len("tau_hat"),
        "status":  len("status"),
    }

    header = (
        f"{'dataset':<{col_widths['dataset']}}  "
        f"{'method':<{col_widths['method']}}  "
        f"{'tau_hat':>{col_widths['tau_hat']}}  "
        f"{'status':<{col_widths['status']}}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    prev_dataset = None
    for _, r in df.iterrows():
        if r["dataset"] != prev_dataset:
            if prev_dataset is not None:
                print(sep)
            prev_dataset = r["dataset"]
        tau = r.get("tau_hat", np.nan)
        tau_s = f"{tau:.6g}" if pd.notna(tau) else "nan"
        status = "ok" if r.get("ok") else f"FAIL: {r.get('error', '')}"
        print(
            f"{r['dataset']:<{col_widths['dataset']}}  "
            f"{r['method']:<{col_widths['method']}}  "
            f"{tau_s:>{col_widths['tau_hat']}}  "
            f"{status:<{col_widths['status']}}"
        )

    print(sep)


def save_report(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    prefix: str = "real_data_report",
) -> Path:
    """
    Write CSV under ``output_dir``.

    Returns
    -------
    csv_path : Path
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results" / "real_data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{dataset_name}" if dataset_name else prefix
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)
    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Real-data causal estimates report.")
    parser.add_argument(
        "dataset",
        help=(
            "Dataset name (required), e.g. smoking, basque — same as load_dataset. "
            "Output: real_data_report_<dataset>.csv"
        ),
    )
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Path to datasets/raw (default: package raw folder).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CSV (default: analysis/results/real_data).",
    )
    args = parser.parse_args(argv)

    df = run_real_data_report(dataset_name=args.dataset, datasets_path=args.raw_path)
    out_path = save_report(df, output_dir=args.out_dir, dataset_name=args.dataset)
    print_report_table(df)
    print(f"\nReport saved to {out_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
