"""
Real-dataset treatment-effect reports using the same estimators as
``utils.common.get_tau_from_method`` (DC-PR, MC-NNM CV, Covariance PCA,
DID, SDID, OLS-SC, Robust SC).

Only datasets that ship with a treatment matrix ``Z`` are included by default
(classic case studies, PWT benchmarks, Dunnhumby promo). Recommendation
datasets without ``Z`` are excluded.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from causaltensor.datasets.dataset_loader import available_datasets, load_dataset
from causaltensor.utils.common import (
    extract_treatment_info_from_Z,
    get_tau_from_method_with_error,
)

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


def _default_raw_path() -> str:
    return str(Path(__file__).resolve().parent.parent / "datasets" / "raw") + "/"


def _prepare_panel(
    Y_df: pd.DataFrame, Z_df: Optional[pd.DataFrame]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Align and convert to float arrays; binarize ``Z`` for estimators."""
    O = Y_df.values.astype(float)
    if Z_df is None:
        return O, None
    Z_aligned = Z_df.reindex(index=Y_df.index, columns=Y_df.columns)
    Z = (Z_aligned.fillna(0).values > 0).astype(float)
    return O, Z


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
        datasets_path = _default_raw_path()

    Y_df, Z_df, _X_df = load_dataset(dataset_name, datasets_path=datasets_path)
    O, Z = _prepare_panel(Y_df, Z_df)

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
    dataset_names: Optional[Iterable[str]] = None,
    methods: Sequence[str] = DEFAULT_METHODS,
    datasets_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Concatenate per-dataset reports for all datasets that have ``Z``.

    Parameters
    ----------
    dataset_names : iterable of str, optional
        If omitted, uses ``datasets_with_treatment_pattern()``.
    methods, datasets_path
        Passed through to :func:`run_report_for_dataset`.
    """
    names = list(dataset_names) if dataset_names is not None else list(datasets_with_treatment_pattern())
    frames = [run_report_for_dataset(name, methods=methods, datasets_path=datasets_path) for name in names]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def format_report_text(df: pd.DataFrame) -> str:
    """Human-readable summary: dataset blocks with method / tau / status."""
    lines: List[str] = []
    for dataset, sub in df.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        if sub.empty:
            lines.append("  (no rows)")
            lines.append("")
            continue
        row0 = sub.iloc[0]
        lines.append(
            f"  Panel: n={row0['n']}, T={row0['T']}, treated cells={row0.get('n_treated_cells', '')}"
        )
        for _, r in sub.iterrows():
            status = "ok" if r.get("ok") else f"fail: {r.get('error', '')}"
            tau = r.get("tau_hat", np.nan)
            tau_s = f"{tau:.6g}" if pd.notna(tau) else "nan"
            lines.append(f"    {r['method']}: tau_hat={tau_s}  ({status})")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def save_report(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
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
    csv_path = output_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)
    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Real-data causal estimates report.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        help="Dataset names (default: all datasets with treatment Z).",
    )
    parser.add_argument(
        "--raw-path",
        default=None,
        help="Path to datasets/raw (default: package raw folder).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CSV and TXT (default: analysis/results/real_data).",
    )
    args = parser.parse_args(argv)

    names = args.datasets if args.datasets else None
    df = run_real_data_report(dataset_names=names, datasets_path=args.raw_path)
    save_report(df, output_dir=args.out_dir)
    print(format_report_text(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
