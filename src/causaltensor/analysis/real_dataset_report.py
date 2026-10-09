"""
Real-dataset treatment-effect reports using the same estimators as
``utils.common.get_fit_result_from_method`` (DCPR, MC-NNM CV, CovPCA,
OLS_DID, SDID, SC, RSC).

Only datasets that ship with a treatment matrix ``Z`` can produce a full report
(classic case studies and PWT benchmarks). Large recommendation-style panels
(retailrocket, dunnhumby, truus, movielens) have loader implementations but are
not exposed in :func:`~causaltensor.datasets.load_dataset` until a sampling
strategy is in place.

The CLI requires one dataset name and writes ``real_data_report_<dataset>.csv`` and
``counterfactual_series_<dataset>.csv`` under ``results/real_data/<dataset>/``
(or ``<out-dir>/<dataset>/`` if ``--out-dir`` is set). Figures are drawn by
``causaltensor.analysis.plot``. Pass ``--methods`` as a comma-separated list to
subset estimators (see ``--help``).
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
    CANONICAL_ESTIMATOR_METHODS,
    extract_treatment_info_from_Z,
    get_fit_result_from_method,
)
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

# Match default methods in semi_synthetic / get_fit_result_from_method.
DEFAULT_METHODS: Tuple[str, ...] = (
    "DCPR",
    "MC_NNM_CV",
    "CovPCA",
    "OLS_DID",
    "SDID",
    "SC",
    "RSC",
)


def _parse_methods_csv(s: str) -> Tuple[str, ...]:
    """Parse ``--methods`` as comma-separated keys (whitespace trimmed)."""
    return tuple(p.strip() for p in s.split(",") if p.strip())


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


def _baseline_row(res, unit: int, expected_T: int) -> Optional[np.ndarray]:
    if res is None or res.baseline is None:
        return None
    b = np.asarray(res.baseline, dtype=float)
    if b.ndim != 2 or not (0 <= unit < b.shape[0]) or b.shape[1] != expected_T:
        return None
    return b[unit, :]


def run_real_data_report(
    dataset_name: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    datasets_path: Optional[str] = None,
    *,
    counterfactual_unit_row: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit each estimator on observed ``(Y, Z)`` (one table row per method).

    Uses :func:`~causaltensor.utils.common.get_fit_result_from_method` once per
    method. The returned frame's ``attrs["counterfactual_series"]`` holds the
    plotted unit's actual path and each method's counterfactual, for
    ``causaltensor.analysis.plot``.

    Parameters
    ----------
    dataset_name : str
        Argument to ``load_dataset``.
    methods : sequence of str
        Estimator keys accepted by ``get_fit_result_from_method``. For dataset
        ``dunnhumby``, the list is filtered to methods valid for general /
        Adaptive assignment (DCPR, MC-NNM CV, CovPCA only).
    datasets_path : str, optional
        ``datasets/raw`` directory; default is the package raw folder.
    counterfactual_unit_row : int, optional
        Treated row index whose path is saved (default: first treated row in ``Z``).

    Returns
    -------
    pd.DataFrame
        One row per method with ``tau_hat``, ``treated_pre_exposure_rmse`` (RMSE of
        ``O - baseline`` on strict pre-periods of ever-treated units), ``ok``, etc.
    """
    if datasets_path is None:
        datasets_path = default_raw_datasets_path()

    if dataset_name.lower() == "dunnhumby":
        methods_seq = ["DCPR", "MC_NNM_CV", "CovPCA"]
    else:
        methods_seq = list(methods)
    methods = tuple(dict.fromkeys(methods_seq))

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
    series_rows: List[dict] = []

    plot_unit: Optional[int] = None
    unit_label: Optional[str] = None
    time_labels: Optional[list[str]] = None
    if Z is not None and np.any(Z):
        tr = np.where(np.any(np.asarray(Z, dtype=float) > 0, axis=1))[0]
        if tr.size > 0:
            plot_unit = int(tr[0]) if counterfactual_unit_row is None else int(counterfactual_unit_row)
            if 0 <= plot_unit < O.shape[0]:
                unit_label = str(Y_df.index[plot_unit])
                time_labels = [str(c) for c in Y_df.columns]
            else:
                logger.warning(
                    "counterfactual_unit_row=%s out of range for N=%s; skipping counterfactual series.",
                    plot_unit,
                    O.shape[0],
                )
                plot_unit = None

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
                    "treated_pre_exposure_rmse": np.nan,
                    "ok": False,
                    "error": "No treatment matrix Z for this dataset.",
                }
            )
        return pd.DataFrame(rows)

    for method in methods:
        res, err = get_fit_result_from_method(method, O, Z)
        tau_hat = _tau_to_report_scalar(res.tau) if res is not None else float("nan")
        std_tau = _tau_to_report_scalar(res.std_tau) if res is not None else float("nan")
        ok = err is None and res is not None and np.isfinite(tau_hat)
        err_out = err if err is not None else ("" if ok else "non-finite tau_hat")
        pre_rmse = res.treated_pre_exposure_rmse if res is not None else float("nan")
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
                "treated_pre_exposure_rmse": pre_rmse,
                "ok": ok,
                "error": err_out,
            }
        )
        if plot_unit is None or unit_label is None or time_labels is None:
            continue
        cf = _baseline_row(res, plot_unit, T)
        for t in range(T):
            series_rows.append(
                {
                    "dataset": dataset_name,
                    "unit_row": plot_unit,
                    "unit_label": unit_label,
                    "time_index": t,
                    "time_label": time_labels[t],
                    "actual": float(O[plot_unit, t]),
                    "z": float(Z[plot_unit, t]),
                    "method": method,
                    "counterfactual": float(cf[t]) if cf is not None else float("nan"),
                    "tau_hat": tau_hat,
                    "std_tau": std_tau,
                    "treated_pre_exposure_rmse": pre_rmse,
                }
            )

    df = pd.DataFrame(rows)
    if series_rows:
        df.attrs["counterfactual_series"] = pd.DataFrame(series_rows)
    return df


def print_report_table(df: pd.DataFrame) -> None:
    """Print tau_hat per dataset/method as an aligned table."""
    if df.empty:
        print("(no results)")
        return

    max_pre_w = len("pre_rmse_tr")
    for v in df["treated_pre_exposure_rmse"]:
        max_pre_w = max(max_pre_w, len(f"{float(v):.6g}"))

    col_widths = {
        "dataset": max(len("dataset"), df["dataset"].str.len().max()),
        "method":  max(len("method"),  df["method"].str.len().max()),
        "tau_hat": len("tau_hat"),
        "pre_tr":  max_pre_w,
        "status":  len("status"),
    }

    header = (
        f"{'dataset':<{col_widths['dataset']}}  "
        f"{'method':<{col_widths['method']}}  "
        f"{'tau_hat':>{col_widths['tau_hat']}}  "
        f"{'pre_rmse_tr':>{col_widths['pre_tr']}}  "
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
        pre_tr = r.get("treated_pre_exposure_rmse", np.nan)
        pre_s = f"{pre_tr:.6g}" if pd.notna(pre_tr) else "nan"
        status = "ok" if r.get("ok") else f"FAIL: {r.get('error', '')}"
        print(
            f"{r['dataset']:<{col_widths['dataset']}}  "
            f"{r['method']:<{col_widths['method']}}  "
            f"{tau_s:>{col_widths['tau_hat']}}  "
            f"{pre_s:>{col_widths['pre_tr']}}  "
            f"{status:<{col_widths['status']}}"
        )

    print(sep)


def save_report(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    prefix: str = "real_data_report",
) -> Path:
    """Write ``<root>/<dataset_name>/<prefix>_<dataset_name>.csv`` (default root: package ``results/real_data``)."""
    root = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "results" / "real_data"
    out = root / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{prefix}_{dataset_name}.csv"
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
        help=(
            "Output root (default: analysis/results/real_data). "
            "CSVs are written to <root>/<dataset>/."
        ),
    )
    parser.add_argument(
        "--plot-unit-row",
        type=int,
        default=None,
        help="Row index of the treated unit saved for counterfactual figures (default: first treated row).",
    )
    parser.add_argument(
        "--methods",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated estimator keys (default: full report set). "
            "Example: --methods DCPR,MC_NNM_CV,CovPCA"
        ),
    )
    args = parser.parse_args(argv)

    methods: Tuple[str, ...] = DEFAULT_METHODS
    if args.methods is not None:
        methods = _parse_methods_csv(args.methods)
        if not methods:
            parser.error("--methods must list at least one non-empty key.")
        valid = set(CANONICAL_ESTIMATOR_METHODS)
        bad = [m for m in methods if m not in valid]
        if bad:
            parser.error(f"Unknown method key(s): {bad}. Valid: {sorted(valid)}")

    df = run_real_data_report(
        dataset_name=args.dataset,
        methods=methods,
        datasets_path=args.raw_path,
        counterfactual_unit_row=args.plot_unit_row,
    )
    out_path = save_report(df, args.dataset, output_dir=args.out_dir)
    print_report_table(df)
    print(f"\nReport saved to {out_path}")

    series = df.attrs.get("counterfactual_series")
    if series is not None and len(series):
        series_path = out_path.with_name(f"counterfactual_series_{args.dataset}.csv")
        series.to_csv(series_path, index=False)
        logger.info("Wrote %s", series_path)
        print(f"Counterfactual series saved to {series_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
