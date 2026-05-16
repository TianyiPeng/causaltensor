"""
Real-dataset treatment-effect reports using the same estimators as
``utils.common.get_fit_result_from_method`` (DC-PR, MC-NNM CV, Covariance PCA,
DID, SDID, OLS-SC, Robust SC).

Only datasets that ship with a treatment matrix ``Z`` can produce a full report
(classic case studies and PWT benchmarks). Large recommendation-style panels
(retailrocket, dunnhumby, truus, movielens) have loader implementations but are
not exposed in :func:`~causaltensor.datasets.load_dataset` until a sampling
strategy is in place.

The CLI requires one dataset name and writes ``real_data_report_<dataset>.csv`` under
``results/real_data/<dataset>/`` (or ``<out-dir>/<dataset>/`` if ``--out-dir`` is set).
Use ``--plots`` to also save counterfactual PNGs in the same folder.
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
    get_fit_result_from_method,
)
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

# Match default methods in semi_synthetic / get_fit_result_from_method.
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


def run_real_data_report(
    dataset_name: str,
    methods: Sequence[str] = DEFAULT_METHODS,
    datasets_path: Optional[str] = None,
    *,
    counterfactual_output_dir: Optional[Union[str, Path]] = None,
    counterfactual_unit_row: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fit each estimator on observed ``(Y, Z)`` (one table row per method).

    Uses :func:`~causaltensor.utils.common.get_fit_result_from_method` once per
    method; ``tau_hat`` and optional counterfactual PNGs come from the same fit.

    Parameters
    ----------
    dataset_name : str
        Argument to ``load_dataset``.
    methods : sequence of str
        Estimator keys accepted by ``get_fit_result_from_method``.
    datasets_path : str, optional
        ``datasets/raw`` directory; default is the package raw folder.
    counterfactual_output_dir : path-like, optional
        If set, write ``counterfactual_<method>.png`` here for each successful fit.
    counterfactual_unit_row : int, optional
        Treated row index for plots (default: first treated row in ``Z``).

    Returns
    -------
    pd.DataFrame
        Report columns plus, when PNGs are written, ``df.attrs["counterfactual_png_paths"]``.
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
    cf_paths: list[Path] = []
    plot_out = Path(counterfactual_output_dir) if counterfactual_output_dir is not None else None

    plot_unit: Optional[int] = None
    unit_label: Optional[str] = None
    time_labels: Optional[list[str]] = None
    if plot_out is not None and Z is not None and np.any(Z):
        plot_out.mkdir(parents=True, exist_ok=True)
        tr = np.where(np.any(np.asarray(Z, dtype=float) > 0, axis=1))[0]
        if tr.size > 0:
            plot_unit = int(tr[0]) if counterfactual_unit_row is None else int(counterfactual_unit_row)
            if 0 <= plot_unit < O.shape[0]:
                unit_label = str(Y_df.index[plot_unit])
                time_labels = [str(c) for c in Y_df.columns]
            else:
                logger.warning(
                    "counterfactual_unit_row=%s out of range for N=%s; skipping plots.",
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
                    "ok": False,
                    "error": "No treatment matrix Z for this dataset.",
                }
            )
        return pd.DataFrame(rows)

    for method in methods:
        res, err = get_fit_result_from_method(method, O, Z)
        tau_hat = _tau_to_report_scalar(res.tau) if res is not None else float("nan")
        ok = err is None and res is not None and np.isfinite(tau_hat)
        err_out = err if err is not None else ("" if ok else "non-finite tau_hat")
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
        if (
            plot_out is not None
            and plot_unit is not None
            and unit_label is not None
            and time_labels is not None
            and res is not None
        ):
            try:
                fig = res.plot_actual_vs_counterfactual(
                    plot_unit,
                    unit_label=unit_label,
                    time_labels=time_labels,
                    title=f"{method} — {dataset_name} — {unit_label}",
                )
                path = plot_out / f"counterfactual_{method}.png"
                fig.write_image(str(path), scale=2)
                logger.info("Wrote %s", path)
                cf_paths.append(path)
            except Exception as exc:
                logger.warning("Counterfactual plot failed for %s: %s", method, exc)

    df = pd.DataFrame(rows)
    if cf_paths:
        df.attrs["counterfactual_png_paths"] = cf_paths
    return df


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
            "CSV is written to <root>/<dataset>/real_data_report_<dataset>.csv; "
            "--plots PNGs use the same folder."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Save counterfactual PNGs in the same folder as the CSV for this dataset."
        ),
    )
    parser.add_argument(
        "--plot-unit-row",
        type=int,
        default=None,
        help="Row index of treated unit to plot (default: first treated row).",
    )
    args = parser.parse_args(argv)

    plot_dir = None
    if args.plots:
        plot_dir = (
            Path(args.out_dir) / args.dataset
            if args.out_dir
            else Path(__file__).resolve().parent / "results" / "real_data" / args.dataset
        )

    df = run_real_data_report(
        dataset_name=args.dataset,
        datasets_path=args.raw_path,
        counterfactual_output_dir=plot_dir,
        counterfactual_unit_row=args.plot_unit_row,
    )
    out_path = save_report(df, args.dataset, output_dir=args.out_dir)
    print_report_table(df)
    print(f"\nReport saved to {out_path}")

    cf_paths = df.attrs.get("counterfactual_png_paths", [])
    if cf_paths:
        print(f"Counterfactual figures ({len(cf_paths)}) under {cf_paths[0].parent}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Plotly PNG export (Kaleido) spawns Chromium per figure; those libraries log loudly at INFO.
    for _name in ("kaleido", "choreographer"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    main()
