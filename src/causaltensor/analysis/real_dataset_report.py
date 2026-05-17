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
Use ``--plots`` to also save counterfactual PNGs in the same folder, plus a
single overlay figure ``counterfactual_all_methods.png`` when at least one
method produced a baseline. Pass ``--methods`` as a comma-separated list to
subset estimators (see ``--help``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from causaltensor.datasets.dataset_loader import available_datasets, load_dataset
from causaltensor.utils.common import (
    extract_treatment_info_from_Z,
    get_fit_result_from_method,
)
from causaltensor.utils.panel import default_raw_datasets_path, prepare_panel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from causaltensor.cauest.result import Result

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


def _parse_methods_csv(s: str) -> Tuple[str, ...]:
    """Parse ``--methods`` as comma-separated keys (whitespace trimmed)."""
    return tuple(p.strip() for p in s.split(",") if p.strip())


_DATASETS_WITHOUT_Z = frozenset({"retailrocket", "truus", "movielens"})

# Distinct, moderately saturated colors for combined counterfactual (readable on white).
_COMBINED_CF_LINE_COLORS: Tuple[str, ...] = (
    "#1976d2",
    "#d32f2f",
    "#388e3c",
    "#9a7209",
    "#f57c00",
    "#00838f",
    "#5e35b1",
    "#e64a19",
    "#c2185b",
    "#455a64",
)


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


def _counterfactual_row_from_result(
    res: Optional["Result"],
    unit: int,
    *,
    expected_T: int,
) -> Optional[np.ndarray]:
    """One unit's counterfactual path; uses ``baseline`` like :meth:`Result.plot_actual_vs_counterfactual`."""
    if res is None or res.baseline is None:
        return None
    b = np.asarray(res.baseline, dtype=float)
    if b.ndim != 2 or not (0 <= unit < b.shape[0]) or b.shape[1] != expected_T:
        return None
    return b[unit, :]


def _x_axis_tick_indices(
    T: int,
    tick_text: Sequence[str],
    *,
    max_full_ticks: int = 35,
    target_sparse_ticks: int = 12,
) -> Tuple[np.ndarray, int]:
    """Avoid cluttered x-axis on long panels (e.g. Dunnhumby with many weeks)."""
    text = [str(v) for v in tick_text]
    if len(text) != T:
        text = [str(i) for i in range(T)]

    if T <= max_full_ticks:
        idx = np.arange(T, dtype=int)
    else:
        idx = np.unique(
            np.round(np.linspace(0, T - 1, target_sparse_ticks)).astype(int)
        )

    idx = np.sort(np.unique(idx))
    labels = [text[i] for i in idx]
    long_label = max(len(s) for s in labels) if labels else 0
    tickangle = -50 if (len(idx) > 12 or long_label > 12) else 0
    return idx, tickangle


def _combined_cf_marker_sizes(T: int) -> Tuple[int, int]:
    """(actual, method) marker diameters — smaller when ``T`` is large to limit overlap."""
    if T <= 35:
        return 7, 6
    if T <= 55:
        return 6, 5
    if T <= 80:
        return 5, 4
    if T <= 110:
        return 4, 3
    return 3, 2


# Combined-figure styling (grey treatment band/line; vertical tick labels).
_COMBINED_TREATMENT_SHADE = "rgba(120, 120, 120, 0.2)"
_COMBINED_TREATMENT_SHADE_COL = "rgba(120, 120, 120, 0.25)"
_COMBINED_TREATMENT_VLINE = "rgba(75, 75, 75, 0.9)"


def _combined_counterfactual_figure(
    O: np.ndarray,
    Z: np.ndarray,
    plot_unit: int,
    unit_label: str,
    time_labels: Sequence[str],
    dataset_name: str,
    method_results: Sequence[Tuple[str, Optional["Result"]]],
):
    """
    Overlay actual outcome and per-method counterfactuals (dashed), similar to the
    combined plot in ``tutorials/guides/01_real_observed_panels.ipynb``.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for counterfactual plots. Install with: pip install plotly"
        ) from None

    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n, T = O.shape
    if not (0 <= plot_unit < n):
        return None

    mk_actual, mk_method = _combined_cf_marker_sizes(T)

    x = list(range(T))
    tick_text = [str(v) for v in time_labels]
    actual = O[plot_unit, :]
    z_unit = Z[plot_unit, :]

    fig = go.Figure()

    treated_idx = np.where(z_unit > 0)[0]
    is_treated = len(treated_idx) > 0
    is_monotone = is_treated and np.all(np.diff(z_unit.astype(float)) >= -1e-9)
    x_tick_idx, _ = _x_axis_tick_indices(T, tick_text)
    x_tickvals = x_tick_idx.tolist()
    x_ticktext = [tick_text[i] for i in x_tick_idx]
    x_tickangle = -90
    max_lbl = max((len(s) for s in x_ticktext), default=0)
    bottom_margin = int(120 + min(max_lbl * 6, 140))
    if is_treated:
        if is_monotone:
            t0 = int(treated_idx[0])
            fig.add_vrect(
                x0=t0 - 0.5,
                x1=T - 0.5,
                fillcolor=_COMBINED_TREATMENT_SHADE,
                layer="below",
                line_width=0,
            )
            fig.add_vline(
                x=t0,
                line=dict(color=_COMBINED_TREATMENT_VLINE, width=2, dash="dash"),
            )
        else:
            for t_idx in treated_idx:
                fig.add_vrect(
                    x0=t_idx - 0.5,
                    x1=t_idx + 0.5,
                    fillcolor=_COMBINED_TREATMENT_SHADE_COL,
                    layer="below",
                    line_width=0,
                )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="#000000", width=3),
            marker=dict(size=mk_actual, color="#000000"),
            hovertemplate="t=%{customdata}  actual=%{y:.4g}<extra></extra>",
            customdata=tick_text,
        )
    )

    palette = _COMBINED_CF_LINE_COLORS
    cf_rows: List[np.ndarray] = []
    n_cf_traces = 0
    for i, (method, res) in enumerate(method_results):
        cf = _counterfactual_row_from_result(res, plot_unit, expected_T=T)
        if cf is None or not np.all(np.isfinite(cf)):
            continue
        cf_rows.append(cf)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cf,
                mode="lines+markers",
                name=method,
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=mk_method, color=color),
                opacity=1.0,
                hovertemplate=(
                    f"{method}  t=%{{customdata}}  counterfactual=%{{y:.4g}}<extra></extra>"
                ),
                customdata=tick_text,
            )
        )
        n_cf_traces += 1

    if n_cf_traces == 0:
        return None

    all_y = np.concatenate([actual.reshape(-1)] + [r.reshape(-1) for r in cf_rows])
    finite = all_y[np.isfinite(all_y)]
    if finite.size == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
    span = y_max - y_min
    if span > 0:
        y_pad = 0.05 * span
    else:
        y_pad = max(0.05 * max(abs(y_min), abs(y_max), 1.0), 1e-9)

    title = f"Actual vs counterfactuals — {dataset_name} ({unit_label})"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=0.97),
        xaxis=dict(
            title="Time period",
            tickvals=x_tickvals,
            ticktext=x_ticktext,
            tickangle=x_tickangle,
            showgrid=True,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(
            title="Outcome",
            range=[y_min - y_pad, y_max + y_pad],
            showgrid=True,
            gridcolor="#eeeeee",
        ),
        height=500,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=60, r=20, t=70, b=bottom_margin),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


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
        Estimator keys accepted by ``get_fit_result_from_method``. For dataset
        ``dunnhumby``, the list is filtered to methods valid for general /
        Adaptive assignment (DC-PR, MC-NNM CV, Covariance PCA only).
    datasets_path : str, optional
        ``datasets/raw`` directory; default is the package raw folder.
    counterfactual_output_dir : path-like, optional
        If set, write ``counterfactual_<method>.png`` here for each successful fit,
        and ``counterfactual_all_methods.png`` when at least one method returns a
        finite counterfactual baseline for the chosen unit.
    counterfactual_unit_row : int, optional
        Treated row index for plots (default: first treated row in ``Z``).

    Returns
    -------
    pd.DataFrame
        One row per method with ``tau_hat``, ``treated_pre_exposure_rmse`` (RMSE of
        ``O - baseline`` on strict pre-periods of ever-treated units), ``ok``, etc.
        When PNGs are written, ``df.attrs["counterfactual_png_paths"]``
        (per-method paths, including the combined figure last) and
        ``df.attrs["combined_counterfactual_png_path"]`` when the overlay is saved.
    """
    if datasets_path is None:
        datasets_path = default_raw_datasets_path()

    if dataset_name.lower() == "dunnhumby":
        methods = ["DC_PR_auto_rank", "MC_NNM_CV", "CovariancePCA"]

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
    combined_cf_path: Optional[Path] = None
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
                    "treated_pre_exposure_rmse": np.nan,
                    "ok": False,
                    "error": "No treatment matrix Z for this dataset.",
                }
            )
        return pd.DataFrame(rows)

    method_results: List[Tuple[str, Optional["Result"]]] = []
    for method in methods:
        res, err = get_fit_result_from_method(method, O, Z)
        method_results.append((method, res))
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
                "treated_pre_exposure_rmse": res.treated_pre_exposure_rmse,
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

    if (
        plot_out is not None
        and plot_unit is not None
        and unit_label is not None
        and time_labels is not None
    ):
        try:
            fig_all = _combined_counterfactual_figure(
                O,
                Z,
                plot_unit,
                unit_label,
                time_labels,
                dataset_name,
                method_results,
            )
            if fig_all is not None:
                combined_cf_path = plot_out / "counterfactual_all_methods.png"
                fig_all.write_image(str(combined_cf_path), scale=2)
                logger.info("Wrote %s", combined_cf_path)
                cf_paths.append(combined_cf_path)
        except Exception as exc:
            logger.warning("Combined counterfactual plot failed: %s", exc)

    df = pd.DataFrame(rows)
    if cf_paths:
        df.attrs["counterfactual_png_paths"] = cf_paths
    if combined_cf_path is not None:
        df.attrs["combined_counterfactual_png_path"] = combined_cf_path
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
            "CSV is written to <root>/<dataset>/real_data_report_<dataset>.csv; "
            "--plots PNGs use the same folder."
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help=(
            "Save counterfactual PNGs per method and a combined overlay "
            "(counterfactual_all_methods.png) in the dataset output folder."
        ),
    )
    parser.add_argument(
        "--plot-unit-row",
        type=int,
        default=None,
        help="Row index of treated unit to plot (default: first treated row).",
    )
    parser.add_argument(
        "--methods",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated estimator keys (default: full report set). "
            "Example: --methods DC_PR_auto_rank,MC_NNM_CV,CovariancePCA"
        ),
    )
    args = parser.parse_args(argv)

    methods: Tuple[str, ...] = DEFAULT_METHODS
    if args.methods is not None:
        methods = _parse_methods_csv(args.methods)
        if not methods:
            parser.error("--methods must list at least one non-empty key.")

    plot_dir = None
    if args.plots:
        plot_dir = (
            Path(args.out_dir) / args.dataset
            if args.out_dir
            else Path(__file__).resolve().parent / "results" / "real_data" / args.dataset
        )

    df = run_real_data_report(
        dataset_name=args.dataset,
        methods=methods,
        datasets_path=args.raw_path,
        counterfactual_output_dir=plot_dir,
        counterfactual_unit_row=args.plot_unit_row,
    )
    out_path = save_report(df, args.dataset, output_dir=args.out_dir)
    print_report_table(df)
    print(f"\nReport saved to {out_path}")

    cf_paths = df.attrs.get("counterfactual_png_paths", [])
    if cf_paths:
        parent = cf_paths[0].parent
        n_per_method = len(cf_paths)
        if df.attrs.get("combined_counterfactual_png_path"):
            n_per_method -= 1
        print(
            f"Counterfactual figures: {n_per_method} per-method PNG(s) "
            f"and combined overlay under {parent}"
        )
        comb = df.attrs.get("combined_counterfactual_png_path")
        if comb:
            print(f"Combined: {comb}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Plotly PNG export (Kaleido) spawns Chromium per figure; those libraries log loudly at INFO.
    for _name in ("kaleido", "choreographer"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    main()
