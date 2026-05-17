import html
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px

from causaltensor.datasets.dataset_loader import load_dataset
from causaltensor.semi_synthetic.aa_test import DEFAULT_METHODS, VALID_PATTERNS
from causaltensor.semi_synthetic.experiment import run_experiment
from causaltensor.utils.common import extract_treatment_info_from_Z
from causaltensor.utils.panel import default_raw_datasets_path

logger = logging.getLogger(__name__)

METHODS_ORDER = list(DEFAULT_METHODS.keys())


def _keep_errors_within_percentiles(
    df: pd.DataFrame,
    *,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> pd.DataFrame:
    """Within each ``treatment_level``, drop rows whose ``error`` is outside [p_low, p_high]."""
    if df.empty:
        return df
    min_n = 6
    out_parts: List[pd.DataFrame] = []
    for _tl, g in df.groupby("treatment_level", observed=True):
        vals = g["error"].dropna().to_numpy(dtype=float)
        if vals.size < min_n:
            out_parts.append(g)
            continue
        lo, hi = np.percentile(vals, [low_pct, high_pct])
        m = (g["error"] >= lo) & (g["error"] <= hi)
        out_parts.append(g.loc[m])
    return pd.concat(out_parts, axis=0, ignore_index=True)


def _layout_yaxis_keys_bottom_to_top(fig) -> List[str]:
    """Layout keys ``yaxis``, ``yaxis2``, ... sorted from bottom row to top row."""
    layout = fig.layout
    keys: List[str] = []
    if getattr(layout, "yaxis", None) is not None and layout.yaxis.domain:
        keys.append("yaxis")
    for i in range(2, 25):
        k = f"yaxis{i}"
        ax = getattr(layout, k, None)
        if ax is not None and ax.domain:
            keys.append(k)
    return sorted(keys, key=lambda k: float(getattr(layout, k).domain[0]))


def _yaxis_domain_ref(layout_key: str) -> str:
    """Plotly annotation ``yref`` string for a layout y-axis key."""
    if layout_key == "yaxis":
        return "y domain"
    return f"y{layout_key[len('yaxis'):]} domain"


def _facet_row_treatment_level_subtitles(fig) -> None:
    """
    Center treatment-level labels above each facet row using **y-axis domain**
    coordinates so the gap above the plotting area is the same for every row (paper
    coordinates + a global cap used to squash only the top row).
    """
    y_keys = _layout_yaxis_keys_bottom_to_top(fig)
    anns = [
        a
        for a in fig.layout.annotations
        if getattr(a, "xref", None) == "paper"
        and (raw := (a.text or ""))
        and "treatment_level" in raw
        and "=" in raw
    ]
    if not y_keys or len(anns) != len(y_keys):
        logger.warning(
            "facet subtitle fallback (%s ann, %s y-axes); labels may be uneven",
            len(anns),
            len(y_keys),
        )
        yranges = sorted(
            [tuple(ax.domain) for ax in fig.select_yaxes() if ax.domain],
            key=lambda d: d[0],
        )
        if not yranges:
            return
        for ann in anns:
            raw = ann.text or ""
            if "treatment_level" not in raw or "=" not in raw:
                continue
            val = raw.split("=", 1)[1].strip()
            label = f"Treatment level = {val}"
            mid = float(ann.y)
            dom = min(yranges, key=lambda d: abs(0.5 * (d[0] + d[1]) - mid))
            bottom, top = dom[0], dom[1]
            row_h = top - bottom
            ann.update(
                text=label,
                textangle=0,
                x=0.5,
                xref="paper",
                xanchor="center",
                y=top + 0.2 * row_h,
                yref="paper",
                yanchor="bottom",
                font=dict(size=11),
            )
        return

    anns_top_first = sorted(anns, key=lambda a: float(a.y), reverse=True)
    y_above_plot = 1.06
    for ann, yk in zip(anns_top_first, reversed(y_keys)):
        raw = ann.text or ""
        val = raw.split("=", 1)[1].strip()
        label = f"Treatment level = {val}"
        ann.update(
            text=label,
            textangle=0,
            xref="x domain",
            x=0.5,
            xanchor="center",
            yref=_yaxis_domain_ref(yk),
            y=y_above_plot,
            yanchor="bottom",
            font=dict(size=11),
        )


def save_semi_synthetic_error_boxplot(
    results_df: pd.DataFrame,
    path: Path,
    *,
    treatment_levels: List[float],
    dataset_name: str,
) -> Path:
    """
    Relative-error box plots by estimator, one horizontal band per treatment level
    (largest level at top); saves PNG via Plotly.

    Rows are dropped so that, **within each treatment level**, ``error`` lies between
    the **2nd and 98th percentiles** of that panel's errors. Levels with fewer than
    six observations are left unchanged. Y-axes autorange to the remaining data.
    """
    sub = results_df.dropna(subset=["error"]).copy()
    if sub.empty:
        raise ValueError("no result rows for box plot")
    sub = sub.loc[sub["treatment_level"].isin(treatment_levels)]
    if sub.empty:
        raise ValueError("no rows after filtering treatment_levels")
    sub = _keep_errors_within_percentiles(sub, low_pct=2.0, high_pct=98.0)
    if sub.empty:
        raise ValueError("no rows left after 2nd–98th percentile trim")

    pat_order = [p for p in VALID_PATTERNS if p in set(sub["pattern"])]
    # facet_row draws first category at the bottom — reverse so 0.2 is on top.
    tl_facet_order = list(reversed(treatment_levels))
    sub["treatment_level"] = pd.Categorical(
        sub["treatment_level"],
        categories=tl_facet_order,
        ordered=True,
    )

    n_rows = len(tl_facet_order)
    fig = px.box(
        sub,
        x="method",
        y="error",
        color="pattern",
        facet_row="treatment_level",
        facet_row_spacing=0.14,
        category_orders={
            "method": METHODS_ORDER,
            "pattern": pat_order,
            "treatment_level": tl_facet_order,
        },
        labels={
            "error": "Relative error",
            # Do not pass ``method`` here: px can wrongly reuse that label on the color legend.
            "pattern": "Pattern",
        },
    )
    _facet_row_treatment_level_subtitles(fig)
    fig.update_yaxes(
        matches=None,
        autorange=True,
        rangemode="tozero",
        automargin=True,
        title_standoff=28,
        title=dict(font=dict(size=12)),
    )
    dn = dataset_name.strip()
    header = dn if dn.lower().endswith(" dataset") else f"{dn} dataset"
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:17px'><b>{html.escape(header)}</b></span>",
            x=0.5,
            xanchor="center",
            pad=dict(t=10, b=6),
        ),
        margin=dict(l=88, r=28, t=88, b=152),
        height=min(230 * n_rows + 130, 2600),
        legend=dict(
            orientation="h",
            title=dict(text="Pattern"),
            yanchor="top",
            y=-0.17,
            x=0.5,
            xanchor="center",
        ),
    )
    fig.update_xaxes(
        tickangle=-25,
        title_text="",
        title_standoff=8,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), scale=2)
    logger.info("Wrote semi-synthetic error box plot: %s", path)
    return path


def save_semi_synthetic_error_boxplot_from_csv(
    detailed_csv: Path,
    *,
    output_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Rebuild the error box-plot PNG from an existing ``*_results_detailed.csv``, applying
    the same 2nd–98th percentile trimming per treatment level as :func:`save_semi_synthetic_error_boxplot`.

    Parameters
    ----------
    detailed_csv : Path
        Path to ``semi_synthetic_<baseline>_results_detailed.csv``.
    output_path : Path, optional
        PNG path; default replaces ``_results_detailed.csv`` with ``_error_boxplot.png``.
    dataset_name : str, optional
        Bold title line; default is the parent directory name (e.g. ``smoking``).
    """
    detailed_csv = Path(detailed_csv)
    results_df = pd.read_csv(detailed_csv)
    levels = sorted(results_df["treatment_level"].dropna().unique(), reverse=True)
    treatment_levels = [float(x) for x in levels]
    if output_path is None:
        name = detailed_csv.name
        if name.endswith("_results_detailed.csv"):
            out = detailed_csv.parent / name.replace(
                "_results_detailed.csv", "_error_boxplot.png"
            )
        else:
            out = detailed_csv.with_name(f"{detailed_csv.stem}_error_boxplot.png")
    else:
        out = Path(output_path)
    if dataset_name is None:
        dataset_name = detailed_csv.parent.name
    return save_semi_synthetic_error_boxplot(
        results_df,
        out,
        treatment_levels=treatment_levels,
        dataset_name=dataset_name,
    )


def run_semi_synthetic_experiment(O, treated_states, treat_start_years,
                                   baseline_type='control',
                                   treatment_levels=None,
                                   methods=None,
                                   patterns=None,
                                   n_trials=10,
                                   seed=0,
                                   verbose=True):
    """
    Run semi-synthetic experiments with different treatment patterns, levels, and methods.

    Thin wrapper around :func:`causaltensor.semi_synthetic.run_experiment` that
    accepts the pre-derived ``treated_states`` / ``treat_start_years`` lists
    (the internal format used by ``analysis.semi_synthetic``) instead of a raw
    ``Z`` array.

    Parameters
    ----------
    O : np.ndarray
        Observed panel data (n × T).
    treated_states : list of int
        Row indices of treated units in ``O``.
    treat_start_years : list of int
        Column index of the first treated period for each treated unit
        (same order as ``treated_states``).
    baseline_type : {'control', 'pre-treatment'}, default 'control'
        How to build the baseline matrix ``M``.
    treatment_levels : list of float, optional
        Fraction of mean(|M|) injected as tau_star.
        Defaults to ``[0.2, 0.1, 0.05, 0.01]``.
    methods : None | list[str] | dict[str, list[str]], optional
        Estimators to evaluate. See :func:`~causaltensor.semi_synthetic.run_experiment`.
    patterns : list[str] or None, optional
        Subset of ``VALID_PATTERNS`` to simulate. ``None`` runs all four.
    n_trials : int, default 10
        Trials per (pattern, treatment_level) combination.
    seed : int, default 0
        Passed to :func:`~causaltensor.semi_synthetic.run_experiment`.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: method, pattern, treatment_level, trial, tau_star, tau_hat, error.
    """
    if treatment_levels is None:
        treatment_levels = [0.2, 0.1, 0.05, 0.01]

    # Reconstruct a block-treatment Z from the treated_states / treat_start_years
    # lists so we can delegate to the user-facing run_experiment(O, Z, ...).
    Z_real = np.zeros(O.shape, dtype=float)
    for state, start in zip(treated_states, treat_start_years):
        Z_real[state, start:] = 1.0

    return run_experiment(
        O, Z_real,
        methods=methods,
        patterns=patterns,
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        n_trials=n_trials,
        seed=seed,
        verbose=verbose,
    )


def run_experiments(
    O,
    treated_states,
    treat_start_years,
    treatment_levels,
    baseline_type,
    dataset_name: str,
    methods=None,
    patterns=None,
    n_trials=10,
    seed=0,
    save_plots: bool = False,
):
    """
    Run experiments for a given baseline type, print a summary, and save results.

    Parameters
    ----------
    O : np.ndarray
        Observed panel data.
    treated_states : list of int
        Row indices of treated units.
    treat_start_years : list of int
        First treatment column index per treated unit.
    treatment_levels : list of float
        Treatment levels to test.
    baseline_type : str
        'control' or 'pre-treatment'.
    dataset_name : str
        Dataset key (same as :func:`load_dataset`); used for results subdirectory and plots.
    methods : optional
        Passed through to :func:`run_semi_synthetic_experiment`.
    patterns : list[str] or None, optional
        Synthetic patterns to run; ``None`` means all ``VALID_PATTERNS``.
    n_trials : int, default 10
        Trials per combination.
    seed : int, default 0
        Random seed for :func:`run_semi_synthetic_experiment`.
    save_plots : bool, default False
        If True, save one PNG with box plots for every ``treatment_level`` (stacked).
    """
    results_df = run_semi_synthetic_experiment(
        O=O,
        treated_states=treated_states,
        treat_start_years=treat_start_years,
        baseline_type=baseline_type,
        treatment_levels=treatment_levels,
        methods=methods,
        patterns=patterns,
        n_trials=n_trials,
        seed=seed,
        verbose=True,
    )

    aggregated = results_df.groupby(['method', 'pattern', 'treatment_level'])['error'].agg(['mean', 'std']).reset_index()
    aggregated.columns = ['method', 'pattern', 'treatment_level', 'mean_error', 'std_error']

    # Save detailed results (all trials) to CSV
    _base = Path(__file__).resolve().parent / "results" / "semi_synthetic_data"
    results_dir = _base / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"semi_synthetic_{baseline_type}_results_detailed.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Detailed results (all trials) saved to: {output_path}")

    output_path_agg = results_dir / f"semi_synthetic_{baseline_type}_results_aggregated.csv"
    aggregated.to_csv(output_path_agg, index=False)
    print(f"Aggregated results (mean ± std) saved to: {output_path_agg}")

    if save_plots:
        box_path = results_dir / f"semi_synthetic_{baseline_type}_error_boxplot.png"
        save_semi_synthetic_error_boxplot(
            results_df,
            box_path,
            treatment_levels=treatment_levels,
            dataset_name=dataset_name,
        )
        print(f"Error box plot saved to: {box_path}")

    return results_df, aggregated


def main(
    dataset_name="smoking",
    save_plots: bool = False,
    methods: Optional[List[str]] = None,
    baseline_type: str = "control",
    patterns: Optional[List[str]] = None,
):
    """
    Load a built-in dataset and run the full semi-synthetic comparison study.

    Runs estimators across synthetic treatment patterns, multiple treatment levels,
    and a chosen baseline type.

    Parameters
    ----------
    dataset_name : str, default "smoking"
        Any name accepted by :func:`causaltensor.datasets.load_dataset`.
    save_plots : bool, default False
        If True, :func:`run_experiments` also saves a PNG with one box-plot row per treatment level.
    methods : None or list[str], optional
        ``None`` runs all default estimators (see ``DEFAULT_METHODS``).
        A list runs only those methods on every pattern.
    baseline_type : {'control', 'pre-treatment'}, default 'control'
        Which baseline ``M`` to use. ``pre-treatment`` requires observed treatment
        in ``Z`` (non-empty ``treat_start_years``).
    patterns : None or list[str], optional
        Subset of ``VALID_PATTERNS``. ``None`` defaults to ``['Block', 'Staggered']``.
    """
    print(f"Loading dataset: {dataset_name}")
    Y_df, Z_df, _X_df = load_dataset(dataset_name, datasets_path=default_raw_datasets_path())

    O = Y_df.values

    treated_states, treat_start_years = extract_treatment_info_from_Z(Y_df, Z_df)

    treated_entities = Y_df.index[treated_states].tolist() if treated_states else []
    treat_start_labels = Y_df.columns[treat_start_years].tolist() if treat_start_years else []

    if patterns is None:
        patterns = ["Block", "Staggered"]

    n_trials = 100
    treatment_levels = [0.2, 0.1, 0.05]

    print("="*80)
    print("Semi-Synthetic Causal Inference Experiments - Comparison Study")
    print("="*80)
    print(f"Dataset: {dataset_name} (shape: {O.shape})")
    print(f"Treated entities: {treated_entities} (indices: {treated_states})")
    print(f"Treatment start: {treat_start_labels} (indices: {treat_start_years})")
    print(f"Treatment levels: {treatment_levels}")
    print(f"Number of trials per method/pattern: {n_trials}")
    print(f"Baseline type: {baseline_type}")
    print(f"Methods: {methods if methods is not None else 'all (DEFAULT_METHODS)'}")
    print(f"Treatment patterns: {patterns}")
    print("="*80)
    print()

    output = {}

    if baseline_type == "control":
        results_control, agg_control = run_experiments(
            O,
            treated_states,
            treat_start_years,
            treatment_levels,
            "control",
            dataset_name,
            methods,
            patterns,
            n_trials,
            seed=0,
            save_plots=save_plots,
        )
        output["control"] = {"detailed": results_control, "aggregated": agg_control}
    elif baseline_type == "pre-treatment":
        if not treat_start_years:
            print(
                "\nSkipping: pre-treatment baseline requires at least one treated unit "
                "with treatment start after the first period."
            )
        else:
            print("\n" + "="*80 + "\n")
            results_pretreatment, agg_pretreatment = run_experiments(
                O,
                treated_states,
                treat_start_years,
                treatment_levels,
                "pre-treatment",
                dataset_name,
                methods,
                patterns,
                n_trials,
                save_plots=save_plots,
            )
            output["pre-treatment"] = {
                "detailed": results_pretreatment,
                "aggregated": agg_pretreatment,
            }
    else:
        raise ValueError(f"baseline_type must be 'control' or 'pre-treatment', got {baseline_type!r}")
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)

    return output


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    for _name in ("kaleido", "choreographer"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Semi-synthetic estimator comparison.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="smoking",
        help="Dataset name (default: smoking).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save PNG box plots of relative error next to the result CSVs.",
    )
    parser.add_argument(
        "--plot-from-csv",
        metavar="PATH",
        default=None,
        help=(
            "Only rebuild the error box PNG from an existing *_results_detailed.csv "
            "(no experiment run). Optional dataset title defaults to the CSV parent folder name."
        ),
    )
    parser.add_argument(
        "--plot-output",
        metavar="PATH",
        default=None,
        help="Output PNG for --plot-from-csv (default: sibling *_error_boxplot.png).",
    )
    parser.add_argument(
        "--plot-dataset-title",
        default=None,
        metavar="NAME",
        help=(
            "Bold top title when using --plot-from-csv "
            "(default: parent directory of the CSV, e.g. smoking)."
        ),
    )
    parser.add_argument(
        "--methods",
        default="all",
        help='Comma-separated method keys (DEFAULT_METHODS), or "all" (default) for every entry.',
    )
    parser.add_argument(
        "--baseline-type",
        choices=("control", "pre-treatment"),
        default="control",
        help="How to build baseline M: control units (default) or pre-treatment columns.",
    )
    parser.add_argument(
        "--treatment-patterns",
        default="Block,Staggered",
        help='Comma-separated pattern names (must match VALID_PATTERNS). Use "all" to run every pattern. Default: Block,Staggered.',
    )
    args = parser.parse_args()
    if args.plot_from_csv:
        out = (
            Path(args.plot_output)
            if args.plot_output
            else None
        )
        written = save_semi_synthetic_error_boxplot_from_csv(
            Path(args.plot_from_csv),
            output_path=out,
            dataset_name=args.plot_dataset_title,
        )
        print(f"Wrote box plot from CSV: {written}")
        sys.exit(0)
    ms = args.methods.strip()
    methods = (
        None
        if ms.lower() in ("", "all")
        else [m.strip() for m in ms.split(",") if m.strip()]
    )
    tp = args.treatment_patterns.strip()
    patterns = (
        list(VALID_PATTERNS)
        if tp.lower() == "all"
        else [p.strip() for p in tp.split(",") if p.strip()]
    )
    print(f"Running semi-synthetic experiments with dataset: {args.dataset}")
    main(
        args.dataset,
        save_plots=args.plots,
        methods=methods,
        baseline_type=args.baseline_type,
        patterns=patterns,
    )
