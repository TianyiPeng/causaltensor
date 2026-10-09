"""
Figures from analysis CSVs.

    python -m causaltensor.analysis.plot
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from causaltensor.analysis.synthetic_ablation import (
    NOISE_GRID,
    RANK_GRID,
    SIGMA_TIME_GRID,
    SIGMA_UNIT_GRID,
    _held_defaults_caption,
)
from causaltensor.analysis.synthetic_robustness import DGPS, _DGP_LABELS
from causaltensor.semi_synthetic.aa_test import DEFAULT_METHODS

logger = logging.getLogger(__name__)

_RESULTS = Path(__file__).resolve().parent / "results"
_PANEL_ORDER = ("Block", "Staggered", "IID", "Adaptive")
_COMPOSITE_PANELS = (
    ("pwt", "Block", "PWT — Block"),
    ("pwt", "Staggered", "PWT — Staggered"),
    ("movielens", "IID", "MovieLens — IID"),
    ("movielens", "Adaptive", "MovieLens — Adaptive"),
)
_WIDE_METHOD_ORDER = ("OLS_DID", "SDID", "DCPR", "MC_NNM_CV", "SC", "RSC", "CovPCA")

# Estimator colors in ``DEFAULT_METHODS`` order.
_METHOD_LINE_COLORS = (
    "#1976d2",
    "#d32f2f",
    "#388e3c",
    "#9a7209",
    "#f57c00",
    "#ff4081",
    "#5e35b1",
    "#e64a19",
    "#c2185b",
    "#455a64",
)


def method_color(method: str) -> str:
    """Line color for ``method``, fixed by its place in ``DEFAULT_METHODS``."""
    keys = list(DEFAULT_METHODS)
    idx = keys.index(method) if method in keys else 0
    return _METHOD_LINE_COLORS[idx % len(_METHOD_LINE_COLORS)]


def _save(fig, path: Path, dpi: int = 120) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(path)


def _kde_density_curve(x: np.ndarray, x_grid: np.ndarray):
    from scipy.stats import gaussian_kde

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.std(x, ddof=1) < 1e-14 * (1.0 + abs(np.mean(x))):
        return None
    return gaussian_kde(x)(x_grid)


def _ordered_methods(present) -> List[str]:
    present = list(dict.fromkeys(present))
    ordered = [m for m in DEFAULT_METHODS if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered


def _style_panel(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color="#eeeeee")
    for spine in ax.spines.values():
        spine.set_color("#e5e5e5")
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_color("#2a3f5f")
    ax.title.set_fontsize(13)
    ax.title.set_fontweight("medium")
    ax.tick_params(labelsize=11)


def _figure_legend(fig, axes) -> None:
    handles, labels, seen = [], [], set()
    for ax in np.atleast_1d(axes).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(
            handles, labels,
            loc="outside lower center", ncol=min(7, len(labels)),
            frameon=False, fontsize=12,
        )


def plot_aa_null_row(panels: Sequence[Tuple[str, pd.DataFrame]], *, grid_points: int = 256, figsize=None):
    """One row of null KDEs. ``panels`` is ``(title, null_df)``."""
    n = len(panels)
    if figsize is None:
        figsize = (4.0 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    n_gp = max(32, int(grid_points))
    for ax, (title, null_df) in zip(axes[0], panels):
        methods = _ordered_methods(null_df["method"])
        colors = [method_color(m) for m in methods]
        _draw_null_panel(ax, null_df, methods, colors, n_gp, title)
    fig.suptitle(
        r"A/A: $\hat\tau$ when true effect is 0 (fixed $M$, random $Z_\mathrm{syn}$)",
        fontsize=15, color="#2a3f5f", fontweight="medium",
    )
    _figure_legend(fig, axes)
    return fig, axes


def _draw_null_panel(ax, null_df, methods, colors, n_gp, title) -> None:
    all_tau = null_df["tau_hat"].to_numpy(dtype=float)
    all_tau = all_tau[np.isfinite(all_tau)]
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel(r"$\hat\tau$")
    ax.set_ylabel("density")
    if all_tau.size == 0:
        _style_panel(ax)
        return
    lo, hi = float(np.min(all_tau)), float(np.max(all_tau))
    span = hi - lo
    pad = 0.08 * span if span > 0 else max(abs(lo), abs(hi), 1.0) * 0.08
    x_grid = np.linspace(lo - pad, hi + pad, n_gp)
    for k, meth in enumerate(methods):
        x = null_df.loc[null_df["method"] == meth, "tau_hat"].to_numpy(dtype=float)
        y = _kde_density_curve(x, x_grid)
        if y is None:
            continue
        ax.plot(x_grid, y, color=colors[k % len(colors)], label=meth, linewidth=2.0)
    _style_panel(ax)


def plot_empirical_power_row(panels: Sequence[Tuple[str, pd.DataFrame]], *, figsize=None):
    """One row of power curves. ``panels`` is ``(title, power_df)``."""
    n = len(panels)
    if figsize is None:
        figsize = (4.0 * n, 4.2)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False, layout="constrained")
    fig.patch.set_facecolor("white")
    for ax, (title, power_df) in zip(axes[0], panels):
        methods = _ordered_methods(power_df["method"])
        colors = [method_color(m) for m in methods]
        _draw_power_panel(ax, power_df, methods, colors, title, short_xlabel=True)
    fig.suptitle(
        r"Empirical power curves: reject $H_0:\tau=0$ when $|\hat\tau|>c$; "
        r"$c$ = empirical $(1-\alpha)$ quantile of $|\hat\tau|$ under null"
        "\n"
        r"(inject $\Delta\cdot\mathrm{mean}(|M|)$ on treated cells)",
        fontsize=14, color="#2a3f5f", fontweight="medium",
    )
    _figure_legend(fig, axes)
    return fig, axes


def _draw_power_panel(ax, power_df, methods, colors, title, *, short_xlabel: bool) -> None:
    for k, meth in enumerate(methods):
        s2 = power_df.loc[power_df["method"] == meth].sort_values("relative_effect")
        if s2.empty:
            continue
        color = colors[k % len(colors)]
        if "ci_low" in s2.columns and "ci_high" in s2.columns:
            ax.fill_between(
                s2["relative_effect"], s2["ci_low"], s2["ci_high"],
                color=color, alpha=0.15, linewidth=0,
            )
        ax.plot(
            s2["relative_effect"], s2["power"],
            marker="o", ms=5, linewidth=1.8, color=color, label=meth,
        )
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel(
        r"relative effect $\Delta$" if short_xlabel
        else r"relative effect $\delta$ (inject $\delta \cdot \mathrm{mean}(|M|)$ on treated cells)"
    )
    ax.set_ylabel("power")
    _style_panel(ax)


def plot_ablation_figure(df: pd.DataFrame) -> plt.Figure:
    """1×4 line plots: x = ablated parameter, y = mean relative error."""
    order = _ordered_methods(df["method"])
    axes_config = [
        ("rank", r"Rank $r$", r"$r$", RANK_GRID),
        ("sigma_unit", r"Unit heterogeneity $\delta$ ($\times |\tau^*|$)", r"$\delta$", SIGMA_UNIT_GRID),
        ("sigma_time", r"Time heterogeneity $\eta$ ($\times |\tau^*|$)", r"$\eta$", SIGMA_TIME_GRID),
        ("noise", r"Noise $\sigma$", r"$\sigma$", NOISE_GRID),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.8), layout="constrained")
    fig.patch.set_facecolor("white")
    for ax, (key, title, xlabel, grid) in zip(axes, axes_config):
        agg = (
            df.loc[df["axis"] == key]
            .groupby(["axis_value", "method"])["error"]
            .mean()
            .reset_index()
        )
        for method in order:
            g = agg.loc[agg["method"] == method]
            if g.empty:
                continue
            xs = g["axis_value"].to_numpy()
            ys = g["error"].to_numpy()
            pos = {float(v): j for j, v in enumerate(grid)}
            idx = np.argsort([pos.get(float(x), 0) for x in xs])
            ax.plot(
                xs[idx], ys[idx], marker="o", ms=5, linewidth=1.8,
                label=method, color=method_color(method),
            )
        ax.set_title(f"{title}\n{_held_defaults_caption(key)}", linespacing=1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean relative error")
        _style_panel(ax)
    fig.suptitle("Synthetic DGP ablations", fontsize=15, color="#2a3f5f", fontweight="medium")
    _figure_legend(fig, axes)
    return fig


def plot_robustness(df: pd.DataFrame) -> plt.Figure:
    """One panel per assignment. Points are mean relative error ± 1 s.d."""
    key_order = _ordered_methods(df["method"])
    dgp_order = [name for name, _ in DGPS]
    x_base = np.arange(len(dgp_order))
    fig, axes = plt.subplots(1, len(_PANEL_ORDER), figsize=(16, 4.8), sharey=True, layout="constrained")
    fig.patch.set_facecolor("white")
    for ax, pattern in zip(axes, _PANEL_ORDER):
        stats = (
            df.loc[df["pattern"] == pattern]
            .groupby(["dgp", "method"])["error"]
            .agg(mean="mean", std="std")
            .reset_index()
        )
        methods = [m for m in key_order if (stats["method"] == m).any()]
        n = max(len(methods), 1)
        dodge = 0.8 / n
        for i, method in enumerate(methods):
            g = stats.loc[stats["method"] == method]
            xs, ys, yerr = [], [], []
            for j, dgp in enumerate(dgp_order):
                row = g.loc[g["dgp"] == dgp]
                if row.empty:
                    continue
                xs.append(x_base[j] + (i - (len(methods) - 1) / 2) * dodge)
                ys.append(float(row["mean"].iloc[0]))
                yerr.append(float(row["std"].iloc[0]))
            ax.errorbar(
                xs, ys, yerr=yerr, fmt="o", color=method_color(method), label=method,
                capsize=2, markersize=5, linewidth=1.2,
            )
        ax.set_xticks(x_base)
        ax.set_xticklabels([_DGP_LABELS[d] for d in dgp_order], rotation=25, ha="right")
        ax.set_title(pattern)
        _style_panel(ax)
    axes[0].set_ylabel("Mean relative error")
    axes[0].yaxis.label.set_size(12)
    _figure_legend(fig, axes)
    return fig


def _pivot_ordered(df, *, metric, methods_order):
    pivots = {}
    for m in methods_order:
        pivot = df[df["method"] == m].pivot_table(index="N", columns="T", values=metric, aggfunc="first")
        pivots[m] = pivot.reindex(sorted(pivot.index)).reindex(columns=sorted(pivot.columns))
    return pivots


def _aligned_int_grid(ptab, counts) -> np.ndarray:
    return np.nan_to_num(counts.reindex(index=ptab.index, columns=ptab.columns).values, nan=0.0).astype(int)


def _draw_load_test_failure_overlays(ax, metric_row, n_timeout, n_memlim, n_err) -> None:
    n_rows, n_cols = n_timeout.shape

    def add_cell(c, r, *, hatched):
        kw = dict(xy=(c - 0.5, r - 0.5), width=1.0, height=1.0, facecolor="white", zorder=10)
        if hatched:
            kw.update(edgecolor="0.45", hatch="...", linewidth=0.0)
        else:
            kw.update(edgecolor="none", linewidth=0)
        ax.add_patch(Rectangle(**kw))

    for r in range(n_rows):
        for c in range(n_cols):
            nt, nm, ne = int(n_timeout[r, c]), int(n_memlim[r, c]), int(n_err[r, c])
            if metric_row == 0:
                if nt > 0:
                    add_cell(c, r, hatched=True)
                elif nm > 0:
                    add_cell(c, r, hatched=False)
            elif metric_row == 1:
                if nm > 0:
                    add_cell(c, r, hatched=True)
                elif nt > 0:
                    add_cell(c, r, hatched=False)
            elif nt > 0 or nm > 0:
                add_cell(c, r, hatched=False)
            elif ne > 0:
                add_cell(c, r, hatched=True)


def plot_load_test_heatmap_figure(df_agg: pd.DataFrame, methods: Sequence[str]) -> plt.Figure:
    """Wide heatmaps: three metric rows, one column per estimator."""
    methods_list = list(methods)
    fig, axes = plt.subplots(3, len(methods_list), figsize=(2.05 * len(methods_list) + 0.8, 6.6), squeeze=False)
    rownames = ["Wall time (median, s)", "RSS fit peak (median, MiB)", "Relative ATT error (median)"]
    time_pv = _pivot_ordered(df_agg, metric="time_s", methods_order=methods_list)
    rss_pv = _pivot_ordered(df_agg, metric="rss_fit_peak_mb", methods_order=methods_list)
    err_pv = _pivot_ordered(df_agg, metric="relative_error", methods_order=methods_list)
    timeout_pv = _pivot_ordered(df_agg, metric="n_timeout", methods_order=methods_list)
    mem_pv = _pivot_ordered(df_agg, metric="n_memory_limit", methods_order=methods_list)
    nerr_pv = _pivot_ordered(df_agg, metric="n_err", methods_order=methods_list)
    t_vals = df_agg["time_s"].to_numpy(dtype=float)
    r_vals = df_agg["rss_fit_peak_mb"].to_numpy(dtype=float)
    e_vals = df_agg["relative_error"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    vmin_time, vmax_time = float(np.nanpercentile(t_vals, 5)), float(np.nanpercentile(t_vals, 98))
    vmin_rss, vmax_rss = float(np.nanpercentile(r_vals, 5)), float(np.nanpercentile(r_vals, 98))
    fin = e_vals[np.isfinite(e_vals)]
    vmax_rel_err = float(np.nanpercentile(fin, 95)) if fin.size else 1.0
    grids = [
        (time_pv, plt.cm.viridis, vmin_time, vmax_time),
        (rss_pv, plt.cm.plasma, vmin_rss, vmax_rss),
        (err_pv, plt.cm.magma, 0.0, vmax_rel_err),
    ]
    for j, method in enumerate(methods_list):
        for i, (pmap, cmap, v0, v1) in enumerate(grids):
            ax = axes[i, j]
            ptab = pmap[method]
            mat = np.ma.array(ptab.values.astype(float), mask=np.isnan(ptab.values.astype(float)))
            im = ax.imshow(mat, aspect="equal", origin="upper", cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            _draw_load_test_failure_overlays(
                ax, i,
                _aligned_int_grid(ptab, timeout_pv[method]),
                _aligned_int_grid(ptab, mem_pv[method]),
                _aligned_int_grid(ptab, nerr_pv[method]),
            )
            ax.set_xticks(range(len(ptab.columns)))
            ax.set_xticklabels([str(c) for c in ptab.columns], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(ptab.index)))
            if j == 0:
                ax.set_yticklabels([str(r) for r in ptab.index], fontsize=7)
                ax.set_ylabel(rownames[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(method, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Load test: median time, rss_fit_peak_mb, relative error over trials", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def _panel_csv(root: Path, dataset: str, pattern: str, suffix: str) -> Optional[Path]:
    folder = root / dataset
    for name in (f"control_{pattern}_{suffix}.csv", f"{pattern}_{suffix}.csv"):
        path = folder / name
        if path.is_file():
            return path
    return None


def _x_axis_tick_indices(T: int) -> np.ndarray:
    if T <= 35:
        return np.arange(T, dtype=int)
    return np.unique(np.round(np.linspace(0, T - 1, 12)).astype(int))


def _shade_treatment(ax, z: np.ndarray) -> None:
    treated = np.where(np.asarray(z, dtype=float) > 0)[0]
    if treated.size == 0:
        return
    T = len(z)
    shade = (120 / 255, 120 / 255, 120 / 255)
    if np.all(np.diff(np.asarray(z, dtype=float)) >= -1e-9):
        t0 = int(treated[0])
        ax.axvspan(t0 - 0.5, T - 0.5, color=shade, alpha=0.2, zorder=0)
        ax.axvline(t0, color=(75 / 255, 75 / 255, 75 / 255), ls=_CF_DASH, lw=2.0, zorder=1)
        return
    for t in treated:
        ax.axvspan(int(t) - 0.5, int(t) + 0.5, color=shade, alpha=0.25, zorder=0)


def _method_order(df: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(df["method"]))


# Dash matches the previous combined counterfactual figure: equal on and off, longer than Matplotlib "--".
_CF_DASH = (0, (5.5, 5.5))


def _combined_counterfactual_figure(df: pd.DataFrame) -> Optional[plt.Figure]:
    methods = _method_order(df)
    base = df[df["method"] == methods[0]].sort_values("time_index")
    labels = [str(v) for v in base["time_label"]]
    actual = base["actual"].to_numpy(dtype=float)
    z = base["z"].to_numpy(dtype=float)
    x = np.arange(len(base))
    T = len(base)
    ms = 7 if T <= 35 else 3
    drawn = False
    fig, ax = plt.subplots(figsize=(11.8, 7.2), layout="constrained")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    _shade_treatment(ax, z)
    ax.plot(x, actual, color="black", lw=3.2, marker="o", ms=ms + 1, label="Actual", zorder=3)
    for method in methods:
        cf = df[df["method"] == method].sort_values("time_index")["counterfactual"].to_numpy(dtype=float)
        if cf.size != T or not np.all(np.isfinite(cf)):
            continue
        ax.plot(x, cf, color=method_color(method), lw=2.2, ls=_CF_DASH, marker="o", ms=ms, label=method, zorder=2)
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    unit_label = str(base["unit_label"].iloc[0])
    dataset_name = str(base["dataset"].iloc[0])
    ax.set_title(f"Actual vs counterfactuals — {dataset_name} ({unit_label})", color="#2a3f5f", fontsize=22, fontweight="medium", pad=10)
    ax.set_xlabel("Time period", fontsize=16, labelpad=8)
    ax.set_ylabel("Outcome", fontsize=16)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_axisbelow(True)
    ax.grid(True, color="#eeeeee")
    for spine in ax.spines.values():
        spine.set_color("#e5e5e5")
    idx = _x_axis_tick_indices(T)
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=90, fontsize=14, ha="center", va="top")
    fig.legend(loc="outside lower center", ncol=4, frameon=False, fontsize=15, handlelength=3.2, borderaxespad=0.15)
    return fig


def save_real_counterfactual_figures(df: pd.DataFrame, output_dir: Path, dpi: int = 120) -> None:
    """Combined counterfactual PNG from a counterfactual-series CSV."""
    fig_all = _combined_counterfactual_figure(df)
    if fig_all is not None:
        _save(fig_all, output_dir / "counterfactual_all_methods.png", dpi)


def plot_from_results(root: Optional[Path] = None, dpi: int = 120) -> None:
    """Write every paper figure found under ``results/``."""
    root = Path(root) if root is not None else _RESULTS
    _plot_power(root / "power_analysis", dpi)
    _plot_ablation(root / "synthetic_ablation", dpi)
    _plot_robustness(root / "synthetic_robustness", dpi)
    _plot_load(root / "load_tests", dpi)
    _plot_real(root / "real_data")


def _plot_power(folder: Path, dpi: int) -> None:
    if not folder.is_dir():
        return
    power_panels, null_panels = [], []
    for dataset, pattern, title in _COMPOSITE_PANELS:
        power_path = _panel_csv(folder, dataset, pattern, "empirical_power")
        null_path = _panel_csv(folder, dataset, pattern, "null_trials")
        if power_path is None or null_path is None:
            logger.info("Composite skipped; missing CSVs for %s / %s", dataset, pattern)
            return
        power_panels.append((title, pd.read_csv(power_path)))
        null_panels.append((title, pd.read_csv(null_path)))
    fig, _ = plot_empirical_power_row(power_panels)
    _save(fig, folder / "composite_power_1x4.png", dpi)
    fig, _ = plot_aa_null_row(null_panels)
    _save(fig, folder / "composite_null_tau_1x4.png", dpi)


def _plot_ablation(folder: Path, dpi: int) -> None:
    if not folder.is_dir():
        return
    for csv in sorted(folder.glob("ablation_*_trials.csv")):
        fig = plot_ablation_figure(pd.read_csv(csv))
        _save(fig, csv.with_name(csv.name.replace("_trials.csv", ".png")), dpi)


def _plot_robustness(folder: Path, dpi: int) -> None:
    if not folder.is_dir():
        return
    for csv in sorted(folder.glob("robustness_*_trials.csv")):
        fig = plot_robustness(pd.read_csv(csv))
        _save(fig, csv.with_name(csv.name.replace("_trials.csv", ".png")), dpi)


def _plot_load(folder: Path, dpi: int) -> None:
    csv = folder / "load_test_cells.csv"
    if not csv.is_file():
        return
    df = pd.read_csv(csv)
    present = set(df["method"].unique())
    order = [m for m in _WIDE_METHOD_ORDER if m in present]
    order += [m for m in DEFAULT_METHODS if m in present and m not in order]
    fig = plot_load_test_heatmap_figure(df, order)
    _save(fig, folder / "load_test_heatmaps.png", dpi)


def _plot_real(folder: Path) -> None:
    if not folder.is_dir():
        return
    for csv in sorted(folder.glob("*/counterfactual_series_*.csv")):
        save_real_counterfactual_figures(pd.read_csv(csv), csv.parent)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    plot_from_results()


if __name__ == "__main__":
    main()
