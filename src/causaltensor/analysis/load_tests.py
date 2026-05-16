"""
Load tests: wall time, **``rss_fit_peak_mb``**, ATT relative error on an N×T grid.

``rss_fit_peak_mb`` samples worker RSS during ``get_tau_from_method_with_error``:

    max(RSS) − RSS at estimator entry      (polling every ``RSS_SAMPLE_FIT_S``)

This is what we persist to CSV/plots—it tracks `(N,T)` better than summed process
trees when the interpreter baseline dominates totals.

Optional ``--memory-mb``: the parent terminates the subprocess when incremental
RSS during the estimator (same definition as ``rss_fit_peak_mb``) exceeds this cap.

Windows: ``spawn`` + ``freeze_support`` under ``__main__``.

``--treatment-level`` matches :func:`~causaltensor.synthetic.dgp.generate`
(ATT = level × mean(|M|) before noise).

::

    poetry run python -m causaltensor.analysis.load_tests --timeout 600 --memory-mb 8192
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import psutil

from causaltensor.synthetic import generate
from causaltensor.utils.common import get_tau_from_method_with_error

DEFAULT_N: Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000)
DEFAULT_T: Tuple[int, ...] = (10, 50, 100, 500, 1000, 5000, 10000)

DEFAULT_METHODS: Tuple[str, ...] = (
    "DID",
    "SDID",
    "DC_PR_auto_rank",
    "MC_NNM_CV",
    "SC",
    "RobustSyntheticControl",
    "CovariancePCA",
)

_RESULTS_DIR = Path(__file__).parent / "results" / "load_tests"

_EPS_TAU = 1e-12

RSS_POLL_S = 0.2

RSS_SAMPLE_FIT_S = 0.05

# ``trial_seed = seed + cell_idx * _SEED_CELL_STRIDE + trial`` keeps reps disjoint
# from other cells without requiring n_reps to be baked into the multiplier.
_SEED_CELL_STRIDE = 1_000_000


def _method_prune_should_skip(N: int, T: int, anchors: Sequence[Tuple[int, int]]) -> bool:
    """Skip ``(N, T)`` if northeast of some **per-method** anchor (weakly larger ``N``, ``T`` not equal)."""

    for fn, ft in anchors:
        if N >= fn and T >= ft and (N > fn or T > ft):
            return True
    return False


def _scalarize_tau_hat(tau_hat: Any) -> float:
    if tau_hat is None:
        return float("nan")
    arr = np.asarray(tau_hat, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    x = float(arr[0])
    return float("nan") if np.isnan(x) else x


def _relative_error(tau_hat: float, tau_true: float) -> float:
    if not np.isfinite(tau_hat) or not np.isfinite(tau_true):
        return float("nan")
    return abs(tau_hat - tau_true) / max(abs(tau_true), _EPS_TAU)


def _verbose_trial_suffix(res: Dict[str, Any]) -> str:
    st = res["status"]
    if st == "ok":
        return (
            f"time_s={res['time_s']:.3f}s  rss_fit_peak={res['rss_fit_peak_mb']:.2f}MiB "
            f"rel_err={res['relative_error']:.3f}"
        )
    if st == "timeout":
        return "TIMEOUT"
    if st == "memory_limit":
        return "MEM_LIMIT"
    err = str(res.get("error_detail") or "")
    snippet = err if len(err) <= 55 else err[:52] + "…"
    return f"ERR: {snippet}"


def _run_measurement_core(
    method: str,
    O: np.ndarray,
    Z: np.ndarray,
    tau_true: float,
    *,
    fit_incr_peak_mb_share: Optional[Any] = None,
    fit_incr_lock: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Estimate τ̂ ; record ``rss_fit_peak_mb`` = max worker RSS − RSS at estimator entry.
    Optionally mirror running incremental peak MiB into shared memory for ``--memory-mb``.
    """
    p = psutil.Process()
    rss0 = int(p.memory_info().rss)
    mx = {"v": rss0}
    stop_ev = threading.Event()

    def mirror_fit_incr_peak_to_share() -> None:
        if fit_incr_peak_mb_share is None:
            return
        incr = max(0.0, float(mx["v"] - rss0)) / (1024.0 * 1024.0)
        with fit_incr_lock:
            if incr > fit_incr_peak_mb_share.value:
                fit_incr_peak_mb_share.value = incr

    def _sample_rss_worker() -> None:
        while not stop_ev.wait(RSS_SAMPLE_FIT_S):
            mx["v"] = max(mx["v"], int(psutil.Process().memory_info().rss))
            mirror_fit_incr_peak_to_share()

    th = threading.Thread(target=_sample_rss_worker, daemon=True)
    th.start()

    try:
        t0 = time.perf_counter()
        tau_hat_raw, err = get_tau_from_method_with_error(method, O, Z)
        elapsed = time.perf_counter() - t0
    finally:
        stop_ev.set()
        mx["v"] = max(mx["v"], int(psutil.Process().memory_info().rss))
        mirror_fit_incr_peak_to_share()
        th.join(timeout=10.0)

    rss_fit_peak_mb = max(0.0, float(mx["v"] - rss0)) / (1024.0 * 1024.0)
    if fit_incr_peak_mb_share is not None:
        with fit_incr_lock:
            fit_incr_peak_mb_share.value = max(float(fit_incr_peak_mb_share.value), rss_fit_peak_mb)

    tau_hat = _scalarize_tau_hat(tau_hat_raw)

    if err is not None:
        return {
            "time_s": float(elapsed),
            "rss_fit_peak_mb": float(rss_fit_peak_mb),
            "tau_hat": float(tau_hat),
            "relative_error": float("nan"),
            "status": "error",
            "error_detail": err,
        }

    return {
        "time_s": float(elapsed),
        "rss_fit_peak_mb": float(rss_fit_peak_mb),
        "tau_hat": float(tau_hat),
        "relative_error": _relative_error(tau_hat, tau_true),
        "status": "ok",
        "error_detail": None,
    }


def _trial_worker(
    conn: Any,
    method: str,
    O: np.ndarray,
    Z: np.ndarray,
    tau_true: float,
    fit_incr_peak_mb_share: Optional[Any],
    fit_incr_lock: Optional[Any],
) -> None:
    """Child process boundary: estimator must always send one dict."""
    try:
        conn.send(
            _run_measurement_core(
                method,
                O,
                Z,
                tau_true,
                fit_incr_peak_mb_share=fit_incr_peak_mb_share,
                fit_incr_lock=fit_incr_lock,
            )
        )
    except Exception as exc:
        conn.send(
            {
                "time_s": float("nan"),
                "rss_fit_peak_mb": float("nan"),
                "tau_hat": float("nan"),
                "relative_error": float("nan"),
                "status": "error",
                "error_detail": repr(exc),
            }
        )
    conn.close()


def run_single_trial_bounded(
    method: str,
    O: np.ndarray,
    Z: np.ndarray,
    tau_true: float,
    *,
    timeout_s: Optional[float],
    memory_limit_mb: Optional[float],
    terminate_grace_s: float = 15.0,
) -> Dict[str, Any]:
    """
    Spawn one trial. Parent enforces ``--memory-mb`` against **incremental** RSS during
    the estimator (aligned with ``rss_fit_peak_mb``), via shared state updated in the worker.
    """
    O = np.asarray(O, dtype=float)
    Z = np.asarray(Z, dtype=float)

    has_timeout = timeout_s is not None
    enforce_mem_mb = memory_limit_mb is not None and float(memory_limit_mb) > 0

    ctx = mp.get_context("spawn")

    if enforce_mem_mb:
        fit_incr_peak_mb_share = ctx.Value("d", 0.0)
        fit_incr_lock = ctx.Lock()
    else:
        fit_incr_peak_mb_share = None
        fit_incr_lock = None

    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_trial_worker,
        args=(child_conn, method, O, Z, float(tau_true), fit_incr_peak_mb_share, fit_incr_lock),
    )
    proc.start()
    child_conn.close()

    wall_start = time.monotonic()
    kill_reason: Optional[str] = None

    while proc.is_alive():
        if enforce_mem_mb:
            if float(fit_incr_peak_mb_share.value) > float(memory_limit_mb):
                kill_reason = "rss"
                break
        if has_timeout and time.monotonic() - wall_start >= float(timeout_s):
            kill_reason = "timeout"
            break
        proc.join(timeout=RSS_POLL_S)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=terminate_grace_s)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5.0)

    nan_mem = {"time_s": float("nan"), "rss_fit_peak_mb": float("nan")}
    nan_payload = {**nan_mem, "tau_hat": float("nan"), "relative_error": float("nan")}

    if kill_reason == "timeout":
        out: Dict[str, Any] = {
            **nan_payload,
            "status": "timeout",
            "error_detail": f"Exceeded wall-clock timeout ({timeout_s} s)",
        }
    elif kill_reason == "rss":
        out = {
            **nan_payload,
            "status": "memory_limit",
            "error_detail": (
                f"Exceeded rss_fit_peak budget (~{float(memory_limit_mb):.1f} MiB ΔRSS during estimate)"
            ),
        }
    else:
        try:
            out = parent_conn.recv()
        except EOFError:
            out = {
                **nan_payload,
                "status": "error",
                "error_detail": "Trial subprocess exited without a result pipe message",
            }
    parent_conn.close()
    return out


def run_load_test(
    N_sizes: Sequence[int] = DEFAULT_N,
    T_sizes: Sequence[int] = DEFAULT_T,
    methods: Optional[List[str]] = None,
    n_reps: int = 3,
    rank: int = 3,
    noise_scale: float = 1.0,
    treatment_pattern: str = "Block",
    treatment_level: float = 0.1,
    seed: int = 0,
    *,
    timeout_s: Optional[float] = None,
    memory_limit_mb: Optional[float] = None,
    verbose: bool = True,
    monotone_prune: bool = True,
) -> pd.DataFrame:
    """
    For each grid cell ``(N, T)`` we draw ``n_reps`` synthetic panels using
    seeds ``seed + cell_flat_idx * _SEED_CELL_STRIDE + trial`` (``trial`` ∈
    ``0 … n_reps-1``), where ``cell_flat_idx`` is the cell's index in ``(n, t)`` nested
    order over ``N_sizes × T_sizes`` (including skipped cells, so RNG matches a full-grid
    numbering). For a fixed rep, **all methods** share the same ``O, Z``, ``tau_true``.

    With ``monotone_prune=True`` (default): **per method**, after finishing ``(n0, t0)`` with **every**
    rep non-``ok`` for that method, record an anchor and skip **that method** on any later ``(n, t)``
    with ``n >= n0`` **and** ``t >= t0`` aside from `(n0, t0)` itself. Other methods keep running.
    Skipped ``(N,T,method,trial)`` combinations produce no rows.

    Relative error compares ``tau_hat`` to ``tau_true`` from ``generate``:
    ``|tau_hat − tau_true| / max(|tau_true|, eps)``.

    Persisted RSS is ``rss_fit_peak_mb`` only (ΔRSS while estimating). Trial rows include
    ``trial_seed`` (RNG seed passed to ``generate``), ``tau_hat``, ``time_s``, ``status``,
    ``error_detail``.
    """
    if methods is None:
        methods = list(DEFAULT_METHODS)

    rows: List[Dict[str, Any]] = []
    n_methods = len(methods)
    n_cells = sum(1 for _ in ((nx, ty) for nx in N_sizes for ty in T_sizes))
    planned_trials = n_cells * n_methods * n_reps

    prune_anchor_by_method: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    done = 0
    cell_flat_idx = 0
    n_prune_skips = 0

    for N in N_sizes:
        for T in T_sizes:
            attempts: defaultdict[str, int] = defaultdict(int)
            saw_ok: set = set()

            for trial in range(n_reps):
                trial_seed = int(seed) + cell_flat_idx * _SEED_CELL_STRIDE + trial
                O, Z, tau_true = generate(
                    N=N,
                    T=T,
                    rank=rank,
                    noise_scale=noise_scale,
                    treatment_pattern=treatment_pattern,
                    treatment_level=treatment_level,
                    seed=trial_seed,
                )

                for method in methods:
                    if monotone_prune and n_reps > 0 and _method_prune_should_skip(N, T, prune_anchor_by_method[method]):
                        n_prune_skips += 1
                        if verbose:
                            print(
                                f"  SKIP  N={N:5d} T={T:5d} tr={trial} {method:<24s}  "
                                "(monotone prune: N,T >= that method's failing anchor)",
                            )
                        continue

                    res = run_single_trial_bounded(
                        method,
                        O,
                        Z,
                        float(tau_true),
                        timeout_s=timeout_s,
                        memory_limit_mb=memory_limit_mb,
                    )
                    attempts[method] += 1
                    if res["status"] == "ok":
                        saw_ok.add(method)

                    rows.append(
                        {
                            "N": int(N),
                            "T": int(T),
                            "method": method,
                            "trial": trial,
                            "trial_seed": trial_seed,
                            "tau_true": float(tau_true),
                            "time_s": res["time_s"],
                            "rss_fit_peak_mb": res["rss_fit_peak_mb"],
                            "tau_hat": res["tau_hat"],
                            "relative_error": res["relative_error"],
                            "status": res["status"],
                            "error_detail": res["error_detail"],
                        }
                    )
                    done += 1
                    if verbose:
                        print(
                            f"  [{done:4d}/{planned_trials}] N={N:5d} T={T:5d} "
                            f"tr={trial} {method:<24s}  {_verbose_trial_suffix(res)}",
                        )

            if monotone_prune:
                for m in methods:
                    if attempts[m] > 0 and m not in saw_ok:
                        prune_anchor_by_method[m].append((int(N), int(T)))
                        if verbose:
                            print(
                                f"  PRUNE_ANCHOR method={m}  N={int(N):5d} T={int(T):5d}  "
                                "(all reps non-ok for this method; pruning N,T cone for this method)",
                            )

            cell_flat_idx += 1

    if verbose and monotone_prune:
        nanchors = sum(len(v) for v in prune_anchor_by_method.values())
        print(
            f"[monotone_prune] anchors recorded across methods: {nanchors}; "
            f"(method,reps) skips before subprocess: {n_prune_skips}",
            flush=True,
        )

    return pd.DataFrame(rows)


def aggregate_trials(df_trials: pd.DataFrame) -> pd.DataFrame:
    def std_safe(series: pd.Series) -> float:
        x = series.dropna().values.astype(float)
        return float("nan") if x.size < 2 else float(np.std(x))

    rows: List[Dict[str, Any]] = []
    for (n, t, method), g in df_trials.groupby(["N", "T", "method"], sort=True):
        rows.append(
            {
                "N": int(n),
                "T": int(t),
                "method": method,
                "tau_true": float(np.nanmean(g["tau_true"].values.astype(float))),
                "time_s": float(np.nanmean(g["time_s"].values.astype(float))),
                "time_std": std_safe(g["time_s"]),
                "rss_fit_peak_mb": float(np.nanmean(g["rss_fit_peak_mb"].values.astype(float))),
                "rss_fit_peak_std": std_safe(g["rss_fit_peak_mb"]),
                "tau_hat": float(np.nanmean(g["tau_hat"].values.astype(float))),
                "relative_error": float(np.nanmean(g["relative_error"].values.astype(float))),
                "relative_error_std": std_safe(g["relative_error"]),
                "n_ok": int((g["status"] == "ok").sum()),
                "n_timeout": int((g["status"] == "timeout").sum()),
                "n_memory_limit": int((g["status"] == "memory_limit").sum()),
                "n_err": int((g["status"] == "error").sum()),
                "n_trials": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_estimator(df_agg: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(DEFAULT_METHODS)}
    rows: List[Dict[str, Any]] = []

    for method in df_agg["method"].unique():
        sub = df_agg[df_agg["method"] == method]
        nt = sub["N"] * sub["T"]
        panel_ok_mask = (
            (sub["n_timeout"] == 0)
            & (sub["n_memory_limit"] == 0)
            & (sub["n_err"] == 0)
            & (sub["n_ok"].astype(float) >= sub["n_trials"])
        )

        median_mask = sub["n_ok"] > 0
        if median_mask.any():
            sub_ok = sub.loc[median_mask]
            median_time = float(np.nanmedian(sub_ok["time_s"].astype(float)))
            median_rss_fit = float(np.nanmedian(sub_ok["rss_fit_peak_mb"].astype(float)))
            median_err = float(np.nanmedian(sub_ok["relative_error"].astype(float)))
        else:
            median_time = median_rss_fit = median_err = float("nan")

        if panel_ok_mask.any():
            idx = nt[panel_ok_mask].idxmax()
            row = sub.loc[idx]
            max_n, max_t = int(row["N"]), int(row["T"])
            max_nt = max_n * max_t
        else:
            max_n = max_t = max_nt = float("nan")

        rows.append(
            {
                "method": method,
                "max_NT_all_trials_ok": max_nt if np.isfinite(max_nt) else None,
                "max_N_when_all_trials_ok": max_n if np.isfinite(max_n) else None,
                "max_T_when_all_trials_ok": max_t if np.isfinite(max_t) else None,
                "median_time_s_across_cells": median_time if np.isfinite(median_time) else None,
                "median_rss_fit_peak_mb_across_cells": median_rss_fit if np.isfinite(median_rss_fit) else None,
                "median_relative_error_across_cells": median_err if np.isfinite(median_err) else None,
            }
        )

    df_summary = pd.DataFrame(rows).sort_values(
        by="method",
        key=lambda series: series.map(lambda m: order.get(m, 999)),
    )
    df_summary.reset_index(drop=True, inplace=True)
    return df_summary


def _pivot_ordered(
    df: pd.DataFrame,
    *,
    metric: str,
    methods_order: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    pivots: Dict[str, pd.DataFrame] = {}
    for m in methods_order:
        chunk = df[df["method"] == m]
        pivot = chunk.pivot_table(index="N", columns="T", values=metric, aggfunc="first")
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot.reindex(columns=sorted(pivot.columns))
        pivots[m] = pivot
    return pivots


def _aligned_int_grid(ptab: pd.DataFrame, counts: pd.DataFrame) -> np.ndarray:
    aligned = counts.reindex(index=ptab.index, columns=ptab.columns)
    return np.nan_to_num(aligned.values, nan=0.0).astype(int)


def _draw_load_test_failure_overlays(
    ax: plt.Axes,
    subplot_col: int,
    n_timeout: np.ndarray,
    n_memlim: np.ndarray,
    n_err: np.ndarray,
) -> None:
    """
    Overlay white cells so timeout vs RSS-cap vs other failures read across panels.

    * Time ``(0)``: dotted white if timeout; solid white if only RSS-cap trials.
    * RSS ``(1)``: dotted if RSS cap; solid white if only timeouts.
    * Relative error ``(2)``: solid white if timeout or RSS-cap; dotted white if-only
      **other** trial errors (status ``error``); no overlay if ``n_ok`` only.
    """
    n_rows, n_cols = n_timeout.shape

    def add_cell(c: int, r: int, *, hatched: bool) -> None:
        kw: Dict[str, Any] = dict(
            xy=(c - 0.5, r - 0.5),
            width=1.0,
            height=1.0,
            facecolor="white",
            zorder=10,
        )
        if hatched:
            kw["edgecolor"] = "0.45"
            kw["hatch"] = "..."
            kw["linewidth"] = 0.0
        else:
            kw["edgecolor"] = "none"
            kw["linewidth"] = 0
        ax.add_patch(Rectangle(**kw))

    for r in range(n_rows):
        for c in range(n_cols):
            nt, nm, ne = int(n_timeout[r, c]), int(n_memlim[r, c]), int(n_err[r, c])
            if subplot_col == 0:
                if nt > 0:
                    add_cell(c, r, hatched=True)
                elif nm > 0:
                    add_cell(c, r, hatched=False)
            elif subplot_col == 1:
                if nm > 0:
                    add_cell(c, r, hatched=True)
                elif nt > 0:
                    add_cell(c, r, hatched=False)
            else:
                if nt > 0 or nm > 0:
                    add_cell(c, r, hatched=False)
                elif ne > 0:
                    add_cell(c, r, hatched=True)


def plot_load_test_heatmap_figure(
    df_agg: pd.DataFrame,
    methods: Sequence[str],
    *,
    vmin_time: Optional[float] = None,
    vmax_time: Optional[float] = None,
    vmin_rss: Optional[float] = None,
    vmax_rss: Optional[float] = None,
    vmax_rel_err: Optional[float] = None,
) -> plt.Figure:
    methods_list = list(methods)
    nrows = len(methods_list)
    fig, axes = plt.subplots(nrows, 3, figsize=(10.5, 2.0 * nrows + 1.0))
    if nrows == 1:
        axes = np.array([axes])
    colnames = ["Wall time (median, s)", "RSS fit peak (median, MiB)", "Relative ATT error (median)"]

    time_pv = _pivot_ordered(df_agg, metric="time_s", methods_order=methods_list)
    rss_pv = _pivot_ordered(df_agg, metric="rss_fit_peak_mb", methods_order=methods_list)
    err_pv = _pivot_ordered(df_agg, metric="relative_error", methods_order=methods_list)
    timeout_pv = _pivot_ordered(df_agg, metric="n_timeout", methods_order=methods_list)
    mem_pv = _pivot_ordered(df_agg, metric="n_memory_limit", methods_order=methods_list)
    nerr_pv = _pivot_ordered(df_agg, metric="n_err", methods_order=methods_list)

    t_vals = df_agg["time_s"].values.astype(float)
    r_vals = df_agg["rss_fit_peak_mb"].values.astype(float)
    e_vals = df_agg["relative_error"].replace([np.inf, -np.inf], np.nan).values.astype(float)

    vmin_time = float(np.nanpercentile(t_vals, 5)) if vmin_time is None else vmin_time
    vmax_time = float(np.nanpercentile(t_vals, 98)) if vmax_time is None else vmax_time
    vmin_rss = float(np.nanpercentile(r_vals, 5)) if vmin_rss is None else vmin_rss
    vmax_rss = float(np.nanpercentile(r_vals, 98)) if vmax_rss is None else vmax_rss

    if vmax_rel_err is None:
        fin = e_vals[np.isfinite(e_vals)]
        vmax_rel_err = float(np.nanpercentile(fin, 95)) if fin.size > 0 else 1.0

    for i, method in enumerate(methods_list):
        grids = [(time_pv, plt.cm.viridis, vmin_time, vmax_time), (rss_pv, plt.cm.plasma, vmin_rss, vmax_rss), (err_pv, plt.cm.magma, 0.0, vmax_rel_err)]
        for j, (pmap, cmap, v0, v1) in enumerate(grids):
            ax = axes[i, j]
            mat = pmap[method].values.astype(float)
            masked = np.ma.array(mat, mask=np.isnan(mat))
            im = ax.imshow(masked, aspect="auto", origin="upper", cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            ptab = pmap[method]
            nt_m = _aligned_int_grid(ptab, timeout_pv[method])
            nl_m = _aligned_int_grid(ptab, mem_pv[method])
            ne_m = _aligned_int_grid(ptab, nerr_pv[method])
            _draw_load_test_failure_overlays(ax, j, nt_m, nl_m, ne_m)
            ax.set_xticks(range(len(ptab.columns)))
            ax.set_xticklabels([str(c) for c in ptab.columns], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(ptab.index)))
            ax.set_yticklabels([str(r) for r in ptab.index], fontsize=8)
            if i == 0:
                ax.set_title(colnames[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(method, fontsize=9, rotation=25, ha="right")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Load test (Windows): median time, rss_fit_peak_mb, relative error over trials", fontsize=12, y=1.01)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.text(
        0.5,
        0.008,
        "Relative error: dotted hatch = trial error(s), no timeout/RSS-cap for that cell. "
        "Dotted elsewhere = timeout (wall clock) / RSS-cap (RSS). Solid white masks cross-metrics for timeout/RSS-cap "
        "and relative error when timed out or capped.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    return fig


def print_load_test_table(df_agg: pd.DataFrame) -> None:
    tabs = [
        ("Wall-clock time (median)", "time_s", "s"),
        ("RSS fit peak (median)", "rss_fit_peak_mb", "MiB"),
        ("Relative ATT error (median)", "relative_error", ""),
    ]
    for title, col, unit in tabs:
        pivot = df_agg.pivot_table(index=["N", "T"], columns="method", values=col)
        pivot.columns.name = None
        pivot.index.names = ["N", "T"]
        u = f" ({unit})" if unit else ""
        print(f"\n{'='*60}\n  {title}{u}\n{'=' * 60}")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
    print()


def save_load_test_bundle(
    df_trials: pd.DataFrame,
    output_dir: Optional[Path] = None,
    prefix: str = "load_test",
    *,
    save_plots: bool = True,
) -> Dict[str, Path]:
    out = Path(output_dir) if output_dir else _RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    base = prefix

    df_agg = aggregate_trials(df_trials)
    df_summary = summarize_by_estimator(df_agg)

    paths: Dict[str, Path] = {
        "trials": out / f"{base}_trials.csv",
        "cells": out / f"{base}_cells.csv",
        "summary": out / f"{base}_summary.csv",
    }
    df_trials.to_csv(paths["trials"], index=False)
    df_agg.to_csv(paths["cells"], index=False)
    df_summary.to_csv(paths["summary"], index=False)

    if save_plots:
        order = [m for m in DEFAULT_METHODS if m in set(df_agg["method"].unique())]
        order += sorted(set(df_agg["method"].unique()) - set(order))
        fig = plot_load_test_heatmap_figure(df_agg, order)
        paths["heatmaps"] = out / f"{base}_heatmaps.png"
        fig.savefig(paths["heatmaps"], dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Load test bundle saved under {out}:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    return paths


def save_load_test(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    prefix: str = "load_test",
    *,
    save_plots: bool = True,
) -> Path:
    """If ``trial`` column present, write trials/cells/summary(+PNG); else one legacy CSV."""
    if "trial" in df.columns:
        return save_load_test_bundle(df, output_dir=output_dir, prefix=prefix, save_plots=save_plots)["cells"]

    out = Path(output_dir) if output_dir else _RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prefix}.csv"
    df.to_csv(path, index=False)
    print(f"Load test saved → {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parser = argparse.ArgumentParser(
        description="Load test (Windows): wall time, rss_fit_peak_mb (during estimate), ATT error.",
    )
    parser.add_argument("--N", nargs="+", type=int, default=list(DEFAULT_N), metavar="N")
    parser.add_argument("--T", nargs="+", type=int, default=list(DEFAULT_T), metavar="T")
    parser.add_argument("--methods", nargs="+", default=None, metavar="METHOD")
    parser.add_argument("--n-reps", type=int, default=3, metavar="K")
    parser.add_argument("--treatment-pattern", default="Block", choices=["IID", "Block", "Staggered", "Adaptive"])
    parser.add_argument("--treatment-level", type=float, default=0.1, metavar="LVL")
    parser.add_argument("--timeout", type=float, default=None, metavar="SEC")
    parser.add_argument(
        "--memory-mb",
        type=float,
        default=None,
        metavar="MB",
        help="Terminate trial if rss_fit_peak_mb (ΔRSS during estimate) exceeds this cap (approximate polling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed; full-grid cell flat index ×1e6 + trial seeds ``generate`` (see ``run_load_test`` docstring)",
    )
    parser.add_argument(
        "--no-monotone-prune",
        action="store_true",
        help=(
            "Run all (N,T,method) trials regardless of monotone pruning; "
            "default skips (per method) runs dominated in (N,T) after that method had all reps "
            "non-ok at some anchor (n0, t0)."
        ),
    )
    parser.add_argument("--out-dir", default=None, metavar="DIR")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args(argv)

    df_trials = run_load_test(
        N_sizes=args.N,
        T_sizes=args.T,
        methods=args.methods,
        n_reps=args.n_reps,
        treatment_pattern=args.treatment_pattern,
        treatment_level=args.treatment_level,
        seed=args.seed,
        timeout_s=args.timeout,
        memory_limit_mb=args.memory_mb,
        monotone_prune=not args.no_monotone_prune,
    )

    df_agg = aggregate_trials(df_trials)
    print_load_test_table(df_agg)
    print("\n=== Summary per estimator ===\n" + summarize_by_estimator(df_agg).to_string(index=False))

    if not args.no_save:
        save_load_test_bundle(df_trials, output_dir=args.out_dir, save_plots=not args.no_plot)

    return df_trials, df_agg


if __name__ == "__main__":
    mp.freeze_support()
    main()
