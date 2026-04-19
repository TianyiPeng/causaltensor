"""
Load tests for all estimators: scalability across N×T panel sizes.

Sweeps a grid of (N, T) panel sizes using fully synthetic data from
``causaltensor.synthetic.generate``, records wall-clock time and peak memory
usage for each method, and saves a CSV report.

CLI usage
---------
::

    poetry run python -m causaltensor.analysis.load_tests
    poetry run python -m causaltensor.analysis.load_tests \\
        --N 50 100 200 --T 50 100 200 \\
        --methods DID SDID MC_NNM_CV \\
        --n-reps 3 --out-dir results/load_tests

Programmatic usage
------------------
>>> from causaltensor.analysis.load_tests import run_load_test
>>> df = run_load_test(N_sizes=[50, 100], T_sizes=[50, 100])
>>> df[["N", "T", "method", "time_s", "memory_mb"]]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causaltensor.synthetic import generate
from causaltensor.utils.common import get_tau_from_method_with_error

DEFAULT_N: Tuple[int, ...] = (20, 50, 100)
DEFAULT_T: Tuple[int, ...] = (20, 50, 100)

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


def _time_and_memory(
    method: str,
    O: np.ndarray,
    Z: np.ndarray,
) -> Tuple[float, float, Optional[str]]:
    """Run one estimator and return (wall_time_s, peak_memory_mb, error_or_None)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    _, err = get_tau_from_method_with_error(method, O, Z)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / 1024**2, err  # bytes → MB


def run_load_test(
    N_sizes: Sequence[int] = DEFAULT_N,
    T_sizes: Sequence[int] = DEFAULT_T,
    methods: Optional[List[str]] = None,
    n_reps: int = 1,
    rank: int = 3,
    noise_scale: float = 1.0,
    treatment_pattern: str = "Block",
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Measure estimator runtime and memory across an N×T grid.

    For each ``(N, T)`` pair a synthetic panel is generated and each estimator
    is called ``n_reps`` times; time and memory are averaged over reps.

    Parameters
    ----------
    N_sizes : sequence of int
        Number of units to sweep.
    T_sizes : sequence of int
        Number of time periods to sweep.
    methods : list of str or None
        Estimators to test. Defaults to all 7 available methods.
    n_reps : int, default 1
        Repeated calls per (N, T, method) cell. Use 3–5 for stable timing.
    rank : int, default 3
        Rank of the synthetic baseline matrix.
    noise_scale : float, default 1.0
        Gaussian noise standard deviation.
    treatment_pattern : str, default 'Block'
        Treatment pattern used for synthetic data generation.
    seed : int, default 0
        Base random seed; each (N, T) cell uses ``seed + idx``.
    verbose : bool, default True
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame
        One row per (N, T, method) with columns:
        ``N, T, method, time_s, time_std, memory_mb, memory_std, error``.
    """
    if methods is None:
        methods = list(DEFAULT_METHODS)

    results = []
    total = len(list(N_sizes)) * len(list(T_sizes)) * len(methods)
    done = 0

    for cell_idx, (N, T) in enumerate(
        (n, t) for n in N_sizes for t in T_sizes
    ):
        O, Z, _ = generate(
            N=N, T=T, rank=rank,
            noise_scale=noise_scale,
            treatment_pattern=treatment_pattern,
            seed=seed + cell_idx,
        )

        for method in methods:
            times, mems = [], []
            last_err = None

            for _ in range(n_reps):
                t_s, mem_mb, err = _time_and_memory(method, O, Z)
                times.append(t_s)
                mems.append(mem_mb)
                if err:
                    last_err = err

            done += 1
            row = dict(
                N=N, T=T, method=method,
                time_s=float(np.mean(times)),
                time_std=float(np.std(times)),
                memory_mb=float(np.mean(mems)),
                memory_std=float(np.std(mems)),
                error=last_err,
            )
            results.append(row)

            if verbose:
                status = f"ERR: {last_err[:40]}" if last_err else f"{row['time_s']:.3f}s  {row['memory_mb']:.1f}MB"
                print(f"  [{done:3d}/{total}]  N={N:4d}  T={T:4d}  {method:<22s}  {status}")

    return pd.DataFrame(results)


def print_load_test_table(df: pd.DataFrame) -> None:
    """
    Print pivot tables of wall-clock time and peak memory.
    Rows = (N, T), columns = methods.
    """
    for metric, col, unit in [("Wall-clock time", "time_s", "s"),
                               ("Peak memory",    "memory_mb", "MB")]:
        pivot = df.pivot_table(index=["N", "T"], columns="method", values=col)
        pivot.columns.name = None
        pivot.index.names = ["N", "T"]
        print(f"\n{'='*60}")
        print(f"  {metric} ({unit})")
        print("="*60)
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print()


def save_load_test(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    prefix: str = "load_test",
) -> Path:
    """Save load test results as CSV and return the path."""
    out = Path(output_dir) if output_dir else _RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out / f"{prefix}_{ts}.csv"
    df.to_csv(path, index=False)
    print(f"Load test saved → {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(
        description="Load test: time & memory per method across N×T grid."
    )
    parser.add_argument(
        "--N", nargs="+", type=int, default=list(DEFAULT_N), metavar="N",
        help="Unit counts to sweep (default: 20 50 100)",
    )
    parser.add_argument(
        "--T", nargs="+", type=int, default=list(DEFAULT_T), metavar="T",
        help="Time counts to sweep (default: 20 50 100)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None, metavar="METHOD",
        help="Estimators to test (default: all 7)",
    )
    parser.add_argument(
        "--n-reps", type=int, default=1, metavar="K",
        help="Repeated calls per cell (default: 1)",
    )
    parser.add_argument(
        "--treatment-pattern", default="Block",
        choices=["IID", "Block", "Staggered", "Adaptive"],
        help="Synthetic treatment pattern (default: Block)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base random seed (default: 0)",
    )
    parser.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Directory for the CSV report (default: analysis/results/load_tests/)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results but do not write CSV",
    )
    args = parser.parse_args(argv)

    df = run_load_test(
        N_sizes=args.N,
        T_sizes=args.T,
        methods=args.methods,
        n_reps=args.n_reps,
        treatment_pattern=args.treatment_pattern,
        seed=args.seed,
    )

    print_load_test_table(df)

    if not args.no_save:
        save_load_test(df, output_dir=args.out_dir)

    return df


if __name__ == "__main__":
    main()
