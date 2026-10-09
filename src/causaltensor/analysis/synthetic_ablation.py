"""
Synthetic DGP ablation: rank, unit heterogeneity, time heterogeneity, and noise.

For each axis, mean relative error |tau_hat - tau_star| / |tau_star| is averaged
over Monte Carlo trials; one subplot per axis (1×4 figure), lines = estimators.

CLI::

    python -m causaltensor.analysis.synthetic_ablation
    python -m causaltensor.analysis.synthetic_ablation --pattern Staggered --trials 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from causaltensor.semi_synthetic.aa_test import (
    DEFAULT_METHODS,
    VALID_PATTERNS,
)
from causaltensor.synthetic.dgp import generate
from causaltensor.utils.common import get_tau_from_method

logger = logging.getLogger(__name__)

_RESULTS_SUBDIR = "synthetic_ablation"

# Grids from spec (fixed N=200, T=50, trials=30 defaults in CLI)
RANK_GRID: Tuple[int, ...] = (2, 3, 5, 8, 12, 20)
SIGMA_UNIT_GRID: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0, 2.0)
SIGMA_TIME_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0)
NOISE_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# Fixed DGP knobs when an axis is not being swept (matches synthetic defaults).
_BASE_RANK = 3
_BASE_SIGMA_UNIT = 0.8
_BASE_SIGMA_TIME = 0.2
_BASE_NOISE = 1.0


def _held_defaults_caption(axis_key: str) -> str:
    """One-line math caption: the three DGP knobs held fixed while `axis_key` is swept."""
    if axis_key == "rank":
        return (
            rf"fixed: $\mathrm{{sd}}(M)=1$, "
            rf"$\delta$={_BASE_SIGMA_UNIT:g}, "
            rf"$\eta$={_BASE_SIGMA_TIME:g}, $\sigma$={_BASE_NOISE:g}"
        )
    if axis_key == "sigma_unit":
        return (
            rf"fixed: $r$={_BASE_RANK}, $\eta$={_BASE_SIGMA_TIME:g}, "
            rf"$\sigma$={_BASE_NOISE:g}"
        )
    if axis_key == "sigma_time":
        return (
            rf"fixed: $r$={_BASE_RANK}, $\delta$={_BASE_SIGMA_UNIT:g}, "
            rf"$\sigma$={_BASE_NOISE:g}"
        )
    if axis_key == "noise":
        return (
            rf"fixed: $r$={_BASE_RANK}, $\delta$={_BASE_SIGMA_UNIT:g}, "
            rf"$\eta$={_BASE_SIGMA_TIME:g}"
        )
    raise ValueError(f"unknown axis_key {axis_key!r}")


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / _RESULTS_SUBDIR


def _methods_dict(
    pattern: str,
    methods: Optional[Sequence[str]],
) -> Dict[str, List[str]]:
    if methods is None:
        return {k: v for k, v in DEFAULT_METHODS.items() if pattern in v}
    out: Dict[str, List[str]] = {}
    for m in methods:
        if m not in DEFAULT_METHODS:
            raise ValueError(f"Unknown method {m!r}. Valid: {tuple(DEFAULT_METHODS.keys())}")
        if pattern not in DEFAULT_METHODS[m]:
            logger.warning("Skipping %s: not defined for pattern %s", m, pattern)
            continue
        out[m] = [pattern]
    if not out:
        raise ValueError(f"No methods valid for pattern {pattern!r}")
    return out


def _relative_error(tau_star: float, tau_hat: float) -> float:
    if tau_star == 0 or np.isnan(tau_hat):
        return float("nan")
    return float(abs(tau_star - tau_hat) / abs(tau_star))


def run_ablation_grid(
    *,
    N: int = 200,
    T: int = 50,
    pattern: str = "Block",
    treatment_level: float = 0.2,
    trials: int = 30,
    seed: int = 0,
    methods: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all four ablation axes and return a long-form DataFrame of per-trial errors.
    """
    if pattern not in VALID_PATTERNS:
        raise ValueError(f"Unknown pattern {pattern!r}. Valid: {VALID_PATTERNS}")
    methods_dict = _methods_dict(pattern, methods)

    rows: List[Dict[str, Any]] = []

    axis_grids: List[Tuple[str, Tuple[Any, ...]]] = [
        ("rank", RANK_GRID),
        ("sigma_unit", SIGMA_UNIT_GRID),
        ("sigma_time", SIGMA_TIME_GRID),
        ("noise", NOISE_GRID),
    ]

    for axis_id, (axis_key, grid) in enumerate(axis_grids):
        for p_idx, param in enumerate(grid):
            rank = int(param) if axis_key == "rank" else _BASE_RANK
            sigma_u = float(param) if axis_key == "sigma_unit" else _BASE_SIGMA_UNIT
            sigma_t = float(param) if axis_key == "sigma_time" else _BASE_SIGMA_TIME
            noise_s = float(param) if axis_key == "noise" else _BASE_NOISE

            for trial in range(trials):
                trial_seed = np.random.SeedSequence(
                    [seed, axis_id, p_idx, trial]
                ).generate_state(1)[0]

                O, Z, tau_star = generate(
                    N,
                    T,
                    rank=rank,
                    noise_scale=noise_s,
                    treatment_pattern=pattern,
                    treatment_level=treatment_level,
                    sigma_unit_scale=sigma_u,
                    sigma_time_scale=sigma_t,
                    seed=int(trial_seed),
                    normalize_M=(axis_key == "rank"),
                )
                Zf = np.asarray(Z, dtype=float)

                for method_name in methods_dict:
                    tau_hat = get_tau_from_method(method_name, O, Zf)
                    err = _relative_error(tau_star, tau_hat)
                    rows.append(
                        {
                            "axis": axis_key,
                            "axis_value": float(param),
                            "method": method_name,
                            "trial": trial,
                            "tau_star": tau_star,
                            "tau_hat": tau_hat,
                            "error": err,
                            "rank": rank,
                            "sigma_unit_scale": sigma_u,
                            "sigma_time_scale": sigma_t,
                            "noise_scale": noise_s,
                        }
                    )

            if verbose:
                logger.info(
                    "Finished axis=%s value=%s (%s/%s params)",
                    axis_key,
                    param,
                    p_idx + 1,
                    len(grid),
                )

    return pd.DataFrame(rows)

def run_and_save(
    output_dir: Optional[Path] = None,
    *,
    N: int = 200,
    T: int = 50,
    pattern: str = "Block",
    treatment_level: float = 0.2,
    trials: int = 30,
    seed: int = 0,
    methods: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    out = Path(output_dir) if output_dir else default_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    df = run_ablation_grid(
        N=N,
        T=T,
        pattern=pattern,
        treatment_level=treatment_level,
        trials=trials,
        seed=seed,
        methods=methods,
        verbose=verbose,
    )
    stem = f"ablation_N{N}_T{T}_{pattern}_trials{trials}"
    path_csv = out / f"{stem}_trials.csv"
    df.to_csv(path_csv, index=False)
    logger.info("Wrote %s", path_csv)

    agg = (
        df.groupby(["axis", "axis_value", "method"])["error"]
        .agg(mean_error="mean", std_error="std")
        .reset_index()
    )
    path_agg = out / f"{stem}_summary.csv"
    agg.to_csv(path_agg, index=False)
    logger.info("Wrote %s", path_agg)

    return {
        "df": df,
        "summary": agg,
        "paths": {"trials": path_csv, "summary": path_agg},
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Synthetic ablation: rank, heterogeneity, noise (mean relative error by estimator)."
    )
    parser.add_argument(
        "--pattern",
        default="Block",
        choices=list(VALID_PATTERNS),
        help="Treatment assignment pattern (default: Block).",
    )
    parser.add_argument("--N", type=int, default=200, help="Units (default: 200).")
    parser.add_argument("--T", type=int, default=50, help="Periods (default: 50).")
    parser.add_argument("--trials", type=int, default=30, help="MC trials per grid point (default: 30).")
    parser.add_argument(
        "--treatment-level",
        type=float,
        default=0.2,
        help="tau* = treatment_level * mean(|M|) (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (default: 0).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Output folder (default: analysis/results/{_RESULTS_SUBDIR}/).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        choices=tuple(DEFAULT_METHODS.keys()),
        help="Estimator keys (default: all valid for the chosen pattern).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    return run_and_save(
        Path(args.out_dir) if args.out_dir else None,
        N=args.N,
        T=args.T,
        pattern=args.pattern,
        treatment_level=args.treatment_level,
        trials=args.trials,
        seed=args.seed,
        methods=args.methods,
        verbose=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = main()
    print("\nSaved:")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")
